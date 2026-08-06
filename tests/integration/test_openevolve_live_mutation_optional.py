"""Explicit one-call synthetic smoke gate; skipped in normal PR and CI runs."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from auto_researcher.agents.call_store import SQLiteAgentCallStore
from auto_researcher.agents.models import ModelCallConfig, ModelPricing
from auto_researcher.providers.anthropic import create_anthropic_client
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.live_models import (
    OpenEvolveModelBridgeContract,
    OpenEvolveModelCallContext,
    parse_live_mutation_approval,
)
from auto_researcher.search.openevolve.production_bridge import (
    DurableOpenEvolveModelBridge,
)
from auto_researcher.search.openevolve.upstream import (
    build_approved_live_upstream_runtime,
    default_adapter_contract,
    mutation_constraints,
)
from auto_researcher.search.openevolve.upstream_models import (
    ExecutorIsolationResult,
    HardenedExecutorPolicy,
)
from auto_researcher.tasks.synthetic import SyntheticEvolvableComponent

pytestmark = [
    pytest.mark.live_agent,
    pytest.mark.skipif(
        os.environ.get("AUTO_RESEARCHER_RUN_LIVE_OPENEVOLVE") != "1"
        or not os.environ.get("AUTO_RESEARCHER_LIVE_MUTATION_APPROVAL")
        or not os.environ.get("AUTO_RESEARCHER_EXECUTOR_POLICY_FILE")
        or not os.environ.get("AUTO_RESEARCHER_EXECUTOR_ISOLATION_FILE")
        or not os.environ.get("ANTHROPIC_API_KEY"),
        reason=(
            "requires explicit live OpenEvolve opt-in, protected approval, exact "
            "executor evidence, and protected provider credentials"
        ),
    ),
]


def test_one_approved_synthetic_mutation(tmp_path):
    approval_path = Path(os.environ["AUTO_RESEARCHER_LIVE_MUTATION_APPROVAL"])
    approval = parse_live_mutation_approval(
        yaml.safe_load(approval_path.read_text(encoding="utf-8"))
    )
    adapter_contract = default_adapter_contract(
        Path(__file__).parents[2] / "constraints/openevolve-0.3.2.lock"
    )
    assert approval.permitted_dataset_class == "synthetic"
    assert approval.maximum_model_calls == 1
    pricing = ModelPricing(
        version=approval.pricing_version,
        input_cost_per_million_tokens=float(os.environ["AUTO_RESEARCHER_INPUT_RATE"]),
        output_cost_per_million_tokens=float(os.environ["AUTO_RESEARCHER_OUTPUT_RATE"]),
        currency=approval.currency,
    )
    call_config = ModelCallConfig(
        provider=approval.provider,
        model_id=approval.model_id,
        temperature=0,
        maximum_output_tokens=approval.maximum_output_tokens,
        timeout_seconds=30,
        maximum_attempts=1,
        maximum_cost_per_call=approval.maximum_total_cost,
        pricing=pricing,
        prompt_version=approval.prompt_version,
    )
    bridge_contract = OpenEvolveModelBridgeContract.model_validate(
        {
            "mutation_operator_id": "pinned-upstream-openevolve",
            "mutation_operator_version": approval.mutation_operator_version,
            "maximum_input_bytes": approval.maximum_input_tokens * 4,
            "model_config": call_config,
        }
    )
    context = OpenEvolveModelCallContext(
        run_id=approval.run_id,
        thread_id="approved-live-smoke-thread",
        contract_id=approval.contract_id,
        contract_hash=approval.contract_hash,
        task_id=approval.task_id,
        task_version=approval.task_version,
        search_request_id="approved-live-smoke-search",
        generation=1,
        parent_candidate_id="synthetic-seed",
        component_id=approval.component_id,
        component_version=approval.component_version,
        component_interface_hash="0" * 64,
        adapter_id=approval.adapter_id,
        adapter_version=approval.adapter_version,
        adapter_identity_hash=payload_hash(adapter_contract),
        executor_policy_hash=approval.executor_policy_hash,
        image_digest=approval.image_digest,
        mutable_file=approval.mutable_file,
        model_budget_identity="approved-live-smoke-budget",
        maximum_model_calls=1,
        maximum_model_cost=approval.maximum_total_cost,
    )
    store = SQLiteAgentCallStore(tmp_path / "agent-calls.sqlite")
    prompt = (
        Path(__file__).parents[2]
        / "src/auto_researcher/prompts/openevolve/openevolve-mutation-prompt-v2.md"
    ).read_text(encoding="utf-8")
    bridge = DurableOpenEvolveModelBridge(
        contract=bridge_contract,
        context=context,
        approval=approval,
        store=store,
        provider_factory=lambda: create_anthropic_client(call_config),
        now=lambda: datetime.now(UTC),
        system_prompt=prompt,
    )
    executor_policy = HardenedExecutorPolicy.model_validate(
        yaml.safe_load(
            Path(os.environ["AUTO_RESEARCHER_EXECUTOR_POLICY_FILE"]).read_text(
                encoding="utf-8"
            )
        )
    )
    isolation = ExecutorIsolationResult.model_validate(
        yaml.safe_load(
            Path(os.environ["AUTO_RESEARCHER_EXECUTOR_ISOLATION_FILE"]).read_text(
                encoding="utf-8"
            )
        )
    )
    build_approved_live_upstream_runtime(
        adapter_contract,
        bridge,
        executor_policy,
        isolation,
        workspace_root=tmp_path / "hardened-workspace",
    )
    component = SyntheticEvolvableComponent().component_spec()
    result, _ = bridge.complete(
        {
            "protocol": "upstream-adapter-mutation-request-v2",
            "parent": {
                "id": "synthetic-seed",
                "code": component.seed_source,
                "generation": 0,
            },
            "mutable_file": component.mutable_file,
            "interface_contract": component.immutable_interface_contract,
            "maximum_source_bytes": component.maximum_source_bytes,
            "mutation_constraints": mutation_constraints(component).model_dump(
                mode="json"
            ),
        },
        "approved-live-smoke-mutation-1",
    )
    assert result["mutable_file"] == approval.mutable_file
    assert len({item.call_id for item in store.list_records()}) == 1
