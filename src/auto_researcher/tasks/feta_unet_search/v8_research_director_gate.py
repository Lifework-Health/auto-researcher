"""Paid, metadata-only Research Director shadow and durable replay gate for V8."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from auto_researcher.agents.call_store import SQLiteAgentCallStore
from auto_researcher.agents.live.research_director import LiveResearchDirectorAgent
from auto_researcher.agents.models import (
    AgentBudgetPolicy,
    ModelCallConfig,
    ResearchDirective,
    ResearchDirectorContext,
)
from auto_researcher.agents.research_director_shadow import (
    ResearchDirectorShadowPolicy,
    ResearchDirectorShadowReport,
    evaluate_shadow_directive,
)
from auto_researcher.cli import (
    _configured_search_type,
    _load_live_agents,
    _load_task_configuration,
    _load_yaml,
)
from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.graph.nodes.initialise import initialise_run
from auto_researcher.graph.nodes.supervisor import supervisor_prepare
from auto_researcher.providers.protocols import StructuredModelClient
from auto_researcher.runtime.dependencies import task_sqlite_dependencies
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks import TaskRuntimeContext, default_task_registry

V8_RESEARCH_DIRECTOR_GATE_SCHEMA = "feta-unet-v8-research-director-gate-v1"
V8_RESEARCH_DIRECTOR_POLICY_ID = "feta-unet-v8-operator-envelope-v1"
V8_RESEARCH_DIRECTOR_OPERATOR_LIMITS = {
    SearchType.DIRECT: 8,
    SearchType.OPTUNA: 26,
    SearchType.OPENEVOLVE: 10,
}
V8_RESEARCH_DIRECTOR_MAXIMUM_TOTAL_ALLOCATION = 44


class _ReplayTrapClient:
    """Provider-compatible trap proving that a completed call is replayed."""

    def __init__(self, *, provider: str, model_id: str) -> None:
        self.provider = provider
        self.model_id = model_id
        self.calls = 0

    def generate_structured(self, **_kwargs):
        self.calls += 1
        raise AssertionError("research_director_replay_called_provider")


def _shadow_policy(context: ResearchDirectorContext) -> ResearchDirectorShadowPolicy:
    return ResearchDirectorShadowPolicy(
        policy_id=V8_RESEARCH_DIRECTOR_POLICY_ID,
        allowed_operators=frozenset(V8_RESEARCH_DIRECTOR_OPERATOR_LIMITS),
        allowed_dimensions=frozenset(context.permitted_target_dimensions),
        maximum_allocation_by_operator=V8_RESEARCH_DIRECTOR_OPERATOR_LIMITS,
        maximum_total_allocation=V8_RESEARCH_DIRECTOR_MAXIMUM_TOTAL_ALLOCATION,
    )


def decide_shadow_and_replay(
    *,
    context: ResearchDirectorContext,
    client: StructuredModelClient,
    call_config: ModelCallConfig,
    budget_policy: AgentBudgetPolicy,
    agent_calls_path: Path,
    clock: Callable[[], datetime],
) -> tuple[
    ResearchDirective,
    ResearchDirectorShadowReport,
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], ...],
]:
    """Perform one bounded call and prove durable replay without dispatch."""

    first_store = SQLiteAgentCallStore(agent_calls_path)
    try:
        first_agent = LiveResearchDirectorAgent(
            client=client,
            call_config=call_config,
            budget_policy=budget_policy,
            call_store=first_store,
            clock=clock,
        )
        directive = first_agent.decide(context)
        first_telemetry = first_agent.consume_telemetry()
        if first_telemetry is None or first_telemetry.failed:
            raise ValueError("research_director_live_smoke_telemetry_invalid")
        shadow = evaluate_shadow_directive(directive, _shadow_policy(context))
        records = tuple(
            item.model_dump(mode="json")
            for item in first_store.list_records(context.run_id)
        )
    finally:
        first_store.close()

    trap = _ReplayTrapClient(
        provider=call_config.provider,
        model_id=call_config.model_id,
    )
    replay_store = SQLiteAgentCallStore(agent_calls_path)
    try:
        replay_agent = LiveResearchDirectorAgent(
            client=trap,
            call_config=call_config,
            budget_policy=budget_policy,
            call_store=replay_store,
            clock=clock,
        )
        replayed = replay_agent.decide(context)
        replay_telemetry = replay_agent.consume_telemetry()
    finally:
        replay_store.close()

    if replayed != directive:
        raise ValueError("research_director_replay_directive_mismatch")
    if trap.calls != 0:
        raise ValueError("research_director_replay_called_provider")
    if replay_telemetry is None or not replay_telemetry.replayed:
        raise ValueError("research_director_replay_telemetry_invalid")

    return (
        directive,
        shadow,
        first_telemetry.model_dump(mode="json"),
        replay_telemetry.model_dump(mode="json"),
        records,
    )


def run_v8_research_director_gate(
    *,
    task_config_path: Path,
    contract_path: Path,
    data_dir: Path,
    gate_dir: Path,
    run_id: str,
    client: StructuredModelClient | None = None,
    call_config: ModelCallConfig | None = None,
    budget_policy: AgentBudgetPolicy | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Run the non-dispatching V8 Director gate in a fresh directory."""

    task_config_path = task_config_path.expanduser().resolve()
    contract_path = contract_path.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    gate_dir = gate_dir.expanduser().resolve()
    if gate_dir.exists():
        raise ValueError("research_director_gate_directory_not_fresh")
    if not data_dir.is_dir():
        raise ValueError("research_director_gate_data_directory_missing")

    raw_config = _load_yaml(task_config_path)
    contract = ResearchContract.model_validate(_load_yaml(contract_path))
    experiment, runtime = _load_task_configuration(
        task_config_path,
        contract.task_id,
        contract.task_version,
    )
    if client is None or call_config is None or budget_policy is None:
        (
            _hypothesis_client,
            _planner_client,
            loaded_client,
            _hypothesis_config,
            _planner_config,
            loaded_config,
            loaded_policy,
            agent_mode,
        ) = _load_live_agents(raw_config)
        if agent_mode != "live" or loaded_client is None or loaded_config is None:
            raise ValueError("research_director_live_configuration_required")
        client = loaded_client
        call_config = loaded_config
        budget_policy = loaded_policy
    if call_config.model_id != "claude-opus-5":
        raise ValueError("research_director_v8_model_identity_mismatch")
    if call_config.effort != "xhigh" or dict(call_config.thinking or {}) != {
        "type": "adaptive"
    }:
        raise ValueError("research_director_v8_reasoning_configuration_mismatch")

    current_time = clock or (lambda: datetime.now(UTC))
    gate_dir.mkdir(parents=True, mode=0o700)
    control_dir = gate_dir / "control"
    control_dir.mkdir(mode=0o700)
    runtime_options = dict(runtime.get("options", {}))
    runtime_context = TaskRuntimeContext(
        run_id=run_id,
        data_dir=data_dir,
        output_dir=gate_dir / "output",
        workspace_dir=gate_dir / "workspace",
        environment=runtime.get("environment", {}),
        task_options=runtime_options,
    )
    task = default_task_registry().get(contract.task_id, contract.task_version)
    search_type = _configured_search_type(task_config_path)
    with task_sqlite_dependencies(
        task,
        runtime_context,
        contract,
        experiment,
        control_dir / "checkpoints.sqlite",
        control_dir / "provenance.sqlite",
        control_dir / "optuna.sqlite",
        control_dir / "unused-agent-calls.sqlite",
        control_dir / "knowledge.sqlite",
        search_type=search_type,
        clock=current_time,
    ) as dependencies:
        initial: dict[str, Any] = {
            "run_id": run_id,
            "thread_id": f"{run_id}-shadow",
            "contract": contract,
        }
        state = {**initial, **initialise_run(initial, dependencies)}
        state = {**state, **supervisor_prepare(state, dependencies)}
        reserve = float(
            runtime_options.get(
                "campaign_finalisation_reserve_seconds",
                contract.constraints["campaign_finalisation_reserve_seconds"],
            )
        )
        context = dependencies.agent_context_assembler.research_director_context(
            state,
            dependencies.task_agent_context,
            dependencies.search_capabilities,
            trigger="campaign_start",
            finalisation_reserve_seconds=reserve,
        )

    (
        directive,
        shadow,
        first_telemetry,
        replay_telemetry,
        records,
    ) = decide_shadow_and_replay(
        context=context,
        client=client,
        call_config=call_config,
        budget_policy=budget_policy,
        agent_calls_path=control_dir / "research-director-calls.sqlite",
        clock=current_time,
    )
    base = {
        "schema_version": V8_RESEARCH_DIRECTOR_GATE_SCHEMA,
        "run_id": run_id,
        "task_config_sha256": payload_hash(raw_config),
        "contract_sha256": payload_hash(contract),
        "research_director_evidence_manifest_sha256": runtime_options.get(
            "research_director_evidence_manifest_sha256"
        ),
        "model": {
            "provider": call_config.provider,
            "model_id": call_config.model_id,
            "prompt_version": call_config.prompt_version,
            "thinking": dict(call_config.thinking or {}),
            "effort": call_config.effort,
        },
        "context_hash": context.context_hash,
        "permitted_dimension_count": len(context.permitted_target_dimensions),
        "permitted_operators": [
            item.value for item in context.installed_search_capabilities
        ],
        "operator_limits": {
            key.value: value
            for key, value in V8_RESEARCH_DIRECTOR_OPERATOR_LIMITS.items()
        },
        "maximum_total_allocation": V8_RESEARCH_DIRECTOR_MAXIMUM_TOTAL_ALLOCATION,
        "directive": directive.model_dump(mode="json"),
        "shadow_report": shadow.model_dump(mode="json"),
        "first_call_telemetry": first_telemetry,
        "replay_telemetry": replay_telemetry,
        "durable_call_records": records,
        "replay_provider_calls": 0,
        "directive_replayed_exactly": True,
        "experiments_dispatched": 0,
        "holdout_subjects_evaluated": 0,
        "passed": shadow.passed,
    }
    report = {**base, "report_sha256": payload_hash(base)}
    report_path = gate_dir / "research-director-gate.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(report_path, 0o600)
    if not shadow.passed:
        raise ValueError("research_director_shadow_policy_failed")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_v8_research_director_gate(
            task_config_path=args.task_config,
            contract_path=args.contract,
            data_dir=args.data_dir,
            gate_dir=args.gate_dir,
            run_id=args.run_id,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"PRE-RUN BLOCKED: {exc}")
        return 2
    print("UNET_32H_V8_RESEARCH_DIRECTOR_GATE_PASS")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
