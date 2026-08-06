from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import os
import json
import copy
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from auto_researcher.agents.call_store import SQLiteAgentCallStore
from auto_researcher.agents.models import (
    ModelCallConfig,
    ModelPricing,
    StructuredModelResponse,
)
from auto_researcher.contracts.enums import AgentCallStatus, ProviderErrorCode
from auto_researcher.providers.protocols import ProviderCallError
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.live_models import (
    LiveMutationApproval,
    OPENEVOLVE_MUTATION_PROMPT_V1,
    OPENEVOLVE_MUTATION_PROMPT_V2,
    OpenEvolveModelBridgeContract,
    OpenEvolveModelCallContext,
    approval_content_hash,
    validate_approval,
)
from auto_researcher.search.openevolve.production_bridge import (
    DurableOpenEvolveModelBridge,
    LiveMutationBridgeError,
    mutation_input_hash,
)

NOW = datetime(2030, 1, 1, tzinfo=UTC)
HASH = "a" * 64
DIGEST = "sha256:" + "b" * 64


def pricing() -> ModelPricing:
    return ModelPricing(
        version="fake-pricing-v1",
        input_cost_per_million_tokens=1,
        output_cost_per_million_tokens=2,
        currency="USD",
    )


def contract(
    prompt_version: str = OPENEVOLVE_MUTATION_PROMPT_V2,
) -> OpenEvolveModelBridgeContract:
    return OpenEvolveModelBridgeContract.model_validate(
        {
            "mutation_operator_id": "pinned-upstream-openevolve",
            "mutation_operator_version": "upstream-openevolve-adapter-v1",
            "prompt_version": prompt_version,
            "maximum_input_bytes": 20_000,
            "model_config": ModelCallConfig(
                provider="fake-production",
                model_id="fake-model-20260101",
                temperature=0,
                maximum_output_tokens=1_000,
                timeout_seconds=10,
                maximum_attempts=1,
                maximum_cost_per_call=0.02,
                pricing=pricing(),
                prompt_version=prompt_version,
            ),
        }
    )


def context() -> OpenEvolveModelCallContext:
    return OpenEvolveModelCallContext(
        run_id="run-8",
        thread_id="thread-8",
        contract_id="contract-8",
        contract_hash=HASH,
        task_id="synthetic",
        task_version="1",
        search_request_id="search-8",
        generation=1,
        parent_candidate_id="candidate-seed",
        component_id="synthetic-component",
        component_version="1",
        component_interface_hash=HASH,
        adapter_id="auto-researcher-upstream-openevolve",
        adapter_version="1",
        adapter_identity_hash=HASH,
        executor_policy_hash=HASH,
        image_digest=DIGEST,
        mutable_file="candidate.py",
        model_budget_identity="budget-8",
        maximum_model_calls=1,
        maximum_model_cost=0.02,
    )


def approval_payload(**updates):
    payload = {
        "approval_id": "approval-8",
        "run_id": "run-8",
        "contract_id": "contract-8",
        "contract_hash": HASH,
        "task_id": "synthetic",
        "task_version": "1",
        "component_id": "synthetic-component",
        "component_version": "1",
        "adapter_id": "auto-researcher-upstream-openevolve",
        "adapter_version": "1",
        "adapter_identity_hash": HASH,
        "provider": "fake-production",
        "model_id": "fake-model-20260101",
        "prompt_id": "openevolve-mutation",
        "prompt_version": OPENEVOLVE_MUTATION_PROMPT_V2,
        "mutation_operator_version": "upstream-openevolve-adapter-v1",
        "maximum_model_calls": 1,
        "maximum_input_tokens": 4_000,
        "maximum_output_tokens": 1_000,
        "maximum_total_cost": 0.02,
        "currency": "USD",
        "pricing_version": "fake-pricing-v1",
        "executor_policy_hash": HASH,
        "image_digest": DIGEST,
        "mutable_file": "candidate.py",
        "created_at": NOW - timedelta(minutes=1),
        "expires_at": NOW + timedelta(minutes=10),
        "reviewer_identity": "operator-8",
        "residual_risk_acknowledged": True,
    }
    payload.update(updates)
    payload["approval_hash"] = approval_content_hash(payload)
    return payload


def approval(**updates) -> LiveMutationApproval:
    return LiveMutationApproval.model_validate(approval_payload(**updates))


LEGACY_REQUEST = {
    "protocol": "upstream-adapter-mutation-request-v1",
    "parent": {"id": "seed", "code": "def run(x): return x", "generation": 0},
    "mutable_file": "candidate.py",
    "interface_contract": "run(x)",
    "maximum_source_bytes": 1_000,
}
REQUEST = {
    **LEGACY_REQUEST,
    "protocol": "upstream-adapter-mutation-request-v2",
    "mutation_constraints": {
        "protocol_version": "openevolve-mutation-constraints-v2",
        "mutable_file": "candidate.py",
        "allowed_files": ["candidate.py"],
        "entry_point": "run",
        "immutable_interface_contract": "run(x)",
        "maximum_source_bytes": 1_000,
        "allowed_imports": [],
        "allowed_dependencies": [],
        "allowed_imports_display": "NONE",
        "allowed_dependencies_display": "NONE",
        "parameter_schema": {"x": "any"},
        "output_schema": {"result": "any"},
    },
}
OUTPUT = {
    "protocol_version": "upstream-mutation-envelope-v1",
    "mutable_file": "candidate.py",
    "source": "def run(x):\n    return x + 1\n",
    "description": "Bounded deterministic mutation.",
    "upstream_program_id": None,
    "dependency_requests": [],
    "provider_configuration": {},
}


class FakeProvider:
    provider = "fake-production"
    model_id = "fake-model-20260101"

    def __init__(self, calls: list[str], error: ProviderCallError | None = None):
        self.calls = calls
        self.error = error

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs["call_id"])
        if self.error:
            raise self.error
        return StructuredModelResponse(
            call_id=kwargs["call_id"],
            provider=self.provider,
            model_id=self.model_id,
            structured_output=OUTPUT,
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.0002,
            latency_ms=2,
            provider_request_id="fake-request-1",
            finish_reason="end_turn",
            prompt_version=kwargs["call_config"].prompt_version,
            context_hash=kwargs["context_hash"],
            response_hash=payload_hash(OUTPUT),
        )


def bridge(store, calls, *, factory=True, crash_after_response=False, error=None):
    return DurableOpenEvolveModelBridge(
        contract=contract(),
        context=context(),
        approval=approval(),
        store=store,
        provider_factory=(lambda: FakeProvider(calls, error)) if factory else None,
        now=lambda: NOW,
        system_prompt="bounded prompt",
        crash_after_response=crash_after_response,
    )


def test_completion_is_durable_and_replays_without_provider_credentials(tmp_path):
    path = tmp_path / "calls.sqlite"
    calls: list[str] = []
    first_store = SQLiteAgentCallStore(path)
    result, reservation = bridge(first_store, calls).complete(REQUEST, "mutation-1")
    first_store.close()
    assert result == OUTPUT
    assert len(calls) == 1
    assert reservation.prompt_version == OPENEVOLVE_MUTATION_PROMPT_V2

    reopened = SQLiteAgentCallStore(path)
    replayed, same = bridge(reopened, calls, factory=False).complete(
        REQUEST, "mutation-1"
    )
    records = reopened.records_for_call(reservation.reservation_id)
    reopened.close()
    assert replayed == OUTPUT
    assert same == reservation
    assert len(calls) == 1
    assert [item.status for item in records] == [
        AgentCallStatus.RESERVED,
        AgentCallStatus.DISPATCHING,
        AgentCallStatus.COMPLETED,
    ]


@pytest.mark.parametrize(
    ("approval_version", "contract_version"),
    [
        (OPENEVOLVE_MUTATION_PROMPT_V1, OPENEVOLVE_MUTATION_PROMPT_V2),
        (OPENEVOLVE_MUTATION_PROMPT_V2, OPENEVOLVE_MUTATION_PROMPT_V1),
    ],
)
def test_prompt_version_approval_mismatches_fail_both_directions(
    approval_version, contract_version
):
    with pytest.raises(ValueError, match="live_mutation_approval_mismatch"):
        validate_approval(
            approval(prompt_version=approval_version),
            context(),
            contract(contract_version),
            now=NOW,
        )


def test_constraint_and_prompt_versions_are_structured_input_identity_bearing():
    baseline = mutation_input_hash(REQUEST, OPENEVOLVE_MUTATION_PROMPT_V2)
    fields = (
        "mutable_file",
        "immutable_interface_contract",
        "allowed_imports",
        "allowed_dependencies",
        "maximum_source_bytes",
        "parameter_schema",
        "output_schema",
    )
    changed_hashes = set()
    for field in fields:
        changed = copy.deepcopy(REQUEST)
        constraints = changed["mutation_constraints"]
        if field == "mutable_file":
            changed["mutable_file"] = constraints[field] = "alternate.py"
            constraints["allowed_files"] = ["alternate.py"]
        elif field == "immutable_interface_contract":
            changed["interface_contract"] = constraints[field] = "run(y)"
        elif field == "maximum_source_bytes":
            changed["maximum_source_bytes"] = constraints[field] = 999
        elif field in {"allowed_imports", "allowed_dependencies"}:
            constraints[field] = ["math"]
            constraints[f"{field}_display"] = "math"
        else:
            constraints[field] = {"changed": True}
        changed_hashes.add(mutation_input_hash(changed, OPENEVOLVE_MUTATION_PROMPT_V2))
    assert baseline not in changed_hashes
    assert len(changed_hashes) == len(fields)
    assert mutation_input_hash(LEGACY_REQUEST, OPENEVOLVE_MUTATION_PROMPT_V1) != (
        mutation_input_hash(LEGACY_REQUEST, OPENEVOLVE_MUTATION_PROMPT_V2)
    )


def test_v1_completion_replays_with_original_identity_and_no_provider(tmp_path):
    prompt = (
        Path(__file__).parents[2]
        / "src/auto_researcher/prompts/openevolve/openevolve-mutation-prompt-v1.md"
    ).read_text(encoding="utf-8")
    assert payload_hash(prompt) == (
        "74e04df133520ab0f0ca7ff5a723f63549f0d88055dcf13a23fff503ae25c3a5"
    )
    path = tmp_path / "v1-calls.sqlite"
    calls: list[str] = []
    v1_approval = approval(prompt_version=OPENEVOLVE_MUTATION_PROMPT_V1)
    first_store = SQLiteAgentCallStore(path)
    first = DurableOpenEvolveModelBridge(
        contract=contract(OPENEVOLVE_MUTATION_PROMPT_V1),
        context=context(),
        approval=v1_approval,
        store=first_store,
        provider_factory=lambda: FakeProvider(calls),
        now=lambda: NOW,
        system_prompt=prompt,
    )
    source, reservation = first.complete(LEGACY_REQUEST, "mutation-v1")
    original_records = tuple(first_store.list_records())
    first_store.close()
    assert len(calls) == 1

    reopened = SQLiteAgentCallStore(path)
    replay = DurableOpenEvolveModelBridge(
        contract=contract(OPENEVOLVE_MUTATION_PROMPT_V1),
        context=context(),
        approval=v1_approval,
        store=reopened,
        provider_factory=None,
        now=lambda: NOW,
        system_prompt=prompt,
    )
    replayed, same = replay.complete(LEGACY_REQUEST, "mutation-v1")
    assert replayed == source == OUTPUT
    assert same == reservation
    assert same.prompt_version == OPENEVOLVE_MUTATION_PROMPT_V1
    assert tuple(reopened.list_records()) == original_records
    assert len(calls) == 1


def test_v1_and_v2_call_identities_cannot_cross_reuse(tmp_path):
    call_ids = []
    for version, request in (
        (OPENEVOLVE_MUTATION_PROMPT_V1, LEGACY_REQUEST),
        (OPENEVOLVE_MUTATION_PROMPT_V2, REQUEST),
    ):
        store = SQLiteAgentCallStore(tmp_path / f"{version}.sqlite")
        item = DurableOpenEvolveModelBridge(
            contract=contract(version),
            context=context(),
            approval=approval(prompt_version=version),
            store=store,
            provider_factory=lambda: FakeProvider([]),
            now=lambda: NOW,
            system_prompt=f"bounded prompt for {version}",
        )
        _, reservation = item.complete(request, "same-mutation")
        call_ids.append(reservation.reservation_id)
    assert call_ids[0] != call_ids[1]


def test_rendered_v2_prompt_is_never_persisted(tmp_path):
    prompt = (
        Path(__file__).parents[2]
        / "src/auto_researcher/prompts/openevolve/openevolve-mutation-prompt-v2.md"
    ).read_text(encoding="utf-8")
    assert payload_hash(prompt) == (
        "4ead06957a06f470578357b3d8b74a0c3ee6eca2027bf01cfff378ebc4a26ce6"
    )
    store = SQLiteAgentCallStore(tmp_path / "calls.sqlite")
    item = DurableOpenEvolveModelBridge(
        contract=contract(),
        context=context(),
        approval=approval(),
        store=store,
        provider_factory=lambda: FakeProvider([]),
        now=lambda: NOW,
        system_prompt=prompt,
    )
    item.complete(REQUEST, "mutation-v2")
    persisted = "\n".join(record.model_dump_json() for record in store.list_records())
    assert prompt not in persisted
    assert "The component contract in the request is authoritative" not in persisted
    assert OPENEVOLVE_MUTATION_PROMPT_V2 in persisted
    assert payload_hash(prompt) in persisted


def test_provider_factory_failure_is_known_before_invocation(tmp_path):
    store = SQLiteAgentCallStore(tmp_path / "calls.sqlite")
    calls: list[str] = []
    with pytest.raises(LiveMutationBridgeError, match="provider_unavailable"):
        bridge(store, calls, factory=False).complete(REQUEST, "mutation-1")
    assert calls == []
    assert store.list_records()[-1].status == AgentCallStatus.FAILED_BEFORE_DISPATCH


def test_missing_approval_rejects_before_reservation_or_provider(tmp_path):
    store = SQLiteAgentCallStore(tmp_path / "calls.sqlite")
    constructions = 0

    def factory():
        nonlocal constructions
        constructions += 1
        return FakeProvider([])

    item = DurableOpenEvolveModelBridge(
        contract=contract(),
        context=context(),
        approval=None,
        store=store,
        provider_factory=factory,
        now=lambda: NOW,
        system_prompt="bounded prompt",
    )
    with pytest.raises(LiveMutationBridgeError, match="approval_required"):
        item.complete(REQUEST, "mutation-1")
    assert constructions == 0
    assert store.list_records() == ()


def test_timeout_and_post_response_crash_are_outcome_unknown(tmp_path):
    timeout_store = SQLiteAgentCallStore(tmp_path / "timeout.sqlite")
    timeout = ProviderCallError(ProviderErrorCode.TIMEOUT, retryable=True)
    with pytest.raises(LiveMutationBridgeError, match="outcome_unknown"):
        bridge(timeout_store, [], error=timeout).complete(REQUEST, "mutation-1")
    assert timeout_store.list_records()[-1].status == AgentCallStatus.OUTCOME_UNKNOWN

    crash_store = SQLiteAgentCallStore(tmp_path / "crash.sqlite")
    with pytest.raises(LiveMutationBridgeError, match="outcome_unknown"):
        bridge(crash_store, [], crash_after_response=True).complete(
            REQUEST, "mutation-1"
        )
    assert crash_store.list_records()[-1].status == AgentCallStatus.OUTCOME_UNKNOWN
    with pytest.raises(LiveMutationBridgeError, match="outcome_unknown"):
        bridge(crash_store, []).complete(REQUEST, "mutation-1")


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"expires_at": NOW}, "live_mutation_approval_expired"),
        ({"run_id": "wrong"}, "live_mutation_approval_mismatch"),
        ({"task_id": "wrong"}, "live_mutation_approval_mismatch"),
        ({"component_id": "wrong"}, "live_mutation_approval_mismatch"),
        ({"provider": "wrong"}, "live_mutation_approval_mismatch"),
        ({"model_id": "wrong-20260101"}, "live_mutation_approval_mismatch"),
        ({"prompt_version": "wrong-v1"}, "live_mutation_approval_mismatch"),
        ({"image_digest": "sha256:" + "c" * 64}, "live_mutation_approval_mismatch"),
        ({"maximum_output_tokens": 10}, "live_mutation_approval_mismatch"),
        ({"maximum_total_cost": 0.001}, "model_call_cost_limit_exceeded"),
    ],
)
def test_approval_scope_fails_closed(updates, code):
    item = approval(**updates)
    with pytest.raises(ValueError, match=code):
        validate_approval(item, context(), contract(), now=NOW)


def test_tampered_approval_and_hostile_upstream_request_are_rejected(tmp_path):
    payload = approval_payload()
    payload["model_id"] = "tampered-model-20260101"
    with pytest.raises(ValueError, match="live_mutation_approval_tampered"):
        LiveMutationApproval.model_validate(payload)
    store = SQLiteAgentCallStore(tmp_path / "calls.sqlite")
    hostile = {**REQUEST, "provider_configuration": {"api_key": "not-a-key"}}
    with pytest.raises(LiveMutationBridgeError, match="direct_provider_forbidden"):
        bridge(store, []).complete(hostile, "mutation-1")
    assert store.list_records() == ()


def test_two_sqlite_connections_allow_at_most_one_dispatch(tmp_path):
    path = tmp_path / "calls.sqlite"
    calls: list[str] = []

    def execute():
        store = SQLiteAgentCallStore(path)
        try:
            return bridge(store, calls).complete(REQUEST, "mutation-1")[0]
        except LiveMutationBridgeError as exc:
            return exc.code
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: execute(), range(2)))
    assert len(calls) == 1
    assert OUTPUT in results
    final = SQLiteAgentCallStore(path)
    statuses = [item.status for item in final.list_records()]
    assert statuses.count(AgentCallStatus.RESERVED) == 1
    assert statuses.count(AgentCallStatus.DISPATCHING) == 1
    assert statuses.count(AgentCallStatus.COMPLETED) == 1


def test_budget_is_reserved_once_and_replay_spends_nothing(tmp_path):
    store = SQLiteAgentCallStore(tmp_path / "calls.sqlite")
    calls: list[str] = []
    item = bridge(store, calls)
    item.complete(REQUEST, "mutation-1")
    before = tuple(store.list_records())
    item.complete(REQUEST, "mutation-1")
    assert tuple(store.list_records()) == before
    with pytest.raises(LiveMutationBridgeError, match="budget_exhausted"):
        item.complete(REQUEST, "mutation-2")
    assert len(calls) == 1


def test_confirmed_failure_is_terminal_and_conservatively_counted(tmp_path):
    store = SQLiteAgentCallStore(tmp_path / "calls.sqlite")
    error = ProviderCallError(
        ProviderErrorCode.PERMANENT_PROVIDER_ERROR,
        retryable=False,
        input_tokens=10,
        output_tokens=2,
        estimated_cost=0.000014,
    )
    with pytest.raises(LiveMutationBridgeError, match="confirmed_failure"):
        bridge(store, [], error=error).complete(REQUEST, "mutation-1")
    assert store.list_records()[-1].status == AgentCallStatus.FAILED_CONFIRMED
    with pytest.raises(LiveMutationBridgeError, match="budget_exhausted"):
        bridge(store, []).complete(REQUEST, "mutation-2")


def test_bridge_contract_rejects_unsupported_or_unbounded_configuration():
    payload = contract().model_dump(mode="python", by_alias=True)
    payload["model_config"]["provider"] = "unsupported"
    with pytest.raises(ValueError, match="provider_not_supported"):
        OpenEvolveModelBridgeContract.model_validate(payload)
    payload = contract().model_dump(mode="python", by_alias=True)
    payload["model_config"]["maximum_attempts"] = 2
    with pytest.raises(ValueError, match="retries are prohibited"):
        OpenEvolveModelBridgeContract.model_validate(payload)


def test_credential_like_approval_field_is_rejected():
    payload = approval_payload()
    payload["reviewer_identity"] = "person@example.com"
    payload["approval_hash"] = approval_content_hash(
        {**payload, "reviewer_identity": "operator-safe"}
    )
    with pytest.raises(ValueError):
        LiveMutationApproval.model_validate(payload)


def test_approval_call_and_completion_identities_are_cross_process_stable():
    script = """
from tempfile import NamedTemporaryFile
from auto_researcher.agents.call_store import SQLiteAgentCallStore
from tests.unit.test_openevolve_production_bridge import REQUEST, approval, bridge
with NamedTemporaryFile(suffix='.sqlite') as handle:
    store = SQLiteAgentCallStore(handle.name)
    _, reservation = bridge(store, []).complete(REQUEST, 'mutation-1')
    record = store.latest(reservation.reservation_id)
    print(approval().approval_hash, reservation.reservation_id, record.completion_identity)
"""
    outputs = []
    for seed in ("1", "73", "999"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            ).stdout.strip()
        )
    assert len(set(outputs)) == 1


def test_tampered_completed_payload_fails_without_provider_construction(tmp_path):
    path = tmp_path / "calls.sqlite"
    calls: list[str] = []
    store = SQLiteAgentCallStore(path)
    bridge(store, calls).complete(REQUEST, "mutation-1")
    store.close()
    connection = sqlite3.connect(path)
    row = connection.execute(
        "SELECT sequence, payload FROM agent_call_records ORDER BY sequence DESC LIMIT 1"
    ).fetchone()
    payload = json.loads(row[1])
    payload["structured_output"]["source"] = "def run(x): return 999"
    connection.execute(
        "UPDATE agent_call_records SET payload = ? WHERE sequence = ?",
        (json.dumps(payload), row[0]),
    )
    connection.commit()
    connection.close()
    reopened = SQLiteAgentCallStore(path)
    constructions = 0

    def forbidden_factory():
        nonlocal constructions
        constructions += 1
        raise AssertionError("provider must not be constructed")

    item = DurableOpenEvolveModelBridge(
        contract=contract(),
        context=context(),
        approval=approval(),
        store=reopened,
        provider_factory=forbidden_factory,
        now=lambda: NOW,
        system_prompt="bounded prompt",
    )
    with pytest.raises(LiveMutationBridgeError, match="completed_response_corrupt"):
        item.complete(REQUEST, "mutation-1")
    assert constructions == 0


@pytest.mark.parametrize(
    "prompt_version",
    [OPENEVOLVE_MUTATION_PROMPT_V1, OPENEVOLVE_MUTATION_PROMPT_V2],
)
def test_checkpoint_05b_offline_completion_reuse_precedes_candidate_evaluation(
    tmp_path, prompt_version
):
    pytest.importorskip("openevolve", reason="pinned optional dependency absent")
    from dataclasses import replace

    from auto_researcher.agents.provenance import append_model_call_events
    from auto_researcher.contracts.enums import SearchType
    from auto_researcher.contracts.models import BudgetState, SearchRequest
    from auto_researcher.graph.nodes.evaluate import evaluate_experiment
    from auto_researcher.graph.nodes.verify import verify_evidence
    from auto_researcher.providers.fake_production import (
        FakeProductionStructuredModelClient,
    )
    from auto_researcher.runtime.dependencies import task_memory_dependencies
    from auto_researcher.search.openevolve.backend import OpenEvolveBackend
    from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
    from auto_researcher.search.openevolve.upstream import (
        UpstreamOpenEvolveAdapter,
        default_adapter_contract,
        mutation_constraints,
    )
    from auto_researcher.tasks.models import TaskRuntimeContext
    from auto_researcher.tasks.synthetic import (
        SyntheticTask,
        default_synthetic_contract,
        default_synthetic_openevolve_configuration,
    )

    class Counting:
        def __init__(self, inner, method):
            self.inner = inner
            self.method = method
            self.calls = 0
            for name in (
                "evaluator_id",
                "verifier_id",
                "version",
                "cost_per_experiment",
            ):
                if hasattr(inner, name):
                    setattr(self, name, getattr(inner, name))

        def evaluate(self, *args, **kwargs):
            self.calls += 1
            return self.inner.evaluate(*args, **kwargs)

        def verify(self, *args, **kwargs):
            self.calls += 1
            return self.inner.verify(*args, **kwargs)

    run_id = "offline-05b-model-replay"
    research_contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPENEVOLVE}),
        maximum_experiments=2,
    )
    configuration = default_synthetic_openevolve_configuration()
    configuration["openevolve"]["maximum_model_calls"] = 1
    request = SearchRequest(
        request_id="offline-05b-search",
        hypothesis_id="offline-05b-hypothesis",
        search_type=SearchType.OPENEVOLVE,
        target="bounded fixture",
        search_space=configuration,
        experiment_budget=2,
        rationale="offline paid-call replay simulation",
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id=run_id, output_dir=tmp_path / "artefacts"),
        research_contract,
        configuration,
        search_type=SearchType.OPENEVOLVE,
        clock=lambda: NOW,
    )
    base = dependencies.openevolve_backend
    assert base is not None
    component = base.component_spec
    adapter_contract = default_adapter_contract(
        Path(__file__).parents[2] / "constraints/openevolve-0.3.2.lock"
    )
    adapter_hash = payload_hash(adapter_contract)
    approval_item = LiveMutationApproval.model_validate(
        approval_payload(
            run_id=run_id,
            contract_id=research_contract.contract_id,
            contract_hash=payload_hash(research_contract),
            task_version=research_contract.task_version,
            component_id=component.component_id,
            component_version=component.component_version,
            adapter_identity_hash=adapter_hash,
            prompt_version=prompt_version,
        )
    )
    bridge_context = context().model_copy(
        update={
            "run_id": run_id,
            "contract_id": research_contract.contract_id,
            "contract_hash": payload_hash(research_contract),
            "task_version": research_contract.task_version,
            "search_request_id": request.request_id,
            "component_id": component.component_id,
            "component_version": component.component_version,
            "component_interface_hash": base.interface_hash,
            "adapter_identity_hash": adapter_hash,
        }
    )
    call_path = tmp_path / "agent-calls.sqlite"
    first_store = SQLiteAgentCallStore(call_path)
    response = {
        "protocol_version": "upstream-mutation-envelope-v1",
        "mutable_file": component.mutable_file,
        "source": (
            "def evolve(configuration):\n"
            '    return {"model_family": "tree", "complexity": 4, '
            '"learning_rate": 0.05}\n'
        ),
        "description": "One bounded synthetic mutation.",
    }
    provider = FakeProductionStructuredModelClient(
        provider="fake-production",
        model_id="fake-model-20260101",
        response=response,
    )
    first_bridge = DurableOpenEvolveModelBridge(
        contract=contract(prompt_version),
        context=bridge_context,
        approval=approval_item,
        store=first_store,
        provider_factory=lambda: provider,
        now=lambda: NOW,
        system_prompt="bounded prompt",
    )
    # Persist completion, then simulate termination before adapter reconciliation.
    first_backend = OpenEvolveBackend(
        base.component,
        base.metadata,
        base.verifier_identity,
        UpstreamOpenEvolveAdapter(adapter_contract, first_bridge),
        LocalSandboxRunner(tmp_path / "first-workspace"),
    )
    search_contract = first_backend.create_search_contract(request, research_contract)
    seed = first_backend.seed_candidate(search_contract)
    population = first_backend.initialise_population(search_contract)
    reservation = first_backend.reserve_mutation(search_contract, population, seed)
    model_request = {
        "protocol": "upstream-adapter-mutation-request-v1",
        "parent": {
            "id": f"upstream-{seed.candidate_id}",
            "authoritative_candidate_id": seed.candidate_id,
            "code": seed.source_payload,
            "generation": seed.generation,
        },
        "mutable_file": component.mutable_file,
        "interface_contract": component.immutable_interface_contract,
        "maximum_source_bytes": component.maximum_source_bytes,
    }
    if prompt_version == OPENEVOLVE_MUTATION_PROMPT_V2:
        model_request.update(
            {
                "protocol": "upstream-adapter-mutation-request-v2",
                "mutation_constraints": mutation_constraints(component).model_dump(
                    mode="json"
                ),
            }
        )
    _, persisted = first_bridge.complete(model_request, reservation.reservation_id)
    model_records = tuple(first_store.list_records())
    first_store.close()
    assert provider.invocation_count == 1

    reopened = SQLiteAgentCallStore(call_path)
    reconstructed_bridge = DurableOpenEvolveModelBridge(
        contract=contract(prompt_version),
        context=bridge_context,
        approval=approval_item,
        store=reopened,
        provider_factory=None,
        now=lambda: NOW,
        system_prompt="bounded prompt",
    )
    reconstructed_adapter = UpstreamOpenEvolveAdapter(
        adapter_contract, reconstructed_bridge
    )
    reconstructed_backend = OpenEvolveBackend(
        base.component,
        base.metadata,
        base.verifier_identity,
        reconstructed_adapter,
        LocalSandboxRunner(tmp_path / "reconstructed-workspace"),
    )
    candidate = reconstructed_backend.mutate_candidate(
        reservation, seed, search_contract
    )
    candidate_again = reconstructed_backend.mutate_candidate(
        reservation, seed, search_contract
    )
    assert candidate_again.candidate_id == candidate.candidate_id
    assert candidate.model_call_id == persisted.reservation_id
    assert persisted.prompt_version == prompt_version
    assert tuple(reopened.list_records()) == model_records
    assert provider.invocation_count == 1

    validation = reconstructed_backend.validate(candidate)
    candidate = candidate.model_copy(update={"validation_result": validation})
    preparation = reconstructed_backend.prepare(candidate, search_contract)
    experiment = reconstructed_backend.component.candidate_to_experiment(
        candidate,
        preparation,
        request,
        research_contract,
        dependencies.experiment_metadata,
        run_id=run_id,
    )
    evaluator = Counting(dependencies.evaluator, "evaluate")
    verifier = Counting(dependencies.verifier, "verify")
    dependencies = replace(dependencies, evaluator=evaluator, verifier=verifier)
    state = {
        "run_id": run_id,
        "contract": research_contract,
        "search_request": request,
        "experiment_spec": experiment,
        "budget": BudgetState(
            maximum_cycles=1,
            maximum_experiments=2,
            maximum_cost=1,
        ),
    }
    evaluated = evaluate_experiment(state, dependencies)
    state.update(evaluated)
    verified = verify_evidence(state, dependencies)
    state.update(verified)
    evaluate_experiment(state, dependencies)
    verify_evidence(state, dependencies)
    assert evaluator.calls == verifier.calls == 1
    assert (
        dependencies.provenance_store.get_evaluation_reuse(
            run_id, experiment.experiment_id
        ).protocol_version
        == "evaluation-reuse-v2"
    )
    append_model_call_events(
        dependencies.provenance_store, reopened, run_id=run_id, cycle=1
    )
    events = dependencies.provenance_store.list_events(run_id)
    append_model_call_events(
        dependencies.provenance_store, reopened, run_id=run_id, cycle=1
    )
    assert dependencies.provenance_store.list_events(run_id) == events
    assert tuple(reopened.list_records()) == model_records
