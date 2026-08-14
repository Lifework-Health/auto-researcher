from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from auto_researcher.agents.call_store import SQLiteAgentCallStore
from auto_researcher.contracts.enums import AgentCallStatus, ProvenanceKind, SearchType
from auto_researcher.contracts.models import SearchRequest
from auto_researcher.providers.fake_production import (
    FakeProductionStructuredModelClient,
)
from auto_researcher.search.openevolve.live_models import (
    MetadataOnlyLiveMutationApproval,
    OpenEvolveModelBridgeContract,
    metadata_only_approval_content_hash,
)
from auto_researcher.search.openevolve.live_runtime import (
    MetadataOnlyLiveOpenEvolveConfiguration,
    MetadataOnlyLiveOpenEvolveRuntime,
    assemble_metadata_only_live_openevolve,
)
from auto_researcher.secrets import (
    ResolvedSecret,
    SecretProviderKind,
    SecretReference,
)
from auto_researcher.search.openevolve.production_bridge import LiveMutationBridgeError
from auto_researcher.search.openevolve.upstream import (
    UpstreamOpenEvolveAdapter,
    mutation_constraints,
)
from auto_researcher.search.openevolve.hardened_executor import HardenedDockerExecutor
from auto_researcher.tasks.feta_seg_evolve import (
    default_feta_evolve_openevolve_configuration,
)
from auto_researcher.tasks.feta_seg.manifests import (
    DATASET_RELEASE,
    EXPECTED_MANIFEST_HASH,
)
from auto_researcher.tasks.feta_seg_evolve.evaluator import (
    EVALUATOR_ID,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_seg_evolve.openevolve import COSINE_SOURCE
from auto_researcher.tasks.models import ExperimentMetadata
from tests.unit.test_openevolve_metadata_only_boundary import (
    NOW,
    PROMPT,
    metadata_only_contract,
    metadata_only_evidence,
)


ROOT = Path(__file__).parents[2]


def _write_json(path: Path, value) -> Path:
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json", by_alias=True)
    else:
        payload = value
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _runtime(
    tmp_path: Path,
    evidence,
    *,
    bridge_contract=None,
    approval=None,
    credential=None,
    secret_provider_factory=None,
    provider_factory=None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text(PROMPT, encoding="utf-8")
    configuration_payload = {
        "approval_file": _write_json(
            tmp_path / "approval.json", approval or evidence["approval"]
        ),
        "bridge_contract_file": _write_json(
            tmp_path / "bridge.json", bridge_contract or metadata_only_contract()
        ),
        "adapter_lock_file": ROOT / "constraints/openevolve-0.3.2.lock",
        "prompt_file": prompt_file,
        "executor_policy_file": _write_json(
            tmp_path / "executor.json", evidence["policy"]
        ),
        "isolation_evidence_file": _write_json(
            tmp_path / "isolation.json", evidence["isolation"]
        ),
    }
    if credential is not None:
        configuration_payload["credential"] = credential
    configuration = MetadataOnlyLiveOpenEvolveConfiguration(**configuration_payload)
    return MetadataOnlyLiveOpenEvolveRuntime(
        configuration=configuration,
        thread_id=evidence["context"].thread_id,
        provider_factory=provider_factory,
        secret_provider_factory=secret_provider_factory,
        executor_validator=lambda executor: None,
    )


def _search_configuration(*, maximum_model_calls: int = 1):
    configuration = default_feta_evolve_openevolve_configuration()
    configuration["openevolve"].update(
        {
            "maximum_generations": 1,
            "maximum_candidate_evaluations": 2,
            "maximum_model_calls": maximum_model_calls,
            "sandbox_policy_id": "openevolve-hardened-executor-v2",
        }
    )
    return configuration


def _anthropic_contract_and_approval(evidence, *, maximum_model_calls: int = 3):
    model_id = "claude-approved-20260801"
    bridge_payload = metadata_only_contract().model_dump(mode="python", by_alias=True)
    bridge_payload["model_config"].update(
        {"provider": "anthropic", "model_id": model_id}
    )
    bridge = OpenEvolveModelBridgeContract.model_validate(bridge_payload)
    approval_payload = evidence["approval"].model_dump(mode="python")
    approval_payload.update(
        {
            "provider": "anthropic",
            "model_id": model_id,
            "maximum_model_calls": maximum_model_calls,
            "maximum_total_cost": 0.06,
            "approval_hash": "0" * 64,
        }
    )
    approval_payload["approval_hash"] = metadata_only_approval_content_hash(
        approval_payload
    )
    return bridge, MetadataOnlyLiveMutationApproval.model_validate(approval_payload)


def _model_request(evidence, *, parent_id: str):
    spec = evidence["spec"]
    return {
        "protocol": "upstream-adapter-mutation-request-v2",
        "parent": {
            "id": f"upstream-{parent_id}",
            "authoritative_candidate_id": parent_id,
            "code": spec.seed_source,
            "generation": 0,
        },
        "mutable_file": spec.mutable_file,
        "interface_contract": spec.immutable_interface_contract,
        "maximum_source_bytes": spec.maximum_source_bytes,
        "mutation_constraints": mutation_constraints(spec).model_dump(mode="json"),
    }


class CountingSecretProviderFactory:
    def __init__(self, values: list[str]):
        self.values = values
        self.references = []

    def __call__(self, reference):
        def resolve(resolved_reference):
            assert resolved_reference == reference
            self.references.append(reference)
            return ResolvedSecret(self.values[len(self.references) - 1])

        return SimpleNamespace(resolve=resolve)


class DispatchOwnershipDeniedStore(SQLiteAgentCallStore):
    """Persist reservations but simulate losing the RESERVED transition race."""

    def transition(self, record, *, expected_status):
        if expected_status is AgentCallStatus.RESERVED:
            return False
        return super().transition(record, expected_status=expected_status)


def test_standard_runtime_assembles_feta_metadata_only_boundary_without_data(tmp_path):
    evidence = metadata_only_evidence()
    store = SQLiteAgentCallStore(tmp_path / "agent-calls.sqlite")
    try:
        operator, runner = assemble_metadata_only_live_openevolve(
            runtime=_runtime(tmp_path, evidence),
            task=evidence["task"],
            component=evidence["component"],
            research_contract=evidence["research_contract"],
            run_id=evidence["context"].run_id,
            experiment_configuration=_search_configuration(),
            call_store=store,
            workspace_root=(tmp_path / "workspace").resolve(),
            now=lambda: NOW,
        )
    finally:
        store.close()

    assert isinstance(operator, UpstreamOpenEvolveAdapter)
    assert isinstance(runner, HardenedDockerExecutor)
    assert operator.provenance == "FAKE_MODEL"
    assert operator.bridge.context.underlying_dataset_class == "mri"
    assert operator.bridge.context.mri_access is False
    assert operator.bridge.context.patient_data_access is False


def test_in_memory_dependency_factory_cannot_enable_live_mutation(tmp_path):
    evidence = metadata_only_evidence()
    from auto_researcher.runtime.dependencies import task_memory_dependencies
    from auto_researcher.tasks.models import TaskRuntimeContext

    with pytest.raises(ValueError, match="live_mutation_durable_runtime_required"):
        task_memory_dependencies(
            evidence["task"],
            TaskRuntimeContext(
                run_id=evidence["context"].run_id,
                data_dir=Path("/must-not-be-read"),
                workspace_dir=tmp_path / "workspace",
            ),
            evidence["research_contract"],
            _search_configuration(),
            search_type=SearchType.OPENEVOLVE,
            openevolve_live_runtime=_runtime(tmp_path, evidence),
        )


def test_anthropic_approved_contract_natively_records_live_model_provenance(tmp_path):
    evidence = metadata_only_evidence()
    bridge_payload = metadata_only_contract().model_dump(mode="python", by_alias=True)
    bridge_payload["model_config"].update(
        {"provider": "anthropic", "model_id": "claude-approved-20260801"}
    )
    bridge = OpenEvolveModelBridgeContract.model_validate(bridge_payload)
    approval_payload = evidence["approval"].model_dump(mode="python")
    approval_payload.update(
        {
            "provider": "anthropic",
            "model_id": "claude-approved-20260801",
            "approval_hash": "0" * 64,
        }
    )
    approval_payload["approval_hash"] = metadata_only_approval_content_hash(
        approval_payload
    )
    approval = MetadataOnlyLiveMutationApproval.model_validate(approval_payload)
    store = SQLiteAgentCallStore(tmp_path / "agent-calls.sqlite")
    try:
        operator, _ = assemble_metadata_only_live_openevolve(
            runtime=_runtime(
                tmp_path,
                evidence,
                bridge_contract=bridge,
                approval=approval,
            ),
            task=evidence["task"],
            component=evidence["component"],
            research_contract=evidence["research_contract"],
            run_id=evidence["context"].run_id,
            experiment_configuration=_search_configuration(),
            call_store=store,
            workspace_root=(tmp_path / "workspace").resolve(),
            now=lambda: NOW,
        )
    finally:
        store.close()
    assert operator.provenance == "LIVE_MODEL"


def test_real_provider_contract_persists_live_model_on_mutated_candidate(tmp_path):
    evidence = metadata_only_evidence()
    bridge_payload = metadata_only_contract().model_dump(mode="python", by_alias=True)
    bridge_payload["model_config"].update(
        {"provider": "anthropic", "model_id": "claude-approved-20260801"}
    )
    bridge = OpenEvolveModelBridgeContract.model_validate(bridge_payload)
    approval_payload = evidence["approval"].model_dump(mode="python")
    approval_payload.update(
        {
            "provider": "anthropic",
            "model_id": "claude-approved-20260801",
            "approval_hash": "0" * 64,
        }
    )
    approval_payload["approval_hash"] = metadata_only_approval_content_hash(
        approval_payload
    )
    provider = FakeProductionStructuredModelClient(
        provider="anthropic",
        model_id="claude-approved-20260801",
        response={
            "protocol_version": "upstream-mutation-envelope-v1",
            "mutable_file": "candidate.py",
            "source": COSINE_SOURCE,
            "description": "Structured stand-in for one approved real response.",
        },
    )
    raw_runtime = _runtime(
        tmp_path,
        evidence,
        bridge_contract=bridge,
        approval=MetadataOnlyLiveMutationApproval.model_validate(approval_payload),
    )
    runtime = MetadataOnlyLiveOpenEvolveRuntime(
        configuration=raw_runtime.configuration,
        thread_id=raw_runtime.thread_id,
        provider_factory=lambda: provider,
        executor_validator=lambda executor: None,
    )
    store = SQLiteAgentCallStore(tmp_path / "agent-calls.sqlite")
    operator, runner = assemble_metadata_only_live_openevolve(
        runtime=runtime,
        task=evidence["task"],
        component=evidence["component"],
        research_contract=evidence["research_contract"],
        run_id=evidence["context"].run_id,
        experiment_configuration=_search_configuration(),
        call_store=store,
        workspace_root=(tmp_path / "workspace").resolve(),
        now=lambda: NOW,
    )
    from auto_researcher.search.openevolve.backend import OpenEvolveBackend

    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"
    backend = OpenEvolveBackend(
        evidence["component"],
        ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(dataset_version),
            dataset_version=dataset_version,
            provenance=ProvenanceKind.REAL,
        ),
        _search_configuration()["openevolve"]["verifier_identity"],
        operator,
        runner,
    )
    request = SearchRequest(
        request_id="real-approved-search",
        hypothesis_id="hypothesis",
        search_type=SearchType.OPENEVOLVE,
        target="score",
        search_space=_search_configuration(),
        experiment_budget=2,
        rationale="Offline structured stand-in for provenance verification.",
    )
    contract = backend.create_search_contract(request, evidence["research_contract"])
    seed = backend.seed_candidate(contract)
    reservation = backend.reserve_mutation(
        contract, backend.initialise_population(contract), seed
    )
    candidate = backend.mutate_candidate(reservation, seed, contract)
    records = store.list_records()
    store.close()
    assert candidate.creation_provenance == "LIVE_MODEL"
    assert provider.invocation_count == 1
    assert records[-1].status.value == "COMPLETED"


@pytest.mark.parametrize(
    ("approval_update", "expected"),
    [
        ({"run_id": "wrong-run"}, "live_mutation_approval_mismatch"),
        ({"component_id": "wrong-component"}, "live_mutation_approval_mismatch"),
        (
            {"model_exposure_identity": "f" * 64},
            "live_mutation_approval_mismatch",
        ),
        ({"expires_at": NOW}, "live_mutation_approval_expired"),
        (
            {"image_digest": "sha256:" + "f" * 64},
            "live_mutation_approval_mismatch",
        ),
    ],
)
def test_standard_runtime_rejects_approval_identity_drift_before_dispatch(
    tmp_path, approval_update, expected
):
    evidence = metadata_only_evidence()
    payload = evidence["approval"].model_dump(mode="python")
    payload.update(approval_update)
    payload["approval_hash"] = metadata_only_approval_content_hash(payload)
    changed = MetadataOnlyLiveMutationApproval.model_validate(payload)
    store = SQLiteAgentCallStore(tmp_path / "agent-calls.sqlite")
    with pytest.raises(LiveMutationBridgeError, match=expected):
        assemble_metadata_only_live_openevolve(
            runtime=_runtime(tmp_path, evidence, approval=changed),
            task=evidence["task"],
            component=evidence["component"],
            research_contract=evidence["research_contract"],
            run_id=evidence["context"].run_id,
            experiment_configuration=_search_configuration(),
            call_store=store,
            workspace_root=(tmp_path / "workspace").resolve(),
            now=lambda: NOW,
        )
    assert store.list_records() == ()
    store.close()


def test_standard_runtime_rejects_hardened_environment_drift_before_dispatch(
    tmp_path,
):
    evidence = metadata_only_evidence()
    raw = _runtime(tmp_path, evidence)
    checked = []

    def reject(executor):
        checked.append(executor.policy.image_digest)
        raise ValueError("hardened_executor_image_mismatch")

    runtime = MetadataOnlyLiveOpenEvolveRuntime(
        configuration=raw.configuration,
        thread_id=raw.thread_id,
        executor_validator=reject,
    )
    store = SQLiteAgentCallStore(tmp_path / "agent-calls.sqlite")
    with pytest.raises(ValueError, match="hardened_executor_image_mismatch"):
        assemble_metadata_only_live_openevolve(
            runtime=runtime,
            task=evidence["task"],
            component=evidence["component"],
            research_contract=evidence["research_contract"],
            run_id=evidence["context"].run_id,
            experiment_configuration=_search_configuration(),
            call_store=store,
            workspace_root=(tmp_path / "workspace").resolve(),
            now=lambda: NOW,
        )
    assert checked == [evidence["policy"].image_digest]
    assert store.list_records() == ()
    store.close()


def test_secret_resolution_is_zero_during_assembly_approval_and_preflight_failures(
    tmp_path,
):
    evidence = metadata_only_evidence()
    bridge, approval = _anthropic_contract_and_approval(evidence)
    resolver_factory = CountingSecretProviderFactory(["must-not-resolve"])
    store = SQLiteAgentCallStore(tmp_path / "assembly-agent-calls.sqlite")
    operator, _ = assemble_metadata_only_live_openevolve(
        runtime=_runtime(
            tmp_path / "valid",
            evidence,
            bridge_contract=bridge,
            approval=approval,
            secret_provider_factory=resolver_factory,
        ),
        task=evidence["task"],
        component=evidence["component"],
        research_contract=evidence["research_contract"],
        run_id=evidence["context"].run_id,
        experiment_configuration=_search_configuration(maximum_model_calls=3),
        call_store=store,
        workspace_root=(tmp_path / "valid-workspace").resolve(),
        now=lambda: NOW,
    )
    assert operator.provenance == "LIVE_MODEL"
    assert resolver_factory.references == []
    store.close()

    expired_payload = approval.model_dump(mode="python")
    expired_payload.update({"expires_at": NOW, "approval_hash": "0" * 64})
    expired_payload["approval_hash"] = metadata_only_approval_content_hash(
        expired_payload
    )
    expired = MetadataOnlyLiveMutationApproval.model_validate(expired_payload)
    expired_store = SQLiteAgentCallStore(tmp_path / "expired-agent-calls.sqlite")
    with pytest.raises(LiveMutationBridgeError, match="approval_expired"):
        assemble_metadata_only_live_openevolve(
            runtime=_runtime(
                tmp_path / "expired",
                evidence,
                bridge_contract=bridge,
                approval=expired,
                secret_provider_factory=resolver_factory,
            ),
            task=evidence["task"],
            component=evidence["component"],
            research_contract=evidence["research_contract"],
            run_id=evidence["context"].run_id,
            experiment_configuration=_search_configuration(maximum_model_calls=3),
            call_store=expired_store,
            workspace_root=(tmp_path / "expired-workspace").resolve(),
            now=lambda: NOW,
        )
    assert resolver_factory.references == []
    expired_store.close()

    mismatch_store = SQLiteAgentCallStore(tmp_path / "mismatch-agent-calls.sqlite")
    with pytest.raises(LiveMutationBridgeError, match="approval_mismatch"):
        assemble_metadata_only_live_openevolve(
            runtime=_runtime(
                tmp_path / "mismatch",
                evidence,
                bridge_contract=bridge,
                approval=approval,
                secret_provider_factory=resolver_factory,
            ),
            task=evidence["task"],
            component=evidence["component"],
            research_contract=evidence["research_contract"],
            run_id="wrong-run",
            experiment_configuration=_search_configuration(maximum_model_calls=3),
            call_store=mismatch_store,
            workspace_root=(tmp_path / "mismatch-workspace").resolve(),
            now=lambda: NOW,
        )
    assert resolver_factory.references == []
    mismatch_store.close()

    raw = _runtime(
        tmp_path / "preflight",
        evidence,
        bridge_contract=bridge,
        approval=approval,
        secret_provider_factory=resolver_factory,
    )
    preflight_runtime = MetadataOnlyLiveOpenEvolveRuntime(
        configuration=raw.configuration,
        thread_id=raw.thread_id,
        secret_provider_factory=resolver_factory,
        executor_validator=lambda _executor: (_ for _ in ()).throw(
            ValueError("hardened_executor_image_mismatch")
        ),
    )
    preflight_store = SQLiteAgentCallStore(tmp_path / "preflight-agent-calls.sqlite")
    with pytest.raises(ValueError, match="hardened_executor_image_mismatch"):
        assemble_metadata_only_live_openevolve(
            runtime=preflight_runtime,
            task=evidence["task"],
            component=evidence["component"],
            research_contract=evidence["research_contract"],
            run_id=evidence["context"].run_id,
            experiment_configuration=_search_configuration(maximum_model_calls=3),
            call_store=preflight_store,
            workspace_root=(tmp_path / "preflight-workspace").resolve(),
            now=lambda: NOW,
        )
    assert resolver_factory.references == []
    preflight_store.close()

    ownership_store = DispatchOwnershipDeniedStore(
        tmp_path / "ownership-agent-calls.sqlite"
    )
    ownership_operator, _ = assemble_metadata_only_live_openevolve(
        runtime=_runtime(
            tmp_path / "ownership",
            evidence,
            bridge_contract=bridge,
            approval=approval,
            secret_provider_factory=resolver_factory,
        ),
        task=evidence["task"],
        component=evidence["component"],
        research_contract=evidence["research_contract"],
        run_id=evidence["context"].run_id,
        experiment_configuration=_search_configuration(maximum_model_calls=3),
        call_store=ownership_store,
        workspace_root=(tmp_path / "ownership-workspace").resolve(),
        now=lambda: NOW,
    )
    ownership_operator.bridge.bind_search_request("ownership-search")
    with pytest.raises(LiveMutationBridgeError, match="already_dispatching"):
        ownership_operator.bridge.complete(
            _model_request(evidence, parent_id="ownership-parent"),
            "ownership-mutation",
        )
    assert resolver_factory.references == []
    ownership_store.close()


def test_secret_resolves_once_per_runtime_rotates_after_restart_and_replay_is_free(
    tmp_path, monkeypatch
):
    evidence = metadata_only_evidence()
    bridge_contract, approval = _anthropic_contract_and_approval(evidence)
    reference = SecretReference(
        logical_name="anthropic_api_key",
        provider=SecretProviderKind.GOOGLE_SECRET_MANAGER,
        provider_identifier="projects/test-project/secrets/anthropic-api-key",
        version="latest",
        required=True,
    )
    resolver_factory = CountingSecretProviderFactory(
        ["runtime-one-credential", "runtime-two-rotated-credential"]
    )
    credentials_used: list[str] = []
    clients = []

    def create_client(config, *, credential):
        credentials_used.append(credential.reveal())
        client = FakeProductionStructuredModelClient(
            provider=config.provider,
            model_id=config.model_id,
            response={
                "protocol_version": "upstream-mutation-envelope-v1",
                "mutable_file": "candidate.py",
                "source": COSINE_SOURCE,
                "description": "Offline credential lifecycle response.",
            },
        )
        clients.append(client)
        return client

    monkeypatch.setattr(
        "auto_researcher.providers.anthropic.create_anthropic_client",
        create_client,
    )
    store_path = tmp_path / "agent-calls.sqlite"
    store = SQLiteAgentCallStore(store_path)
    runtime_one = _runtime(
        tmp_path / "runtime-one",
        evidence,
        bridge_contract=bridge_contract,
        approval=approval,
        credential=reference,
        secret_provider_factory=resolver_factory,
    )
    operator_one, _ = assemble_metadata_only_live_openevolve(
        runtime=runtime_one,
        task=evidence["task"],
        component=evidence["component"],
        research_contract=evidence["research_contract"],
        run_id=evidence["context"].run_id,
        experiment_configuration=_search_configuration(maximum_model_calls=3),
        call_store=store,
        workspace_root=(tmp_path / "runtime-one-workspace").resolve(),
        now=lambda: NOW,
    )
    assert resolver_factory.references == []
    first_request = _model_request(evidence, parent_id="parent-one")
    operator_one.bridge.bind_search_request("search-one")
    first_result = operator_one.bridge.complete(first_request, "mutation-one")
    assert len(resolver_factory.references) == 1
    operator_one.bridge.bind_search_request("search-two")
    operator_one.bridge.complete(
        _model_request(evidence, parent_id="parent-two"), "mutation-two"
    )
    assert len(resolver_factory.references) == 1
    assert credentials_used == ["runtime-one-credential", "runtime-one-credential"]
    assert len(clients) == 2
    assert all(client.invocation_count == 1 for client in clients)
    store.close()

    reopened = SQLiteAgentCallStore(store_path)
    runtime_two = _runtime(
        tmp_path / "runtime-two",
        evidence,
        bridge_contract=bridge_contract,
        approval=approval,
        credential=reference,
        secret_provider_factory=resolver_factory,
    )
    operator_two, _ = assemble_metadata_only_live_openevolve(
        runtime=runtime_two,
        task=evidence["task"],
        component=evidence["component"],
        research_contract=evidence["research_contract"],
        run_id=evidence["context"].run_id,
        experiment_configuration=_search_configuration(maximum_model_calls=3),
        call_store=reopened,
        workspace_root=(tmp_path / "runtime-two-workspace").resolve(),
        now=lambda: NOW,
    )
    operator_two.bridge.bind_search_request("search-one")
    replay = operator_two.bridge.complete(first_request, "mutation-one")
    assert replay == first_result
    assert len(resolver_factory.references) == 1
    operator_two.bridge.bind_search_request("search-three")
    operator_two.bridge.complete(
        _model_request(evidence, parent_id="parent-three"), "mutation-three"
    )
    assert len(resolver_factory.references) == 2
    assert credentials_used[-1] == "runtime-two-rotated-credential"
    reopened.close()


def test_credential_reference_is_absent_from_scientific_and_model_call_identity(
    tmp_path,
):
    evidence = metadata_only_evidence()
    bridge_contract, approval = _anthropic_contract_and_approval(evidence)
    references = (
        SecretReference(
            logical_name="anthropic_api_key",
            provider=SecretProviderKind.ENVIRONMENT,
            provider_identifier="ANTHROPIC_API_KEY",
        ),
        SecretReference(
            logical_name="anthropic_api_key",
            provider=SecretProviderKind.GOOGLE_SECRET_MANAGER,
            provider_identifier="projects/test-project/secrets/anthropic-api-key",
            version="42",
        ),
    )
    call_ids = []
    context_payloads = []
    for index, reference in enumerate(references):
        provider = FakeProductionStructuredModelClient(
            provider="anthropic",
            model_id="claude-approved-20260801",
            response={
                "protocol_version": "upstream-mutation-envelope-v1",
                "mutable_file": "candidate.py",
                "source": COSINE_SOURCE,
                "description": "Credential-independent identity response.",
            },
        )
        store = SQLiteAgentCallStore(tmp_path / f"identity-{index}.sqlite")
        runtime = _runtime(
            tmp_path / f"identity-runtime-{index}",
            evidence,
            bridge_contract=bridge_contract,
            approval=approval,
            credential=reference,
            provider_factory=lambda provider=provider: provider,
        )
        operator, _ = assemble_metadata_only_live_openevolve(
            runtime=runtime,
            task=evidence["task"],
            component=evidence["component"],
            research_contract=evidence["research_contract"],
            run_id=evidence["context"].run_id,
            experiment_configuration=_search_configuration(maximum_model_calls=3),
            call_store=store,
            workspace_root=(tmp_path / f"identity-workspace-{index}").resolve(),
            now=lambda: NOW,
        )
        operator.bridge.bind_search_request("identity-search")
        operator.bridge.complete(
            _model_request(evidence, parent_id="identity-parent"),
            "identity-mutation",
        )
        call_ids.append(store.list_records()[-1].call_id)
        context_payloads.append(operator.bridge.context.model_dump_json())
        store.close()

    assert call_ids[0] == call_ids[1]
    assert context_payloads[0] == context_payloads[1]
    assert "ANTHROPIC_API_KEY" not in context_payloads[0]
    assert "projects/" not in context_payloads[0]


def test_standard_runtime_has_no_fake_provider_or_local_sandbox_fallback(tmp_path):
    evidence = metadata_only_evidence()
    store = SQLiteAgentCallStore(tmp_path / "agent-calls.sqlite")
    operator, runner = assemble_metadata_only_live_openevolve(
        runtime=_runtime(tmp_path, evidence),
        task=evidence["task"],
        component=evidence["component"],
        research_contract=evidence["research_contract"],
        run_id=evidence["context"].run_id,
        experiment_configuration=_search_configuration(),
        call_store=store,
        workspace_root=(tmp_path / "workspace").resolve(),
        now=lambda: NOW,
    )
    backend_request = SearchRequest(
        request_id="authoritative-search",
        hypothesis_id="hypothesis",
        search_type=SearchType.OPENEVOLVE,
        target="score",
        search_space=_search_configuration(),
        experiment_budget=2,
        rationale="Prove provider-unavailable failure is closed.",
    )
    from auto_researcher.search.openevolve.backend import OpenEvolveBackend

    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"
    backend = OpenEvolveBackend(
        evidence["component"],
        ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(dataset_version),
            dataset_version=dataset_version,
            provenance=ProvenanceKind.REAL,
        ),
        _search_configuration()["openevolve"]["verifier_identity"],
        operator,
        runner,
    )
    contract = backend.create_search_contract(
        backend_request, evidence["research_contract"]
    )
    seed = backend.seed_candidate(contract)
    population = backend.initialise_population(contract)
    reservation = backend.reserve_mutation(contract, population, seed)
    with pytest.raises(LiveMutationBridgeError, match="provider_unavailable"):
        backend.mutate_candidate(reservation, seed, contract)
    assert isinstance(backend.sandbox_runner, HardenedDockerExecutor)
    records = store.list_records()
    store.close()
    assert len({item.call_id for item in records}) == 1
    assert records[-1].status.value == "FAILED_BEFORE_DISPATCH"


def test_runtime_configuration_rejects_relative_paths_and_credentials(tmp_path):
    evidence = metadata_only_evidence()
    with pytest.raises(ValueError, match="paths_must_be_absolute"):
        MetadataOnlyLiveOpenEvolveConfiguration.model_validate(
            {
                "approval_file": "approval.json",
                "bridge_contract_file": "bridge.json",
                "adapter_lock_file": "lock",
                "prompt_file": "prompt.md",
                "executor_policy_file": "executor.json",
                "isolation_evidence_file": "isolation.json",
            }
        )

    from auto_researcher.cli import _load_live_openevolve_runtime

    configured = _runtime(tmp_path, evidence).configuration.model_dump(mode="json")
    configured["credentials"] = {"token": "forbidden"}
    with pytest.raises(ValueError, match="must use a secret reference"):
        _load_live_openevolve_runtime(
            {"openevolve_live_mutation": configured}, thread_id="thread"
        )


@pytest.mark.parametrize(
    "raw_field",
    [
        "api_key",
        "access_token",
        "credential_value",
        "credentials",
        "password",
        "secret_value",
        "service_account_json",
        "token",
        "provider_client",
    ],
)
def test_cli_live_runtime_rejects_raw_credential_fields_without_echo(
    tmp_path, raw_field
):
    from auto_researcher.cli import _load_live_openevolve_runtime

    configured = _runtime(tmp_path, metadata_only_evidence()).configuration.model_dump(
        mode="json"
    )
    configured[raw_field] = "never-echo-this-secret"
    with pytest.raises(ValueError) as caught:
        _load_live_openevolve_runtime(
            {"openevolve_live_mutation": configured}, thread_id="thread"
        )
    assert "never-echo-this-secret" not in str(caught.value)


def test_cli_live_runtime_accepts_default_environment_and_google_references(tmp_path):
    from auto_researcher.cli import _load_live_openevolve_runtime
    from auto_researcher.providers.anthropic import ANTHROPIC_ENVIRONMENT_SECRET

    configured = _runtime(tmp_path, metadata_only_evidence()).configuration.model_dump(
        mode="json"
    )
    configured.pop("credential")
    default = _load_live_openevolve_runtime(
        {"openevolve_live_mutation": configured}, thread_id="thread"
    )
    assert default is not None
    assert default.configuration.credential == ANTHROPIC_ENVIRONMENT_SECRET

    configured["credential"] = {
        "logical_name": "anthropic_api_key",
        "provider": "google_secret_manager",
        "provider_identifier": "projects/test-project/secrets/anthropic-api-key",
        "version": "latest",
        "required": True,
    }
    google = _load_live_openevolve_runtime(
        {"openevolve_live_mutation": configured}, thread_id="thread"
    )
    assert google is not None
    assert google.configuration.credential.provider is (
        SecretProviderKind.GOOGLE_SECRET_MANAGER
    )

    configured["credential"]["required"] = False
    with pytest.raises(ValueError, match="credentials_must_be_required"):
        _load_live_openevolve_runtime(
            {"openevolve_live_mutation": configured}, thread_id="thread"
        )

    configured["credential"] = {
        "logical_name": "anthropic_api_key",
        "provider": "google_secret_manager",
        "provider_identifier": "not-fully-qualified-and-never-echo",
        "required": True,
    }
    with pytest.raises(ValueError) as invalid:
        _load_live_openevolve_runtime(
            {"openevolve_live_mutation": configured}, thread_id="thread"
        )
    assert "not-fully-qualified-and-never-echo" not in str(invalid.value)


def test_cli_live_template_exposes_only_value_free_runtime_references():
    from auto_researcher.cli import _load_yaml

    payload = _load_yaml(
        ROOT
        / "examples/tasks/feta_seg_evolve/openevolve-live-metadata-only-template.yaml"
    )
    live = payload["openevolve_live_mutation"]
    assert set(live) == {
        "protocol_version",
        "mode",
        "approval_file",
        "bridge_contract_file",
        "adapter_lock_file",
        "prompt_file",
        "executor_policy_file",
        "isolation_evidence_file",
        "credential",
    }
    assert not {
        "api_key",
        "credentials",
        "token",
        "provider_url",
        "data_dir",
    }.intersection(live)
    assert live["credential"] == {
        "logical_name": "anthropic_api_key",
        "provider": "google_secret_manager",
        "provider_identifier": "projects/<project>/secrets/<secret>",
        "version": "latest",
        "required": True,
    }
    assert payload["search"]["openevolve"]["population_size"] == 1
    bridge = OpenEvolveModelBridgeContract.model_validate_json(
        (
            ROOT / "examples/tasks/feta_seg_evolve/model-bridge-contract-template.json"
        ).read_text(encoding="utf-8")
    )
    assert bridge.approval_policy_version == ("live-mutation-approval-v2-metadata-only")
    assert bridge.model_config_contract.maximum_attempts == 1
