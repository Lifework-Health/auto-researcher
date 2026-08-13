from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_researcher.agents.call_store import SQLiteAgentCallStore
from auto_researcher.contracts.enums import ProvenanceKind, SearchType
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
from auto_researcher.search.openevolve.production_bridge import LiveMutationBridgeError
from auto_researcher.search.openevolve.upstream import UpstreamOpenEvolveAdapter
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


def _runtime(tmp_path: Path, evidence, *, bridge_contract=None, approval=None):
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text(PROMPT, encoding="utf-8")
    configuration = MetadataOnlyLiveOpenEvolveConfiguration(
        approval_file=_write_json(
            tmp_path / "approval.json", approval or evidence["approval"]
        ),
        bridge_contract_file=_write_json(
            tmp_path / "bridge.json", bridge_contract or metadata_only_contract()
        ),
        adapter_lock_file=ROOT / "constraints/openevolve-0.3.2.lock",
        prompt_file=prompt_file,
        executor_policy_file=_write_json(
            tmp_path / "executor.json", evidence["policy"]
        ),
        isolation_evidence_file=_write_json(
            tmp_path / "isolation.json", evidence["isolation"]
        ),
    )
    return MetadataOnlyLiveOpenEvolveRuntime(
        configuration=configuration,
        thread_id=evidence["context"].thread_id,
        executor_validator=lambda executor: None,
    )


def _search_configuration():
    configuration = default_feta_evolve_openevolve_configuration()
    configuration["openevolve"].update(
        {
            "maximum_generations": 1,
            "maximum_candidate_evaluations": 2,
            "maximum_model_calls": 1,
            "sandbox_policy_id": "openevolve-hardened-executor-v2",
        }
    )
    return configuration


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
    with pytest.raises(ValueError, match="credentials must come from the environment"):
        _load_live_openevolve_runtime(
            {"openevolve_live_mutation": configured}, thread_id="thread"
        )


def test_cli_live_template_exposes_only_credential_free_runtime_artifact_paths():
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
    }
    assert not {
        "api_key",
        "credentials",
        "token",
        "provider_url",
        "data_dir",
    }.intersection(live)
    bridge = OpenEvolveModelBridgeContract.model_validate_json(
        (
            ROOT / "examples/tasks/feta_seg_evolve/model-bridge-contract-template.json"
        ).read_text(encoding="utf-8")
    )
    assert bridge.approval_policy_version == ("live-mutation-approval-v2-metadata-only")
    assert bridge.model_config_contract.maximum_attempts == 1
