from __future__ import annotations

import copy
from datetime import timedelta
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_researcher.agents.call_store import InMemoryAgentCallStore
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.hardened_executor import (
    HardenedDockerExecutor,
    docker_policy,
)
from auto_researcher.search.openevolve.identity import component_interface_identity
from auto_researcher.search.openevolve.live_boundary import (
    assert_no_prohibited_dynamic_content,
    metadata_only_model_exposure_identity,
)
from auto_researcher.search.openevolve.live_models import (
    LiveMutationApproval,
    MetadataOnlyLiveMutationApproval,
    MetadataOnlyOpenEvolveModelCallContext,
    OpenEvolveModelBridgeContract,
    metadata_only_approval_content_hash,
    parse_live_mutation_approval,
)
from auto_researcher.search.openevolve.production_bridge import (
    DurableOpenEvolveModelBridge,
    LiveMutationBridgeError,
)
from auto_researcher.search.openevolve.upstream import (
    build_approved_live_upstream_runtime,
    default_adapter_contract,
    mutation_constraints,
)
from auto_researcher.search.openevolve.upstream_models import ExecutorIsolationResult
from auto_researcher.tasks.feta_seg_evolve import (
    FeTASegEvolveTask,
    default_feta_evolve_contract,
)
from auto_researcher.tasks.models import TaskRuntimeContext
from tests.unit.test_openevolve_production_bridge import (
    NOW,
    approval,
    approval_payload,
    context,
    contract,
)

ROOT = Path(__file__).parents[2]
LOCK = ROOT / "constraints/openevolve-0.3.2.lock"
PROMPT = "Attested metadata-only FeTA mutation prompt."
RETAINED_DIGEST = (
    "sha256:11065476cf60be49b54c709b185202b9fbd3b308c44ffa2278a25259ba2b6d2c"
)


def metadata_only_contract() -> OpenEvolveModelBridgeContract:
    payload = contract().model_dump(mode="python", by_alias=True)
    payload["approval_policy_version"] = "live-mutation-approval-v2-metadata-only"
    return OpenEvolveModelBridgeContract.model_validate(payload)


def metadata_only_evidence():
    task = FeTASegEvolveTask()
    research_contract = default_feta_evolve_contract(maximum_experiments=2)
    component = task.create_evolvable_component(
        research_contract,
        TaskRuntimeContext(
            data_dir=Path("/must-not-cross-boundary/feta"),
            task_options={
                "hpo_observations": ["Aggregate HPO summary available."]
            },
        ),
    )
    spec = component.component_spec()
    adapter = default_adapter_contract(LOCK)
    policy = docker_policy(
        "auto-researcher/openevolve-executor:retained",
        RETAINED_DIGEST,
        ROOT / "docker/openevolve-executor/Dockerfile",
        ROOT / "docker/openevolve-executor/worker.py",
        "fixture-runtime",
    )
    policy_hash = payload_hash(policy)
    interface_identity = component_interface_identity(spec)
    exposure_identity = metadata_only_model_exposure_identity(spec)
    boundary = task.live_mutation_boundary()
    call_context = MetadataOnlyOpenEvolveModelCallContext(
        run_id="feta-metadata-only-run",
        thread_id="feta-metadata-only-thread",
        contract_id=research_contract.contract_id,
        contract_hash=payload_hash(research_contract),
        task_id=task.task_id,
        task_version=task.task_version,
        search_request_id="feta-metadata-only-search",
        generation=1,
        parent_candidate_id="seed-placeholder",
        component_id=spec.component_id,
        component_version=spec.component_version,
        component_interface_hash=interface_identity,
        model_exposure_identity=exposure_identity,
        underlying_dataset_class="mri",
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.adapter_version,
        adapter_identity_hash=payload_hash(adapter),
        executor_policy_hash=policy_hash,
        image_digest=policy.image_digest,
        mutable_file=spec.mutable_file,
        model_budget_identity="feta-metadata-only-budget",
        maximum_model_calls=1,
        maximum_model_cost=0.02,
    )
    approval_payload_v2 = {
        "protocol_version": "live-mutation-approval-v2-metadata-only",
        "approval_id": "feta-metadata-only-approval",
        "run_id": call_context.run_id,
        "contract_id": call_context.contract_id,
        "contract_hash": call_context.contract_hash,
        "task_id": task.task_id,
        "task_version": task.task_version,
        "component_id": spec.component_id,
        "component_version": spec.component_version,
        "component_interface_hash": interface_identity,
        "model_exposure_identity": exposure_identity,
        "underlying_dataset_class": "mri",
        "exposure_class": "metadata_only",
        "adapter_id": adapter.adapter_id,
        "adapter_version": adapter.adapter_version,
        "adapter_identity_hash": payload_hash(adapter),
        "provider": "fake-production",
        "model_id": "fake-model-20260101",
        "prompt_id": "openevolve-mutation",
        "prompt_version": "openevolve-mutation-prompt-v2",
        "prompt_hash": payload_hash(PROMPT),
        "mutation_operator_version": "upstream-openevolve-adapter-v1",
        "maximum_model_calls": 1,
        "maximum_input_tokens": 20_000,
        "maximum_output_tokens": 1_000,
        "maximum_total_cost": 0.02,
        "currency": "USD",
        "pricing_version": "fake-pricing-v1",
        "executor_policy_hash": policy_hash,
        "image_digest": policy.image_digest,
        "mutable_file": spec.mutable_file,
        "created_at": NOW - timedelta(minutes=1),
        "expires_at": NOW + timedelta(minutes=10),
        "reviewer_identity": "operator-metadata-only",
        "residual_risk_acknowledged": True,
    }
    approval_payload_v2["approval_hash"] = metadata_only_approval_content_hash(
        approval_payload_v2
    )
    approval_v2 = MetadataOnlyLiveMutationApproval.model_validate(approval_payload_v2)
    isolation = ExecutorIsolationResult(
        executor_policy_hash=policy_hash,
        network_isolation_verified=True,
        mount_isolation_verified=True,
        environment_sanitisation_verified=True,
        safe_checks={"offline_fixture": True},
    )
    return {
        "task": task,
        "research_contract": research_contract,
        "component": component,
        "spec": spec,
        "adapter": adapter,
        "policy": policy,
        "isolation": isolation,
        "boundary": boundary,
        "context": call_context,
        "approval": approval_v2,
        "approval_payload": approval_payload_v2,
    }


def make_bridge(evidence, *, provider_factory=None, prompt=PROMPT):
    return DurableOpenEvolveModelBridge(
        contract=metadata_only_contract(),
        context=evidence["context"],
        approval=evidence["approval"],
        store=InMemoryAgentCallStore(),
        provider_factory=provider_factory,
        now=lambda: NOW,
        system_prompt=prompt,
        metadata_only_boundary=evidence["boundary"],
    )


def model_request(evidence):
    spec = evidence["spec"]
    return {
        "protocol": "upstream-adapter-mutation-request-v2",
        "parent": {
            "id": "upstream-seed",
            "authoritative_candidate_id": "candidate-seed",
            "code": spec.seed_source,
            "generation": 0,
        },
        "mutable_file": spec.mutable_file,
        "interface_contract": spec.immutable_interface_contract,
        "maximum_source_bytes": spec.maximum_source_bytes,
        "mutation_constraints": mutation_constraints(spec).model_dump(mode="json"),
    }


def test_v1_approval_hash_schema_and_runtime_path_remain_exactly_compatible(tmp_path):
    assert approval().approval_hash == (
        "aae59c1136e59965443a59d32b3cfde8d854bd2d07db14cc132b2b9276bae678"
    )
    parsed = parse_live_mutation_approval(approval_payload())
    assert type(parsed) is LiveMutationApproval
    assert parsed.protocol_version == "live-mutation-approval-v1"
    assert contract().approval_policy_version == "live-mutation-approval-v1"

    evidence = metadata_only_evidence()
    with pytest.raises(ValueError, match="live_mutation_dataset_class_unavailable"):
        build_approved_live_upstream_runtime(
            evidence["adapter"],
            DurableOpenEvolveModelBridge(
                contract=contract(),
                context=context().model_copy(
                    update={
                        "task_id": "feta_seg_evolve",
                        "task_version": "1.0",
                    }
                ),
                approval=approval(
                    task_id="feta_seg_evolve",
                    task_version="1.0",
                ),
                store=InMemoryAgentCallStore(),
                provider_factory=None,
                now=lambda: NOW,
                system_prompt="bounded prompt",
            ),
            evidence["policy"],
            evidence["isolation"],
            task=evidence["task"],
            component_spec=evidence["spec"],
            workspace_root=tmp_path,
        )


def test_metadata_only_approval_is_distinct_identity_and_old_approval_cannot_upgrade():
    evidence = metadata_only_evidence()
    parsed = parse_live_mutation_approval(evidence["approval_payload"])
    assert type(parsed) is MetadataOnlyLiveMutationApproval
    assert parsed.approval_hash != approval().approval_hash
    assert parsed.underlying_dataset_class == "mri"
    assert parsed.exposure_class == "metadata_only"
    assert parsed.mri_access is False
    assert parsed.patient_data_access is False
    assert parsed.filesystem_access is False
    assert parsed.network_access is False

    old = approval_payload()
    old.update(
        {
            "underlying_dataset_class": "mri",
            "exposure_class": "metadata_only",
            "component_interface_hash": evidence["context"].component_interface_hash,
            "model_exposure_identity": evidence["context"].model_exposure_identity,
        }
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LiveMutationApproval.model_validate(old)


@pytest.mark.parametrize(
    "field",
    (
        "underlying_data_access",
        "mri_access",
        "patient_data_access",
        "filesystem_access",
        "network_access",
        "evaluator_runtime_context_access",
        "direct_upstream_provider_access",
        "local_subprocess_fallback",
        "model_retries",
        "package_installation",
        "multiple_mutable_files",
        "evaluator_or_verifier_mutation",
    ),
)
def test_metadata_only_approval_cannot_enable_any_access_axis(field):
    evidence = metadata_only_evidence()
    payload = dict(evidence["approval_payload"])
    payload[field] = True
    payload["approval_hash"] = "0" * 64
    with pytest.raises(ValidationError):
        MetadataOnlyLiveMutationApproval.model_validate(payload)


@pytest.mark.parametrize(
    "value",
    (
        {"path": "/protected/feta"},
        {"voxels": [1, 2]},
        {"masks": [1, 2]},
        {"subject_records": [{"id": "sub-001"}]},
        {"case_records": [{"id": 7}]},
        {"predictions": [0, 1]},
        {"checkpoint": "best.pt"},
        {"holdout": {"score": 0.9}},
        {"message": "MRI scan"},
    ),
)
def test_metadata_only_dynamic_content_vocabulary_fails_closed(value):
    with pytest.raises(ValueError, match="metadata_only_prohibited_content"):
        assert_no_prohibited_dynamic_content(value)


def test_metadata_only_cost_budgets_must_be_finite():
    evidence = metadata_only_evidence()
    payload = dict(evidence["approval_payload"])
    payload["maximum_total_cost"] = float("inf")
    payload["approval_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="finite_budget"):
        MetadataOnlyLiveMutationApproval.model_validate(payload)
    context_payload = evidence["context"].model_dump(mode="python")
    context_payload["maximum_model_cost"] = float("inf")
    with pytest.raises(ValidationError, match="finite_budget"):
        MetadataOnlyOpenEvolveModelCallContext.model_validate(context_payload)


@pytest.mark.parametrize(
    "field",
    (
        "component_interface_hash",
        "model_exposure_identity",
        "underlying_dataset_class",
        "exposure_class",
        "executor_policy_hash",
        "prompt_hash",
        "mutable_file",
    ),
)
def test_every_metadata_only_boundary_identity_is_approval_bound(field):
    evidence = metadata_only_evidence()
    payload = dict(evidence["approval_payload"])
    if field in {
        "component_interface_hash",
        "model_exposure_identity",
        "executor_policy_hash",
        "prompt_hash",
    }:
        payload[field] = "f" * 64
    elif field == "underlying_dataset_class":
        payload[field] = "patient_data"
    elif field == "exposure_class":
        payload[field] = "full_data"
    else:
        payload[field] = "other.py"
    if field == "exposure_class":
        with pytest.raises(ValidationError):
            payload["approval_hash"] = "0" * 64
            MetadataOnlyLiveMutationApproval.model_validate(payload)
        return
    payload["approval_hash"] = metadata_only_approval_content_hash(payload)
    changed = MetadataOnlyLiveMutationApproval.model_validate(payload)
    evidence["approval"] = changed
    if field == "prompt_hash":
        bridge = make_bridge(evidence)
        with pytest.raises(
            LiveMutationBridgeError, match="live_mutation_approval_mismatch"
        ):
            bridge.complete(model_request(evidence), "prompt-mismatch")
        assert bridge.store.list_records() == ()
        return
    with pytest.raises(ValueError, match="live_mutation_approval_mismatch"):
        build_approved_live_upstream_runtime(
            evidence["adapter"],
            make_bridge(evidence),
            evidence["policy"],
            evidence["isolation"],
            task=evidence["task"],
            component_spec=evidence["spec"],
        )


def test_runtime_recomputes_attestation_and_returns_hardened_executor(tmp_path):
    evidence = metadata_only_evidence()
    adapter, executor = build_approved_live_upstream_runtime(
        evidence["adapter"],
        make_bridge(evidence),
        evidence["policy"],
        evidence["isolation"],
        task=evidence["task"],
        component_spec=evidence["spec"],
        workspace_root=tmp_path,
    )
    assert adapter.operator_version == "upstream-openevolve-adapter-v1"
    assert isinstance(executor, HardenedDockerExecutor)
    assert executor.runner_id == "openevolve-hardened-executor-v2"
    assert evidence["task"].live_mutation_boundary().underlying_dataset_class == "mri"
    assert not hasattr(evidence["task"], "live_mutation_dataset_class")


@pytest.mark.parametrize(
    "smuggle",
    (
        {"runtime_context": {"data_dir": "/protected/feta"}},
        {"exception_message": "subject 001 failed"},
        {"nested": {"prediction": [1, 2, 3]}},
    ),
)
def test_extra_or_nested_model_input_is_rejected_before_provider(smuggle):
    evidence = metadata_only_evidence()
    request = model_request(evidence)
    request.update(smuggle)
    constructions = 0

    def provider_factory():
        nonlocal constructions
        constructions += 1
        raise AssertionError("provider must not be constructed")

    with pytest.raises(
        LiveMutationBridgeError, match="metadata_only_model_input_rejected"
    ):
        make_bridge(evidence, provider_factory=provider_factory).complete(
            request, "smuggle-input"
        )
    assert constructions == 0


def test_nested_attested_context_tampering_is_rejected_before_provider():
    evidence = metadata_only_evidence()
    request = model_request(evidence)
    request = copy.deepcopy(request)
    request["mutation_constraints"]["parameter_schema"]["mutation_context"][
        "nested"
    ] = {"case_records": ["case 7"]}
    constructions = 0

    def provider_factory():
        nonlocal constructions
        constructions += 1
        raise AssertionError("provider must not be constructed")

    with pytest.raises(
        LiveMutationBridgeError, match="metadata_only_model_input_rejected"
    ):
        make_bridge(evidence, provider_factory=provider_factory).complete(
            request, "nested-context-smuggle"
        )
    assert constructions == 0


def test_unsafe_static_component_context_cannot_be_attested():
    evidence = metadata_only_evidence()
    parameter_schema = json.loads(json.dumps(evidence["spec"].parameter_schema))
    mutation_context = copy.deepcopy(parameter_schema["mutation_context"])
    mutation_context["nested"] = {"subject_records": [{"id": "sub-001"}]}
    parameter_schema["mutation_context"] = mutation_context
    unsafe = evidence["spec"].model_copy(
        update={
            "parameter_schema": parameter_schema,
            "task_mutation_context": mutation_context,
        }
    )
    with pytest.raises(ValueError, match="metadata_only_prohibited_content"):
        metadata_only_model_exposure_identity(unsafe)


def test_configuration_and_hpo_observation_smuggling_fail_closed():
    task = FeTASegEvolveTask()
    contract_item = default_feta_evolve_contract()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        task.create_evolvable_component(
            contract_item,
            TaskRuntimeContext(
                task_options={"base_configuration": {"data_dir": "/data/feta"}}
            ),
        )
    with pytest.raises(ValueError, match="feta_evolve_hpo_observations_invalid"):
        task.create_evolvable_component(
            contract_item,
            TaskRuntimeContext(
                task_options={"hpo_observations": ["subject 001 had low Dice"]}
            ),
        )


def test_model_output_smuggling_is_rejected_as_safe_code_only():
    from auto_researcher.providers.fake_production import (
        FakeProductionStructuredModelClient,
    )

    evidence = metadata_only_evidence()
    output = {
        "protocol_version": "upstream-mutation-envelope-v1",
        "mutable_file": "candidate.py",
        "source": "def evolve(configuration):\n    return {'path': '/data/feta'}\n",
        "description": "Use subject records from the checkpoint.",
    }
    provider = FakeProductionStructuredModelClient(
        provider="fake-production",
        model_id="fake-model-20260101",
        response=output,
    )
    bridge = make_bridge(evidence, provider_factory=lambda: provider)
    with pytest.raises(LiveMutationBridgeError, match="model_call_response_unsafe"):
        bridge.complete(model_request(evidence), "smuggle-output")
    records = bridge.store.list_records()
    assert records[-1].error_code is not None
    assert "/data/feta" not in "\n".join(item.model_dump_json() for item in records)


def test_provider_exception_content_is_not_forwarded_or_persisted():
    evidence = metadata_only_evidence()

    def provider_factory():
        raise RuntimeError("subject 001 /data/feta checkpoint failed")

    bridge = make_bridge(evidence, provider_factory=provider_factory)
    with pytest.raises(LiveMutationBridgeError) as caught:
        bridge.complete(model_request(evidence), "unsafe-exception")
    assert caught.value.code == "model_call_provider_unavailable"
    persisted = "\n".join(
        item.model_dump_json() for item in bridge.store.list_records()
    )
    assert "subject 001" not in persisted
    assert "/data/feta" not in persisted
