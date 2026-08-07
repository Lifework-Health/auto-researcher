from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from auto_researcher.agents.call_store import InMemoryAgentCallStore
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.hardened_executor import docker_policy
from auto_researcher.search.openevolve.live_dataset import (
    ALLOWED_LIVE_MUTATION_DATASET_CLASSES,
    PROHIBITED_LIVE_MUTATION_DATASET_CLASSES,
)
from auto_researcher.search.openevolve.live_models import (
    LiveMutationApproval,
    OpenEvolveModelCallContext,
    approval_content_hash,
    validate_approval,
)
from auto_researcher.search.openevolve.production_bridge import (
    DurableOpenEvolveModelBridge,
)
from auto_researcher.search.openevolve.upstream import (
    build_approved_live_upstream_runtime,
    default_adapter_contract,
)
from auto_researcher.search.openevolve.upstream_models import ExecutorIsolationResult
from auto_researcher.tasks.icca_nbs import ICCANBSTask
from auto_researcher.tasks.iris_knn import IrisKNNTask
from auto_researcher.tasks.synthetic import SyntheticTask
from tests.unit.test_openevolve_production_bridge import (
    FakeProvider,
    NOW,
    REQUEST,
    approval_payload,
    context,
    contract,
)

ROOT = Path(__file__).parents[2]
LOCK = ROOT / "constraints/openevolve-0.3.2.lock"
RETAINED_DIGEST = (
    "sha256:11065476cf60be49b54c709b185202b9fbd3b308c44ffa2278a25259ba2b6d2c"
)


def _iris_context(**updates) -> OpenEvolveModelCallContext:
    return context().model_copy(
        update={
            "task_id": "iris_knn",
            "task_version": "1.0",
            "component_id": "iris-weighted-knn-configuration",
            "component_version": "1.0",
            "dataset_class": "public_benchmark",
            **updates,
        }
    )


def _iris_approval(**updates) -> LiveMutationApproval:
    values = {
        "task_id": "iris_knn",
        "task_version": "1.0",
        "component_id": "iris-weighted-knn-configuration",
        "component_version": "1.0",
        "permitted_dataset_class": "public_benchmark",
        **updates,
    }
    payload = approval_payload(**values)
    return LiveMutationApproval.model_validate(payload)


def _bridge(call_context, approval, *, provider_factory=None):
    return DurableOpenEvolveModelBridge(
        contract=contract(),
        context=call_context,
        approval=approval,
        store=InMemoryAgentCallStore(),
        provider_factory=provider_factory,
        now=lambda: NOW,
        system_prompt="bounded prompt-v2 fixture",
    )


def _runtime_evidence(call_context, approval):
    adapter = default_adapter_contract(LOCK)
    policy = docker_policy(
        "auto-researcher/openevolve-executor:retained",
        RETAINED_DIGEST,
        ROOT / "docker/openevolve-executor/Dockerfile",
        ROOT / "docker/openevolve-executor/worker.py",
        "fixture-runtime",
    )
    policy_hash = payload_hash(policy)
    call_context = call_context.model_copy(
        update={
            "adapter_identity_hash": payload_hash(adapter),
            "executor_policy_hash": policy_hash,
            "image_digest": policy.image_digest,
        }
    )
    payload = approval.model_dump(mode="python")
    payload.update(
        {
            "adapter_identity_hash": payload_hash(adapter),
            "executor_policy_hash": policy_hash,
            "image_digest": policy.image_digest,
        }
    )
    payload["approval_hash"] = approval_content_hash(payload)
    approval = LiveMutationApproval.model_validate(payload)
    isolation = ExecutorIsolationResult(
        executor_policy_hash=policy_hash,
        network_isolation_verified=True,
        mount_isolation_verified=True,
        environment_sanitisation_verified=True,
        safe_checks={"offline_fixture": True},
    )
    return adapter, policy, isolation, call_context, approval


def test_dataset_class_vocabulary_is_closed_and_task_owned():
    assert ALLOWED_LIVE_MUTATION_DATASET_CLASSES == {
        "synthetic",
        "public_benchmark",
    }
    assert PROHIBITED_LIVE_MUTATION_DATASET_CLASSES == {
        "aura",
        "genuine_icca",
        "mri",
        "patient_data",
    }
    assert SyntheticTask().live_mutation_dataset_class() == "synthetic"
    assert IrisKNNTask().live_mutation_dataset_class() == "public_benchmark"
    assert not hasattr(ICCANBSTask(), "live_mutation_dataset_class")


def test_public_benchmark_approval_validates_only_exact_public_context():
    public = _iris_context()
    approval = _iris_approval()
    validate_approval(approval, public, contract(), now=NOW)

    with pytest.raises(ValueError, match="live_mutation_approval_mismatch"):
        validate_approval(
            approval,
            public.model_copy(update={"dataset_class": "synthetic"}),
            contract(),
            now=NOW,
        )
    with pytest.raises(ValueError, match="live_mutation_approval_mismatch"):
        validate_approval(
            _iris_approval(permitted_dataset_class="synthetic"),
            public,
            contract(),
            now=NOW,
        )


@pytest.mark.parametrize("prohibited", ["patient_data", "genuine_icca", "mri", "aura"])
def test_prohibited_classes_cannot_instantiate_call_context(prohibited):
    payload = _iris_context().model_dump(mode="python")
    payload["dataset_class"] = prohibited
    with pytest.raises(
        ValidationError, match="Input should be 'synthetic' or 'public_benchmark'"
    ):
        OpenEvolveModelCallContext.model_validate(payload)


def test_dataset_class_is_identity_bearing_without_reinterpreting_v1():
    synthetic_context = context()
    public_context = _iris_context()
    synthetic_approval = LiveMutationApproval.model_validate(approval_payload())
    public_approval = _iris_approval()
    assert (
        synthetic_approval.protocol_version
        == public_approval.protocol_version
        == ("live-mutation-approval-v1")
    )
    assert payload_hash(synthetic_context) != payload_hash(public_context)
    assert synthetic_approval.approval_hash != public_approval.approval_hash

    synthetic_calls: list[str] = []
    public_calls: list[str] = []
    _bridge(
        synthetic_context,
        synthetic_approval,
        provider_factory=lambda: FakeProvider(synthetic_calls),
    ).complete(REQUEST, "dataset-class-identity")
    _bridge(
        synthetic_context.model_copy(update={"dataset_class": "public_benchmark"}),
        LiveMutationApproval.model_validate(
            approval_payload(permitted_dataset_class="public_benchmark")
        ),
        provider_factory=lambda: FakeProvider(public_calls),
    ).complete(REQUEST, "dataset-class-identity")
    assert synthetic_calls != public_calls


def test_runtime_assembly_uses_trusted_task_classification_and_fails_closed(tmp_path):
    constructions = 0

    def provider_factory():
        nonlocal constructions
        constructions += 1
        raise AssertionError("provider must remain lazy")

    evidence = _runtime_evidence(_iris_context(), _iris_approval())
    adapter, policy, isolation, call_context, approval = evidence
    build_approved_live_upstream_runtime(
        adapter,
        _bridge(call_context, approval, provider_factory=provider_factory),
        policy,
        isolation,
        task=IrisKNNTask(),
        workspace_root=tmp_path / "workspace",
    )
    assert constructions == 0

    class InvalidClassTask:
        task_id = "iris_knn"
        task_version = "1.0"

        def live_mutation_dataset_class(self):
            return "patient_data"

    for task, code in (
        (
            SimpleNamespace(task_id="iris_knn", task_version="1.0"),
            "live_mutation_dataset_class_unavailable",
        ),
        (InvalidClassTask(), "live_mutation_dataset_class_unavailable"),
        (ICCANBSTask(), "live_mutation_dataset_class_unavailable"),
        (SyntheticTask(), "live_mutation_approval_mismatch"),
    ):
        with pytest.raises(ValueError, match=code):
            build_approved_live_upstream_runtime(
                adapter,
                _bridge(call_context, approval, provider_factory=provider_factory),
                policy,
                isolation,
                task=task,  # type: ignore[arg-type]
            )
    assert constructions == 0

    synthetic_evidence = _runtime_evidence(
        context(), LiveMutationApproval.model_validate(approval_payload())
    )
    adapter, policy, isolation, call_context, approval = synthetic_evidence
    with pytest.raises(ValueError, match="live_mutation_approval_mismatch"):
        build_approved_live_upstream_runtime(
            adapter,
            _bridge(call_context, approval, provider_factory=provider_factory),
            policy,
            isolation,
            task=IrisKNNTask(),
        )
    assert constructions == 0


def test_hostile_untrusted_fields_cannot_widen_classification():
    task = IrisKNNTask()
    hostile_payloads = (
        {"task_options": {"dataset_class": "patient_data"}},
        {"search_space": {"dataset_class": "synthetic"}},
        {"provider_response": {"dataset_class": "aura"}},
        {"candidate_source": "dataset_class = 'genuine_icca'"},
    )
    for payload in hostile_payloads:
        assert payload
        assert task.live_mutation_dataset_class() == "public_benchmark"

    context_payload = _iris_context().model_dump(mode="python")
    context_payload["dataset_class_override"] = "patient_data"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OpenEvolveModelCallContext.model_validate(context_payload)
