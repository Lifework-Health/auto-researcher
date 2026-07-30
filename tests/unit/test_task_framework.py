from __future__ import annotations

from datetime import UTC, datetime

import pytest

from auto_researcher.contracts.enums import EvidenceStatus, ProvenanceKind, SearchType
from auto_researcher.contracts.models import EvaluationResult, ExperimentSpec
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.search.direct import DirectSearchBackend
from auto_researcher.tasks import ResearchTask, VerificationPolicy
from auto_researcher.tasks.artifacts import artefact_references
from auto_researcher.tasks.icca_nbs import ICCANBSTask
from auto_researcher.tasks.models import (
    DuplicateTaskError,
    ExperimentMetadata,
    PolicyDecision,
    ReadinessCheck,
    ReadinessResult,
    TaskNotReadyError,
    TaskRuntimeContext,
    UnknownTaskError,
)
from auto_researcher.tasks.registry import TaskRegistry
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)
from auto_researcher.tasks.synthetic.verification import SyntheticVerificationPolicy
from auto_researcher.verification.verifier import DeterministicVerifier


def test_task_descriptors_validate_and_serialise():
    for task in (SyntheticTask(), ICCANBSTask()):
        descriptor = task.descriptor()
        assert descriptor.model_validate_json(descriptor.model_dump_json()) == descriptor
        assert descriptor.supported_search_types


def test_task_models_are_deeply_immutable():
    context = TaskRuntimeContext(
        environment={"MODE": "offline"},
        task_options={"nested": {"value": 1}},
    )
    with pytest.raises(TypeError, match="immutable"):
        context.environment["MODE"] = "live"
    with pytest.raises(TypeError, match="immutable"):
        context.task_options["nested"]["value"] = 2
    manifest = SyntheticTask().dataset_manifest(context)
    with pytest.raises(TypeError, match="immutable"):
        manifest.hashes["new"] = "hash"
    with pytest.raises(TypeError, match="immutable"):
        manifest.metadata["generator"] = "changed"


def test_duplicate_registration_fails():
    registry = TaskRegistry()
    registry.register(SyntheticTask)
    with pytest.raises(DuplicateTaskError, match="already registered"):
        registry.register(SyntheticTask)


def test_unknown_task_lookup_fails_clearly():
    registry = TaskRegistry()
    registry.register(SyntheticTask)
    with pytest.raises(UnknownTaskError, match="unknown research task"):
        registry.get("missing")


def test_task_version_selection_is_deterministic():
    class SyntheticV2(SyntheticTask):
        task_version = "2.0"

        def descriptor(self):
            return super().descriptor().model_copy(update={"task_version": "2.0"})

    registry = TaskRegistry()
    registry.register(SyntheticV2)
    registry.register(SyntheticTask)
    assert registry.get("synthetic").task_version == "2.0"
    assert registry.get("synthetic", "1.0").task_version == "1.0"


def test_contract_carries_task_identity_and_mismatch_fails():
    contract = default_synthetic_contract()
    assert (contract.task_id, contract.task_version) == ("synthetic", "1.0")
    mismatched = contract.model_copy(update={"task_id": "icca_nbs"})
    with pytest.raises(ValueError, match="not synthetic"):
        SyntheticTask().validate_contract(mismatched)


def test_contract_provenance_must_match_task_metadata():
    contract = default_synthetic_contract().model_copy(
        update={"provenance": ProvenanceKind.REAL}
    )
    with pytest.raises(ValueError, match="contract provenance"):
        task_memory_dependencies(
            SyntheticTask(),
            TaskRuntimeContext(),
            contract,
            default_synthetic_configuration(),
        )


def test_task_runtime_context_is_not_persisted_in_graph_state(tmp_path):
    contract = default_synthetic_contract()
    context = TaskRuntimeContext(
        run_id="runtime-separation",
        data_dir=tmp_path / "private-data",
        output_dir=tmp_path / "outputs",
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        context,
        contract,
        default_synthetic_configuration(),
    )
    final = build_graph(dependencies).invoke(
        {
            "run_id": "runtime-separation",
            "thread_id": "runtime-thread",
            "contract": contract,
        },
        {"configurable": {"thread_id": "runtime-thread"}},
    )
    assert "runtime_context" not in final
    assert str(context.data_dir) not in repr(final)
    assert str(context.data_dir) not in repr(
        dependencies.provenance_store.list_events("runtime-separation")
    )


def test_runtime_and_artefact_identifiers_reject_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="path-safe"):
        TaskRuntimeContext(
            run_id="../outside",
            output_dir=tmp_path,
        )
    context = TaskRuntimeContext(run_id="safe-run", output_dir=tmp_path)
    with pytest.raises(ValueError, match="experiment_id"):
        artefact_references(context, "../../outside")


def test_readiness_failure_stops_before_dependency_assembly():
    class NotReadySynthetic(SyntheticTask):
        def readiness(self, runtime_context):
            return ReadinessResult(
                ready=False,
                checks=(
                    ReadinessCheck(
                        code="blocked",
                        passed=False,
                        message="required resource unavailable",
                    ),
                ),
                errors=("required resource unavailable",),
            )

        def create_evaluator(self, runtime_context):
            raise AssertionError("evaluator must not be created")

    with pytest.raises(TaskNotReadyError, match="required resource unavailable"):
        task_memory_dependencies(
            NotReadySynthetic(),
            TaskRuntimeContext(),
            default_synthetic_contract(),
            default_synthetic_configuration(),
        )


def test_experiment_metadata_and_normalisation_come_from_task():
    calls = {"normalise": 0}

    class CountingSynthetic(SyntheticTask):
        def normalise_configuration(self, configuration):
            calls["normalise"] += 1
            return super().normalise_configuration(configuration)

    task = CountingSynthetic()
    contract = default_synthetic_contract()
    dependencies = task_memory_dependencies(
        task,
        TaskRuntimeContext(),
        contract,
        default_synthetic_configuration(),
    )
    metadata = task.experiment_metadata(TaskRuntimeContext())
    assert dependencies.direct_search_backend.metadata == metadata
    assert calls["normalise"] == 1


def test_direct_backend_is_task_neutral_and_invokes_normaliser():
    calls = []
    metadata = ExperimentMetadata(
        evaluator_id="custom-evaluator",
        code_version="custom-code",
        dataset_version="custom-data",
        provenance=ProvenanceKind.REAL,
    )

    def normalise(configuration):
        calls.append(configuration)
        return {"custom_parameter": str(configuration["custom_parameter"]).upper()}

    backend = DirectSearchBackend(metadata, normalise)
    from auto_researcher.contracts.models import SearchRequest

    request = SearchRequest(
        request_id="request",
        hypothesis_id="hypothesis",
        search_type=SearchType.DIRECT,
        target="custom_metric",
        search_space={"custom_parameter": ["value"]},
        experiment_budget=1,
        rationale="task-defined",
    )
    experiment = backend.create_experiment(
        request,
        default_synthetic_contract(),
        run_id="run",
    )
    assert calls == [{"custom_parameter": "value"}]
    assert experiment.configuration == {"custom_parameter": "VALUE"}
    assert experiment.dataset_version == "custom-data"


class _CountingPolicy:
    policy_id = "counting-policy"
    required_metrics = frozenset({"objective_score"})

    def __init__(self):
        self.calls = 0

    def evaluate_constraints(self, evaluation, contract):
        self.calls += 1
        return PolicyDecision(
            constraint_compliant=True,
            evidence_status=EvidenceStatus.SUPPORTED,
        )


def _verification_pair(provenance=ProvenanceKind.REAL):
    experiment = ExperimentSpec(
        experiment_id="experiment",
        hypothesis_id="hypothesis",
        search_request_id="request",
        configuration={"value": 1},
        evaluator_id="synthetic-evaluator",
        code_version="code",
        dataset_version="data",
        provenance=provenance,
    )
    evaluation = EvaluationResult(
        experiment_id="experiment",
        success=True,
        primary_score=0.8,
        metrics={"objective_score": 0.8},
        constraint_results={"valid": True},
        evaluator_version="v1",
        provenance=provenance,
    )
    return experiment, evaluation


def test_structural_verifier_runs_before_task_policy():
    policy = _CountingPolicy()
    experiment, evaluation = _verification_pair()
    evaluation = evaluation.model_copy(update={"experiment_id": "other"})
    result = DeterministicVerifier(policy).verify(
        experiment,
        evaluation,
        default_synthetic_contract(),
    )
    assert result.verified is False
    assert policy.calls == 0
    assert "experiment_result_mismatch" in result.reasons


def test_task_policy_cannot_support_mock_evidence():
    policy = _CountingPolicy()
    experiment, evaluation = _verification_pair(ProvenanceKind.MOCK)
    result = DeterministicVerifier(policy).verify(
        experiment,
        evaluation,
        default_synthetic_contract(),
    )
    assert policy.calls == 1
    assert result.evidence_status == EvidenceStatus.INCONCLUSIVE
    assert "synthetic_evidence_cannot_support" in result.reasons


def test_different_policies_share_the_same_structural_verifier():
    experiment, evaluation = _verification_pair()
    supporting = DeterministicVerifier(_CountingPolicy()).verify(
        experiment,
        evaluation,
        default_synthetic_contract(),
    )

    class RefutingPolicy(_CountingPolicy):
        def evaluate_constraints(self, evaluation, contract):
            return PolicyDecision(
                constraint_compliant=False,
                evidence_status=EvidenceStatus.REFUTED,
                reasons=("task_refuted",),
            )

    refuting = DeterministicVerifier(RefutingPolicy()).verify(
        experiment,
        evaluation,
        default_synthetic_contract(),
    )
    assert supporting.evidence_status == EvidenceStatus.SUPPORTED
    assert refuting.evidence_status == EvidenceStatus.REFUTED


def test_synthetic_and_icca_tasks_satisfy_same_protocol():
    assert isinstance(SyntheticTask(), ResearchTask)
    assert isinstance(ICCANBSTask(), ResearchTask)
    assert isinstance(SyntheticVerificationPolicy(), VerificationPolicy)
