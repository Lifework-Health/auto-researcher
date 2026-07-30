from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import replace
import importlib.util

import pytest

from auto_researcher.contracts.enums import (
    EvidenceStatus,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import (
    EvaluationResult,
    SearchRequest,
    VerificationResult,
)
from auto_researcher.search.optuna.backend import OptunaAskTellBackend
from auto_researcher.search.optuna.models import OptunaTrialStatus
from auto_researcher.search.optuna.naming import build_study_identity
from auto_researcher.search.optuna.recovery import (
    ConflictingTrialReportError,
    StudyIdentityMismatchError,
)
from auto_researcher.search.optuna.storage import in_memory_storage
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import SyntheticTask, default_synthetic_contract

NOW = datetime(2026, 7, 30, tzinfo=UTC)
pytestmark = [
    pytest.mark.hpo,
    pytest.mark.skipif(
        importlib.util.find_spec("optuna") is None,
        reason="install the hpo extra to run Optuna backend tests",
    ),
]


def setup_backend():
    task = SyntheticTask()
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPTUNA}),
        maximum_experiments=3,
    )
    request = SearchRequest(
        request_id="request",
        hypothesis_id="hypothesis",
        search_type=SearchType.OPTUNA,
        target="optimise",
        search_space={"trial_budget": 3, "seed": 11},
        experiment_budget=3,
        rationale="test",
    )
    spec = task.create_optuna_study_spec(contract, request)
    metadata = task.experiment_metadata(
        TaskRuntimeContext(manifest_created_at=NOW)
    )
    identity = build_study_identity(
        run_id="run",
        contract=contract,
        request=request,
        metadata=metadata,
        spec=spec,
    )
    backend = OptunaAskTellBackend(in_memory_storage().storage)
    backend.prepare_or_load_study(
        identity,
        spec,
        started_at=NOW,
        trial_budget=3,
    )
    return backend, task, contract, request, spec, metadata, identity


def result(experiment, score=0.5, *, success=True):
    return EvaluationResult(
        experiment_id=experiment.experiment_id,
        success=success,
        primary_score=score if success else None,
        metrics={"objective_score": score, "stability": 0.9, "runtime": 1.0}
        if success
        else {},
        constraint_results={"ok": True} if success else {},
        evaluator_version="test",
        provenance=ProvenanceKind.SIMULATED,
        error=None if success else "failed",
    )


def verification(experiment, score=0.5, *, verified=True, feasible=True):
    return VerificationResult(
        experiment_id=experiment.experiment_id,
        verified=verified,
        claimed_score=score,
        measured_score=score,
        constraint_compliant=feasible,
        evidence_status=EvidenceStatus.INCONCLUSIVE,
        reasons=(),
        provenance=ProvenanceKind.SIMULATED,
    )


def test_ask_samples_all_parameters_and_recovers_without_duplicate():
    backend, _, _, _, spec, _, identity = setup_backend()
    first, recovered = backend.ask_or_recover_trial(
        identity, spec, slot_index=0, asked_at=NOW
    )
    second, recovered_again = backend.ask_or_recover_trial(
        identity, spec, slot_index=0, asked_at=NOW
    )
    assert recovered is False
    assert recovered_again is True
    assert first.trial_number == second.trial_number
    assert set(first.parameters) == {
        "model_family",
        "complexity",
        "learning_rate",
    }


@pytest.mark.parametrize(
    ("success", "verified", "feasible", "expected_status", "expected_feasible"),
    [
        (True, True, True, OptunaTrialStatus.COMPLETE, True),
        (True, True, False, OptunaTrialStatus.COMPLETE, False),
        (False, False, False, OptunaTrialStatus.FAIL, False),
        (True, False, False, OptunaTrialStatus.FAIL, False),
    ],
)
def test_tell_state_rules(
    success,
    verified,
    feasible,
    expected_status,
    expected_feasible,
):
    backend, task, _, request, spec, metadata, identity = setup_backend()
    reference, _ = backend.ask_or_recover_trial(
        identity, spec, slot_index=0, asked_at=NOW
    )
    experiment = backend.create_experiment_spec(
        task=task,
        metadata=metadata,
        spec=spec,
        request=request,
        reference=reference,
    )
    evaluation = result(experiment, success=success)
    checked = verification(
        experiment,
        verified=verified,
        feasible=feasible,
    )
    outcome = backend.tell_trial(
        spec=spec,
        reference=reference,
        experiment=experiment,
        evaluation=evaluation,
        verification=checked,
        reported_at=NOW,
    )
    assert outcome.status == expected_status
    assert outcome.feasible is expected_feasible
    assert (
        backend.tell_trial(
            spec=spec,
            reference=reference,
            experiment=experiment,
            evaluation=evaluation,
            verification=checked,
            reported_at=NOW,
        )
        == outcome
    )


def test_conflicting_repeated_tell_fails():
    backend, task, _, request, spec, metadata, identity = setup_backend()
    reference, _ = backend.ask_or_recover_trial(
        identity, spec, slot_index=0, asked_at=NOW
    )
    experiment = backend.create_experiment_spec(
        task=task,
        metadata=metadata,
        spec=spec,
        request=request,
        reference=reference,
    )
    backend.tell_trial(
        spec=spec,
        reference=reference,
        experiment=experiment,
        evaluation=result(experiment, 0.5),
        verification=verification(experiment, 0.5),
        reported_at=NOW,
    )
    with pytest.raises(ConflictingTrialReportError):
        backend.tell_trial(
            spec=spec,
            reference=reference,
            experiment=experiment,
            evaluation=result(experiment, 0.6),
            verification=verification(experiment, 0.6),
            reported_at=NOW,
        )


def test_non_finite_score_is_failed_without_penalty():
    backend, task, _, request, spec, metadata, identity = setup_backend()
    reference, _ = backend.ask_or_recover_trial(
        identity, spec, slot_index=0, asked_at=NOW
    )
    experiment = backend.create_experiment_spec(
        task=task,
        metadata=metadata,
        spec=spec,
        request=request,
        reference=reference,
    )
    outcome = backend.tell_trial(
        spec=spec,
        reference=reference,
        experiment=experiment,
        evaluation=result(experiment, float("nan")),
        verification=verification(experiment, float("nan")),
        reported_at=NOW,
    )
    assert outcome.status == OptunaTrialStatus.FAIL
    assert outcome.objective_value is None


def test_study_identity_attributes_are_validated_on_resume():
    backend, _, _, _, spec, _, identity = setup_backend()
    mismatched = replace(
        identity,
        attributes={**identity.attributes, "dataset_version": "different"},
    )
    with pytest.raises(StudyIdentityMismatchError, match="dataset_version"):
        backend.prepare_or_load_study(
            mismatched,
            spec,
            started_at=NOW,
            trial_budget=3,
        )
