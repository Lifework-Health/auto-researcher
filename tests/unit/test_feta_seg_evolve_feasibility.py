from __future__ import annotations

import math
from pathlib import Path

import pytest

from auto_researcher.contracts.enums import ProvenanceKind
from auto_researcher.cli import _load_yaml
from auto_researcher.contracts.models import EvaluationResult, ResearchContract
from auto_researcher.tasks.feta_seg.manifests import EXPECTED_MANIFEST_HASH
from auto_researcher.tasks.feta_seg.metrics import LABEL_NAMES
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    FOLD_ID,
    SPLIT_ID,
)
from auto_researcher.tasks.feta_seg_evolve.feasibility import (
    EMPTY_PREDICTION_REASON,
    INVALID_METRICS_REASON,
    PER_TISSUE_DICE_REASON,
    SCIENTIFIC_FEASIBILITY_POLICY,
)
from auto_researcher.tasks.feta_seg_evolve.task import (
    FeTASegEvolveTask,
    default_feta_evolve_contract,
)
from auto_researcher.tasks.feta_seg_evolve.verification import (
    FeTASegEvolveVerificationPolicy,
)


def _metrics(*, empty: object = 0, minimum: float = 0.65) -> dict:
    return {
        "mean_subject_macro_dice": 0.7304405171288445,
        "subject_metrics": [],
        "per_tissue_dice": {name: minimum for name in LABEL_NAMES.values()},
        "reconstruction_macro_dice": {"MIAL": 0.78, "IRTK": 0.68},
        "reconstruction_gap": 0.145061,
        "empty_prediction_count": empty,
        "training_policy_identity": "policy",
        "base_configuration_identity": "base",
        "candidate_provenance": {},
        "policy_trace": [],
        "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
        "split_identity": SPLIT_ID,
        "split_hash": EXPECTED_SPLIT_HASH,
        "fold_identity": FOLD_ID,
        "fold_hash": EXPECTED_FOLD_HASH,
        "fold": 0,
        "training_subject_count": 54,
        "validation_subject_count": 14,
        "holdout_subjects_evaluated": 0,
    }


def _evaluation(metrics: dict | None = None) -> EvaluationResult:
    return EvaluationResult(
        experiment_id="feta-feasibility",
        success=True,
        primary_score=0.7304405171288445,
        metrics=metrics or _metrics(),
        constraint_results={"evaluator_integrity": True},
        evaluator_version="unchanged-evaluator",
        provenance=ProvenanceKind.REAL,
    )


def _decision(metrics: dict | None = None):
    return FeTASegEvolveVerificationPolicy().evaluate_constraints(
        _evaluation(metrics), default_feta_evolve_contract()
    )


def test_a49_like_metrics_are_scientifically_feasible():
    decision = _decision()
    assert decision.constraint_compliant is True
    assert decision.reasons == ("feta_evolve_evidence_integrity_verified",)


@pytest.mark.parametrize("empty", [1, 28])
def test_empty_predictions_fail_with_stable_reason(empty):
    decision = _decision(_metrics(empty=empty))
    assert decision.constraint_compliant is False
    assert EMPTY_PREDICTION_REASON in decision.reasons


def test_per_tissue_threshold_is_inclusive_and_collapse_fails():
    passing = _metrics(minimum=0.5)
    assert _decision(passing).constraint_compliant is True

    below = _metrics()
    below["per_tissue_dice"]["brainstem"] = 0.49
    decision = _decision(below)
    assert decision.constraint_compliant is False
    assert PER_TISSUE_DICE_REASON in decision.reasons

    collapsed = _metrics()
    collapsed["per_tissue_dice"].update({"brainstem": 0.0, "cerebellum": 0.0})
    decision = _decision(collapsed)
    assert decision.constraint_compliant is False
    assert PER_TISSUE_DICE_REASON in decision.reasons


@pytest.mark.parametrize(
    "mutate",
    [
        lambda metrics: metrics.pop("per_tissue_dice"),
        lambda metrics: metrics["per_tissue_dice"].pop("brainstem"),
        lambda metrics: metrics["per_tissue_dice"].update({"unexpected": 0.9}),
        lambda metrics: metrics["per_tissue_dice"].update({"brainstem": True}),
        lambda metrics: metrics["per_tissue_dice"].update({"brainstem": "0.9"}),
        lambda metrics: metrics.update({"empty_prediction_count": False}),
        lambda metrics: metrics.update({"empty_prediction_count": 0.0}),
    ],
)
def test_malformed_feasibility_metrics_fail_closed(mutate):
    metrics = _metrics()
    mutate(metrics)
    decision = _decision(metrics)
    assert decision.constraint_compliant is False
    assert INVALID_METRICS_REASON in decision.reasons


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.1, 1.1])
def test_nonfinite_or_out_of_range_tissue_dice_fails_closed(value):
    metrics = _metrics()
    metrics["per_tissue_dice"]["brainstem"] = value
    evaluation = EvaluationResult.model_construct(
        experiment_id="feta-feasibility",
        success=True,
        primary_score=0.73,
        metrics=metrics,
        constraint_results={"evaluator_integrity": True},
        artefact_references=(),
        evaluator_version="unchanged-evaluator",
        provenance=ProvenanceKind.REAL,
        error=None,
    )
    decision = FeTASegEvolveVerificationPolicy().evaluate_constraints(
        evaluation, default_feta_evolve_contract()
    )
    assert decision.constraint_compliant is False
    assert INVALID_METRICS_REASON in decision.reasons


def test_reconstruction_gap_is_diagnostic_not_a_hard_gate():
    metrics = _metrics()
    metrics["reconstruction_gap"] = 0.99
    assert _decision(metrics).constraint_compliant is True


def test_contract_binds_exact_feasibility_identity_and_thresholds():
    contract = default_feta_evolve_contract()
    assert contract.constraints["scientific_feasibility_policy"] == (
        SCIENTIFIC_FEASIBILITY_POLICY
    )
    assert contract.constraints["maximum_empty_predictions"] == 0
    assert contract.constraints["minimum_per_tissue_dice"] == 0.5
    FeTASegEvolveTask().validate_contract(contract)

    example = ResearchContract.model_validate(
        _load_yaml(
            Path(__file__).parents[2] / "examples/tasks/feta_seg_evolve/contract.yaml"
        )
    )
    FeTASegEvolveTask().validate_contract(example)
    assert example.constraints["scientific_feasibility_policy"] == (
        SCIENTIFIC_FEASIBILITY_POLICY
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("scientific_feasibility_policy", None),
        ("scientific_feasibility_policy", "feta-evolve-scientific-feasibility-v0"),
        ("maximum_empty_predictions", None),
        ("maximum_empty_predictions", 1),
        ("maximum_empty_predictions", False),
        ("minimum_per_tissue_dice", None),
        ("minimum_per_tissue_dice", 0.49),
        ("minimum_per_tissue_dice", 0.6),
    ],
)
def test_missing_altered_or_weakened_feasibility_contract_fails(key, value):
    contract = default_feta_evolve_contract()
    constraints = dict(contract.constraints)
    if value is None:
        constraints.pop(key)
    else:
        constraints[key] = value
    changed = contract.model_copy(update={"constraints": constraints})
    with pytest.raises(
        ValueError, match="feta_evolve_contract_feasibility_identity_mismatch"
    ):
        FeTASegEvolveTask().validate_contract(changed)
