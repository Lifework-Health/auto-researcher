"""Public-safe metric and learning-curve comparisons for a protected FeTA panel."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from auto_researcher.diagnostics.models import (
    DiagnosticExperiment,
    DiagnosticObservation,
    DiagnosticResult,
)
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_seg.metrics import LABELS, LABEL_NAMES
from auto_researcher.tasks.feta_unet_diagnostics.panel import FeTADiagnosticPanel

COMPARISON_METHOD = "feta-panel-error-comparison-v1"
LEARNING_CURVE_METHOD = "feta-learning-curve-summary-v1"


def _row_index(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        subject_id = row.get("subject_id")
        if not isinstance(subject_id, str) or not subject_id or subject_id in result:
            raise ValueError("feta_diagnostic_subject_metrics_invalid")
        per_class = row.get("per_class")
        if not isinstance(per_class, dict) or set(map(int, per_class)) != set(LABELS):
            raise ValueError("feta_diagnostic_tissue_metrics_incomplete")
        macro = row.get("macro_dice")
        if not isinstance(macro, (int, float)) or not math.isfinite(float(macro)):
            raise ValueError("feta_diagnostic_subject_metrics_invalid")
        result[subject_id] = row
    return result


def _class_dice(row: dict[str, Any], label: int) -> float:
    per_class = row["per_class"]
    item = per_class.get(str(label), per_class.get(label))
    value = item.get("dice") if isinstance(item, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("feta_diagnostic_tissue_metrics_invalid")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("feta_diagnostic_tissue_metrics_invalid")
    return result


def summarise_learning_curve(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        raise ValueError("feta_diagnostic_learning_curve_empty")
    points: list[tuple[int, float]] = []
    for entry in entries:
        epoch = entry.get("epoch")
        score = entry.get("validation_score")
        if (
            not isinstance(epoch, int)
            or epoch < 1
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("feta_diagnostic_learning_curve_invalid")
        points.append((epoch, float(score)))
    if points != sorted(points) or len({epoch for epoch, _ in points}) != len(points):
        raise ValueError("feta_diagnostic_learning_curve_invalid")
    best_epoch, best_score = max(points, key=lambda item: (item[1], -item[0]))
    start_epoch, start_score = points[0]
    end_epoch, end_score = points[-1]
    denominator = end_epoch - start_epoch
    return {
        "first_epoch": start_epoch,
        "first_score": start_score,
        "last_epoch": end_epoch,
        "last_score": end_score,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "score_gain": end_score - start_score,
        "score_gain_per_epoch": (
            0.0 if denominator == 0 else (end_score - start_score) / denominator
        ),
    }


def compare_panel_metrics(
    experiment: DiagnosticExperiment,
    panel: FeTADiagnosticPanel,
    *,
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Mapping[str, Sequence[dict[str, Any]]],
    learning_curves: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    material_delta: float = 0.01,
) -> DiagnosticResult:
    """Compare candidates on identical protected cases without publishing case IDs."""

    if experiment.panel.panel_identity != panel.panel_identity:
        raise ValueError("feta_diagnostic_panel_identity_mismatch")
    if not 0.0 < material_delta < 1.0:
        raise ValueError("feta_diagnostic_material_delta_invalid")
    expected_candidates = {item.experiment_id for item in experiment.candidates}
    if set(candidate_rows) != expected_candidates:
        raise ValueError("feta_diagnostic_candidate_set_mismatch")

    panel_ids = tuple(case.subject_id for case in panel.cases)
    baseline = _row_index(baseline_rows)
    candidates = {
        experiment_id: _row_index(rows)
        for experiment_id, rows in candidate_rows.items()
    }
    if not set(panel_ids).issubset(baseline):
        raise ValueError("feta_diagnostic_panel_metrics_missing")
    if any(not set(panel_ids).issubset(rows) for rows in candidates.values()):
        raise ValueError("feta_diagnostic_panel_metrics_missing")

    observations: list[DiagnosticObservation] = []
    for experiment_id in sorted(candidates):
        rows = candidates[experiment_id]
        per_label: dict[str, Any] = {}
        improved_pairs = 0
        regressed_pairs = 0
        displaced_cases = 0
        for label in LABELS:
            deltas = [
                _class_dice(rows[subject_id], label)
                - _class_dice(baseline[subject_id], label)
                for subject_id in panel_ids
            ]
            improved = sum(delta >= material_delta for delta in deltas)
            regressed = sum(delta <= -material_delta for delta in deltas)
            improved_pairs += improved
            regressed_pairs += regressed
            per_label[str(label)] = {
                "label_name": LABEL_NAMES[label],
                "mean_dice_delta": sum(deltas) / len(deltas),
                "material_improvement_count": improved,
                "material_regression_count": regressed,
            }
        for subject_id in panel_ids:
            deltas = [
                _class_dice(rows[subject_id], label)
                - _class_dice(baseline[subject_id], label)
                for label in LABELS
            ]
            if any(delta >= material_delta for delta in deltas) and any(
                delta <= -material_delta for delta in deltas
            ):
                displaced_cases += 1
        subgroup_deltas: dict[str, float] = {}
        for subgroup in ("mial", "irtk"):
            subgroup_ids = [
                case.subject_id
                for case in panel.cases
                if case.reconstruction_method == subgroup
            ]
            subgroup_deltas[subgroup.upper()] = sum(
                float(rows[subject_id]["macro_dice"])
                - float(baseline[subject_id]["macro_dice"])
                for subject_id in subgroup_ids
            ) / len(subgroup_ids)
        metrics = {
            "panel_identity": panel.panel_identity,
            "case_count": len(panel_ids),
            "material_delta": material_delta,
            "mean_macro_dice_delta": sum(
                float(rows[subject_id]["macro_dice"])
                - float(baseline[subject_id]["macro_dice"])
                for subject_id in panel_ids
            )
            / len(panel_ids),
            "subgroup_macro_dice_delta": subgroup_deltas,
            "per_class": per_label,
            "material_improvement_pairs": improved_pairs,
            "material_regression_pairs": regressed_pairs,
            "error_displacement_case_count": displaced_cases,
            "contains_case_identifiers": False,
        }
        observations.append(
            DiagnosticObservation(
                observation_id=payload_hash(
                    {
                        "diagnostic_id": experiment.diagnostic_id,
                        "method": COMPARISON_METHOD,
                        "experiment_id": experiment_id,
                        "metrics": metrics,
                    }
                ),
                diagnostic_id=experiment.diagnostic_id,
                method=COMPARISON_METHOD,
                model_experiment_ids=(experiment.baseline.experiment_id, experiment_id),
                metrics=metrics,
            )
        )

    for index, left_id in enumerate(sorted(candidates)):
        for right_id in sorted(candidates)[index + 1 :]:
            left_wins = right_wins = ties = 0
            for subject_id in panel_ids:
                for label in LABELS:
                    delta = _class_dice(
                        candidates[left_id][subject_id], label
                    ) - _class_dice(candidates[right_id][subject_id], label)
                    if abs(delta) < material_delta:
                        ties += 1
                    elif delta > 0:
                        left_wins += 1
                    else:
                        right_wins += 1
            metrics = {
                "panel_identity": panel.panel_identity,
                "material_delta": material_delta,
                "left_material_win_count": left_wins,
                "right_material_win_count": right_wins,
                "near_tie_count": ties,
                "complementary_advantage_observed": left_wins > 0 and right_wins > 0,
                "contains_case_identifiers": False,
            }
            observations.append(
                DiagnosticObservation(
                    observation_id=payload_hash(
                        {
                            "diagnostic_id": experiment.diagnostic_id,
                            "method": "feta-panel-complementarity-v1",
                            "models": [left_id, right_id],
                            "metrics": metrics,
                        }
                    ),
                    diagnostic_id=experiment.diagnostic_id,
                    method="feta-panel-complementarity-v1",
                    model_experiment_ids=(left_id, right_id),
                    metrics=metrics,
                )
            )

    if learning_curves is not None:
        expected_models = {experiment.baseline.experiment_id, *expected_candidates}
        if set(learning_curves) != expected_models:
            raise ValueError("feta_diagnostic_learning_curve_set_mismatch")
        for experiment_id in sorted(learning_curves):
            metrics = summarise_learning_curve(learning_curves[experiment_id])
            observations.append(
                DiagnosticObservation(
                    observation_id=payload_hash(
                        {
                            "diagnostic_id": experiment.diagnostic_id,
                            "method": LEARNING_CURVE_METHOD,
                            "experiment_id": experiment_id,
                            "metrics": metrics,
                        }
                    ),
                    diagnostic_id=experiment.diagnostic_id,
                    method=LEARNING_CURVE_METHOD,
                    model_experiment_ids=(experiment_id,),
                    metrics=metrics,
                )
            )

    return DiagnosticResult(
        diagnostic_id=experiment.diagnostic_id,
        success=True,
        observations=tuple(observations),
        aggregate={
            "panel_identity": panel.panel_identity,
            "case_count": len(panel_ids),
            "candidate_count": len(candidates),
            "observation_count": len(observations),
            "contains_case_identifiers": False,
            "interpretation_included": False,
        },
    )
