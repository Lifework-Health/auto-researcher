"""Fidelity-dependent endpoint metric policy for FeTA search."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

from auto_researcher.tasks.feta_seg.metrics import LABELS, LABEL_NAMES

MetricTier = Literal["screen", "full"]
METRIC_TIER_POLICY_VERSION = "feta-search-fidelity-metric-tier-v1"
SCREEN_METRIC_VERSION = "feta-dice-reconstruction-screen-panel-v1"
SCREEN_FIDELITIES = (25, 50)
FULL_FIDELITIES = (100, 150, 300, 350)
FULL_PANEL_METRIC_NAMES = frozenset(
    {
        "mean_subject_macro_hd95_mm",
        "mean_subject_macro_volume_similarity",
        "mean_subject_macro_euler_distance",
        "per_class_summary",
    }
)


def metric_tier_for_fidelity(maximum_epochs: int) -> MetricTier:
    if maximum_epochs in SCREEN_FIDELITIES:
        return "screen"
    if maximum_epochs in FULL_FIDELITIES:
        return "full"
    raise ValueError("feta_search_fidelity_invalid")


def evaluate_screen_subject(actual: Any, predicted: Any) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    actual_values = np.asarray(actual)
    predicted_values = np.asarray(predicted)
    if actual_values.shape != predicted_values.shape or actual_values.ndim != 3:
        raise ValueError("feta_metric_shape_mismatch")
    per_class: dict[str, Any] = {}
    empty_count = 0
    scores: list[float] = []
    for label in LABELS:
        actual_mask = actual_values == label
        predicted_mask = predicted_values == label
        actual_count = int(actual_mask.sum())
        if actual_count == 0:
            raise ValueError("feta_subject_tissue_absent")
        predicted_count = int(predicted_mask.sum())
        empty = predicted_count == 0
        intersection = int(np.logical_and(actual_mask, predicted_mask).sum())
        score = 0.0 if empty else 2.0 * intersection / (actual_count + predicted_count)
        scores.append(score)
        empty_count += int(empty)
        per_class[str(label)] = {
            "label_name": LABEL_NAMES[label],
            "dice": score,
            "empty_prediction": empty,
        }
    return {
        "per_class": per_class,
        "macro_dice": sum(scores) / len(scores),
        "empty_prediction_count": empty_count,
    }


def aggregate_screen_subject_metrics(
    subjects: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not subjects:
        raise ValueError("feta_subject_metrics_empty")
    per_label: dict[int, list[float]] = defaultdict(list)
    by_method: dict[str, list[float]] = defaultdict(list)
    subject_rows: list[dict[str, Any]] = []
    total_empty = 0
    for row in subjects:
        per_class = row["per_class"]
        if set(map(int, per_class)) != set(LABELS):
            raise ValueError("feta_tissue_metrics_incomplete")
        macro_dice = float(row["macro_dice"])
        method = str(row["reconstruction_method"])
        empty_count = int(row["empty_prediction_count"])
        by_method[method].append(macro_dice)
        total_empty += empty_count
        for label in LABELS:
            per_label[label].append(float(per_class[str(label)]["dice"]))
        subject_rows.append(
            {
                "subject_id": str(row["subject_id"]),
                "reconstruction_method": method,
                "fold": int(row["fold"]),
                "macro_dice": macro_dice,
                "empty_prediction_count": empty_count,
                "per_class": per_class,
            }
        )

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    macros = [float(row["macro_dice"]) for row in subject_rows]
    method_scores = {name: mean(values) for name, values in sorted(by_method.items())}
    return {
        "mean_subject_macro_dice": mean(macros),
        "median_subject_macro_dice": sorted(macros)[len(macros) // 2],
        "subject_metrics": subject_rows,
        "per_tissue_dice": {
            LABEL_NAMES[label]: mean(per_label[label]) for label in LABELS
        },
        "reconstruction_macro_dice": method_scores,
        "reconstruction_gap": abs(
            method_scores.get("mial", 0.0) - method_scores.get("irtk", 0.0)
        ),
        "empty_prediction_count": total_empty,
    }
