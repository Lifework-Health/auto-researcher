"""Versioned FeTA development metrics with finite empty-prediction handling."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

LABELS = tuple(range(1, 8))
LABEL_NAMES = {
    1: "external_cerebrospinal_fluid",
    2: "grey_matter",
    3: "white_matter",
    4: "ventricles",
    5: "cerebellum",
    6: "deep_grey_matter",
    7: "brainstem",
}

METRIC_PANEL_VERSION = "feta-dice-hd95-vs-euler-panel-v2"
HD95_VERSION = "monai-symmetric-percentile95-physical-mm-v1"
EMPTY_PREDICTION_VERSION = "fov-diagonal-hd95-penalty-v1"
TOPOLOGY_VERSION = "cubical-euler-fg26-bg6-betti-v1"
EXPECTED_BETTI = {label: (2 if label == 2 else 1, 0, 0) for label in LABELS}


def dice(actual: Sequence[int], predicted: Sequence[int], label: int) -> float:
    """Foreground-label Dice for small sequence fixtures."""

    if len(actual) != len(predicted):
        raise ValueError("feta_metric_shape_mismatch")
    actual_count = sum(item == label for item in actual)
    predicted_count = sum(item == label for item in predicted)
    if not actual_count:
        raise ValueError("feta_subject_tissue_absent")
    intersection = sum(
        left == label and right == label
        for left, right in zip(actual, predicted, strict=True)
    )
    return 2.0 * intersection / (actual_count + predicted_count)


def volume_similarity_counts(actual_count: int, predicted_count: int) -> float:
    if actual_count <= 0:
        raise ValueError("feta_subject_tissue_absent")
    if predicted_count <= 0:
        return 0.0
    return 1.0 - abs(predicted_count - actual_count) / (predicted_count + actual_count)


def physical_fov_diagonal(shape: Sequence[int], spacing_mm: Sequence[float]) -> float:
    if len(shape) != 3 or len(spacing_mm) != 3:
        raise ValueError("feta_metric_geometry_invalid")
    return math.sqrt(
        sum(
            ((int(size) - 1) * float(spacing)) ** 2
            for size, spacing in zip(shape, spacing_mm, strict=True)
        )
    )


def physical_hd95(
    actual_mask: Any,
    predicted_mask: Any,
    spacing_mm: Sequence[float],
) -> tuple[float, bool]:
    """Symmetric MONAI HD95 in mm with a deterministic empty-mask penalty."""

    try:
        import numpy as np
        import torch
        from monai.metrics import compute_hausdorff_distance
    except ImportError as exc:
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    actual = np.asarray(actual_mask, dtype=bool)
    predicted = np.asarray(predicted_mask, dtype=bool)
    if actual.shape != predicted.shape or actual.ndim != 3:
        raise ValueError("feta_metric_shape_mismatch")
    if not bool(actual.any()):
        raise ValueError("feta_subject_tissue_absent")
    if not bool(predicted.any()):
        return physical_fov_diagonal(actual.shape, spacing_mm), True
    value = compute_hausdorff_distance(
        torch.as_tensor(predicted[None, None]),
        torch.as_tensor(actual[None, None]),
        include_background=True,
        distance_metric="euclidean",
        percentile=95.0,
        directed=False,
        spacing=tuple(float(item) for item in spacing_mm),
    )
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("feta_hd95_non_finite")
    return result, False


def cubical_euler_characteristic(mask: Any) -> int:
    """Euler characteristic of the union of closed unit cubes in a 3-D mask."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError("feta_topology_requires_3d")
    cubes = int(values.sum())
    if cubes == 0:
        return 0

    adjacency = sum(
        int(
            np.logical_and(
                values[
                    tuple(
                        slice(1, None) if index == axis else slice(None)
                        for index in range(3)
                    )
                ],
                values[
                    tuple(
                        slice(None, -1) if index == axis else slice(None)
                        for index in range(3)
                    )
                ],
            ).sum()
        )
        for axis in range(3)
    )
    faces = 6 * cubes - adjacency

    # Unique grid edges are unions of the four incident voxels. Allocate one
    # orientation at a time to keep peak memory bounded on native FeTA volumes.
    edge_counts: list[int] = []
    for axis in range(3):
        pad_width = [(0, 0), (0, 0), (0, 0)]
        other_axes = [item for item in range(3) if item != axis]
        for other in other_axes:
            pad_width[other] = (1, 1)
        padded = np.pad(values, pad_width, mode="constant")
        edge_shape = list(values.shape)
        edge_shape[other_axes[0]] += 1
        edge_shape[other_axes[1]] += 1
        occupied = np.zeros(edge_shape, dtype=bool)
        for first in (0, 1):
            for second in (0, 1):
                slices = [slice(None), slice(None), slice(None)]
                slices[other_axes[0]] = slice(first, first + edge_shape[other_axes[0]])
                slices[other_axes[1]] = slice(
                    second, second + edge_shape[other_axes[1]]
                )
                occupied |= padded[tuple(slices)]
        edge_counts.append(int(occupied.sum()))
    edges = sum(edge_counts)

    padded = np.pad(values, 1, mode="constant")
    vertex_shape = tuple(size + 1 for size in values.shape)
    vertices_present = np.zeros(vertex_shape, dtype=bool)
    for x_offset in (0, 1):
        for y_offset in (0, 1):
            for z_offset in (0, 1):
                vertices_present |= padded[
                    x_offset : x_offset + vertex_shape[0],
                    y_offset : y_offset + vertex_shape[1],
                    z_offset : z_offset + vertex_shape[2],
                ]
    vertices = int(vertices_present.sum())
    return vertices - edges + faces - cubes


def _component_count(mask: Any, *, connectivity: int) -> tuple[int, Any]:
    try:
        import numpy as np
        from scipy import ndimage  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("feta_topology_dependencies_unavailable") from exc
    structure = ndimage.generate_binary_structure(3, connectivity)
    labelled, count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    return int(count), labelled


def cubical_betti_numbers(mask: Any) -> tuple[int, int, int]:
    """Return Betti (b0, b1, b2) for the versioned fg26/bg6 convention."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3:
        raise ValueError("feta_topology_requires_3d")
    beta0, _ = _component_count(values, connectivity=3)
    padded_background = np.pad(~values, 1, mode="constant", constant_values=True)
    background_count, background_labels = _component_count(
        padded_background, connectivity=1
    )
    exterior_label = int(background_labels[(0, 0, 0)])
    beta2 = background_count - (1 if exterior_label else 0)
    euler = cubical_euler_characteristic(values)
    beta1 = beta0 + beta2 - euler
    if min(beta0, beta1, beta2) < 0:
        raise ValueError("feta_topology_inconsistent")
    return beta0, beta1, beta2


def topology_metrics(mask: Any, label: int) -> dict[str, Any]:
    if label not in EXPECTED_BETTI:
        raise ValueError("feta_topology_label_invalid")
    actual = cubical_betti_numbers(mask)
    expected = EXPECTED_BETTI[label]
    euler = actual[0] - actual[1] + actual[2]
    expected_euler = expected[0] - expected[1] + expected[2]
    return {
        "betti": list(actual),
        "expected_betti": list(expected),
        "euler_characteristic": euler,
        "expected_euler_characteristic": expected_euler,
        "euler_distance": abs(euler - expected_euler),
    }


def evaluate_subject_segmentation(
    actual: Any,
    predicted: Any,
    spacing_mm: Sequence[float],
) -> dict[str, Any]:
    """Calculate the complete metric panel for one native-geometry subject."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    reference = np.asarray(actual)
    estimate = np.asarray(predicted)
    if reference.shape != estimate.shape or reference.ndim != 3:
        raise ValueError("feta_metric_shape_mismatch")
    if not set(np.unique(estimate)).issubset(set(range(8))):
        raise ValueError("feta_prediction_labels_invalid")
    if set(np.unique(reference)) != set(range(8)):
        raise ValueError("feta_subject_tissue_absent")

    per_class: dict[str, dict[str, Any]] = {}
    empty_labels: list[int] = []
    for label in LABELS:
        actual_mask = reference == label
        predicted_mask = estimate == label
        actual_count = int(actual_mask.sum())
        predicted_count = int(predicted_mask.sum())
        intersection = int(np.logical_and(actual_mask, predicted_mask).sum())
        dice_value = (
            0.0
            if predicted_count == 0
            else 2.0 * intersection / (actual_count + predicted_count)
        )
        hd95_value, empty = physical_hd95(actual_mask, predicted_mask, spacing_mm)
        if empty:
            empty_labels.append(label)
        topology = topology_metrics(predicted_mask, label)
        per_class[str(label)] = {
            "label_name": LABEL_NAMES[label],
            "dice": dice_value,
            "hd95_mm": hd95_value,
            "volume_similarity": volume_similarity_counts(
                actual_count, predicted_count
            ),
            **topology,
            "empty_prediction": empty,
        }

    def mean(field: str) -> float:
        return sum(float(per_class[str(label)][field]) for label in LABELS) / len(
            LABELS
        )

    return {
        "per_class": per_class,
        "macro_dice": mean("dice"),
        "macro_hd95_mm": mean("hd95_mm"),
        "macro_volume_similarity": mean("volume_similarity"),
        "macro_euler_distance": mean("euler_distance"),
        "empty_prediction_labels": empty_labels,
        "empty_prediction_count": len(empty_labels),
    }


def aggregate_subject_metrics(subjects: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not subjects:
        raise ValueError("feta_subject_metrics_empty")
    per_label: dict[int, dict[str, list[float]]] = {
        label: defaultdict(list) for label in LABELS
    }
    subject_rows: list[dict[str, Any]] = []
    by_method: dict[str, list[float]] = defaultdict(list)
    empty_prediction_count = 0
    for row in subjects:
        if "per_class" in row:
            class_values = row["per_class"]
            if set(map(int, class_values)) != set(LABELS):
                raise ValueError("feta_tissue_metrics_incomplete")
            dice_values = {
                str(label): float(class_values[str(label)]["dice"]) for label in LABELS
            }
            macro_dice = float(row.get("macro_dice", sum(dice_values.values()) / 7))
            macro_hd95 = float(row["macro_hd95_mm"])
            macro_volume = float(row["macro_volume_similarity"])
            macro_euler = float(row["macro_euler_distance"])
            empty_count = int(row.get("empty_prediction_count", 0))
            for label in LABELS:
                for field in ("dice", "hd95_mm", "volume_similarity", "euler_distance"):
                    per_label[label][field].append(
                        float(class_values[str(label)][field])
                    )
                per_label[label]["empty_prediction"].append(
                    1.0 if class_values[str(label)]["empty_prediction"] else 0.0
                )
        else:
            dice_values = row["dice"]
            if set(map(int, dice_values)) != set(LABELS):
                raise ValueError("feta_tissue_metrics_incomplete")
            values = [
                float(
                    dice_values[str(label)]
                    if str(label) in dice_values
                    else dice_values[label]
                )
                for label in LABELS
            ]
            macro_dice = sum(values) / len(values)
            macro_hd95 = macro_volume = macro_euler = 0.0
            empty_count = 0
            for label, value in zip(LABELS, values, strict=True):
                per_label[label]["dice"].append(value)

        numeric = (macro_dice, macro_hd95, macro_volume, macro_euler)
        if not all(math.isfinite(item) for item in numeric):
            raise ValueError("feta_tissue_metric_invalid")
        if not 0 <= macro_dice <= 1:
            raise ValueError("feta_tissue_metric_invalid")
        method = str(row["reconstruction_method"])
        by_method[method].append(macro_dice)
        empty_prediction_count += empty_count
        subject_rows.append(
            {
                "subject_id": str(
                    row.get("subject_id", f"subject-{len(subject_rows)}")
                ),
                "reconstruction_method": method,
                "fold": int(row.get("fold", -1)),
                "macro_dice": macro_dice,
                "macro_hd95_mm": macro_hd95,
                "macro_volume_similarity": macro_volume,
                "macro_euler_distance": macro_euler,
                "empty_prediction_count": empty_count,
                "per_class": row.get(
                    "per_class",
                    {str(label): {"dice": dice_values[str(label)]} for label in LABELS},
                ),
            }
        )

    def average(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    method_scores = {key: average(values) for key, values in sorted(by_method.items())}
    macros = [float(row["macro_dice"]) for row in subject_rows]
    complete_panel = all(
        "per_class" in row and "macro_hd95_mm" in row for row in subjects
    )
    result: dict[str, Any] = {
        "mean_subject_macro_dice": average(macros),
        "median_subject_macro_dice": sorted(macros)[len(macros) // 2],
        "subject_metrics": subject_rows,
        "per_tissue_dice": {
            LABEL_NAMES[label]: average(per_label[label]["dice"]) for label in LABELS
        },
        "reconstruction_macro_dice": method_scores,
        "reconstruction_gap": abs(
            method_scores.get("mial", 0.0) - method_scores.get("irtk", 0.0)
        ),
        "empty_prediction_count": empty_prediction_count,
    }
    if complete_panel:
        result.update(
            {
                "mean_subject_macro_hd95_mm": average(
                    [float(row["macro_hd95_mm"]) for row in subject_rows]
                ),
                "mean_subject_macro_volume_similarity": average(
                    [float(row["macro_volume_similarity"]) for row in subject_rows]
                ),
                "mean_subject_macro_euler_distance": average(
                    [float(row["macro_euler_distance"]) for row in subject_rows]
                ),
                "per_class_summary": {
                    LABEL_NAMES[label]: {
                        "dice": average(per_label[label]["dice"]),
                        "hd95_mm": average(per_label[label]["hd95_mm"]),
                        "volume_similarity": average(
                            per_label[label]["volume_similarity"]
                        ),
                        "euler_distance": average(per_label[label]["euler_distance"]),
                        "empty_prediction_count": int(
                            sum(per_label[label]["empty_prediction"])
                        ),
                    }
                    for label in LABELS
                },
            }
        )
    return result
