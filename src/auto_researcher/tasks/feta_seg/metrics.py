"""FeTA subject-level foreground Dice aggregation."""

import math
from collections import defaultdict
from collections.abc import Sequence

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


def dice(actual: Sequence[int], predicted: Sequence[int], label: int) -> float:
    if len(actual) != len(predicted):
        raise ValueError("feta_metric_shape_mismatch")
    a = sum(item == label for item in actual)
    p = sum(item == label for item in predicted)
    if not a:
        raise ValueError("feta_subject_tissue_absent")
    intersection = sum(
        x == label and y == label for x, y in zip(actual, predicted, strict=True)
    )
    return 2.0 * intersection / (a + p)


def aggregate_subject_metrics(subjects: Sequence[dict]) -> dict:
    if not subjects:
        raise ValueError("feta_subject_metrics_empty")
    tissue: dict[int, list[float]] = defaultdict(list)
    macros: list[float] = []
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in subjects:
        scores = row["dice"]
        if set(map(int, scores)) != set(LABELS):
            raise ValueError("feta_tissue_metrics_incomplete")
        values = [
            float(scores[str(label)] if str(label) in scores else scores[label])
            for label in LABELS
        ]
        if not all(math.isfinite(item) and 0 <= item <= 1 for item in values):
            raise ValueError("feta_tissue_metric_invalid")
        macro = sum(values) / len(values)
        macros.append(macro)
        by_method[str(row["reconstruction_method"])].append(macro)
        for label, value in zip(LABELS, values, strict=True):
            tissue[label].append(value)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    method = {key: mean(values) for key, values in by_method.items()}
    return {
        "mean_subject_macro_dice": mean(macros),
        "median_subject_macro_dice": sorted(macros)[len(macros) // 2],
        "subject_macro_dice": macros,
        "per_tissue_dice": {
            LABEL_NAMES[label]: mean(tissue[label]) for label in LABELS
        },
        "reconstruction_macro_dice": method,
        "reconstruction_gap": abs(method.get("mial", 0.0) - method.get("irtk", 0.0)),
    }
