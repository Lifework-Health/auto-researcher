"""Deterministic, reconstruction-stratified FeTA development partitions."""

import random
from collections import Counter
from dataclasses import dataclass

from auto_researcher.runtime.identity import payload_hash

SPLIT_ID = "feta-development-holdout-v1"
FOLD_ID = "feta-dev-5fold-v1"
SPLIT_SEED = 20260807


@dataclass(frozen=True)
class Partition:
    holdout: tuple[str, ...]
    development: tuple[str, ...]
    folds: dict[str, int]

    @property
    def split_hash(self) -> str:
        return payload_hash(
            {
                "identity": SPLIT_ID,
                "holdout": self.holdout,
                "development": self.development,
            }
        )

    @property
    def fold_hash(self) -> str:
        return payload_hash({"identity": FOLD_ID, "assignments": self.folds})


def locked_partition(subject_methods: dict[str, str]) -> Partition:
    if len(subject_methods) != 80 or Counter(subject_methods.values()) != {
        "mial": 40,
        "irtk": 40,
    }:
        raise ValueError("feta_inventory_not_80_40_40")
    holdout: list[str] = []
    development: list[str] = []
    folds: dict[str, int] = {}
    for method in ("mial", "irtk"):
        values = sorted(
            key for key, value in subject_methods.items() if value == method
        )
        random.Random(SPLIT_SEED + (0 if method == "mial" else 1)).shuffle(values)
        holdout.extend(values[:6])
        remaining = values[6:]
        development.extend(remaining)
        for index, subject_id in enumerate(remaining):
            folds[subject_id] = index % 5
    result = Partition(
        tuple(sorted(holdout)), tuple(sorted(development)), dict(sorted(folds.items()))
    )
    validate_partition(result, subject_methods)
    return result


def validate_partition(partition: Partition, subject_methods: dict[str, str]) -> None:
    if set(partition.holdout) & set(partition.development):
        raise ValueError("feta_split_overlap")
    if set(partition.holdout) | set(partition.development) != set(subject_methods):
        raise ValueError("feta_split_incomplete")
    if len(partition.holdout) != 12 or len(partition.development) != 68:
        raise ValueError("feta_split_size_mismatch")
    if set(partition.folds) != set(partition.development) or set(
        partition.folds.values()
    ) != set(range(5)):
        raise ValueError("feta_fold_coverage_invalid")
    for method in ("mial", "irtk"):
        if sum(subject_methods[item] == method for item in partition.holdout) != 6:
            raise ValueError("feta_holdout_stratification_invalid")
        counts = Counter(
            partition.folds[item]
            for item in partition.development
            if subject_methods[item] == method
        )
        if max(counts.values()) - min(counts.values()) > 1:
            raise ValueError("feta_fold_stratification_invalid")
