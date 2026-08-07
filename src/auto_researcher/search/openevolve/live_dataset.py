"""Closed dataset classifications permitted at the live mutation boundary."""

from typing import Final, Literal, TypeAlias

LiveMutationDatasetClass: TypeAlias = Literal["synthetic", "public_benchmark"]

ALLOWED_LIVE_MUTATION_DATASET_CLASSES: Final[frozenset[LiveMutationDatasetClass]] = (
    frozenset({"synthetic", "public_benchmark"})
)

PROHIBITED_LIVE_MUTATION_DATASET_CLASSES: Final[frozenset[str]] = frozenset(
    {"aura", "genuine_icca", "mri", "patient_data"}
)
