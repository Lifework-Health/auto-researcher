"""Protected, deterministic FeTA U-Net ensemble evaluation sidecar."""

from auto_researcher.tasks.feta_unet_ensemble.aggregation import (
    aggregate_probabilities,
    equal_weight_specification,
    predicted_labels,
    validate_compatible_members,
)
from auto_researcher.tasks.feta_unet_ensemble.models import (
    EnsembleMember,
    EnsembleSpecification,
    ProbabilityCacheRecord,
)
from auto_researcher.tasks.feta_unet_ensemble.evaluation import (
    candidate_subsets,
    evaluate_manifest,
    load_member_source,
)

__all__ = [
    "EnsembleMember",
    "EnsembleSpecification",
    "ProbabilityCacheRecord",
    "aggregate_probabilities",
    "equal_weight_specification",
    "predicted_labels",
    "validate_compatible_members",
    "candidate_subsets",
    "evaluate_manifest",
    "load_member_source",
]
