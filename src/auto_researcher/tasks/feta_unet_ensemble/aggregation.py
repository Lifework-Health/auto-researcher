"""Deterministic probability aggregation with compatibility checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_unet_ensemble.models import (
    EnsembleMember,
    EnsembleSpecification,
)


def validate_compatible_members(
    members: Sequence[EnsembleMember],
) -> tuple[EnsembleMember, ...]:
    validated = tuple(EnsembleMember.model_validate(item) for item in members)
    if not 2 <= len(validated) <= 4:
        raise ValueError("feta_unet_ensemble_member_count_invalid")
    if len({item.experiment_id for item in validated}) != len(validated):
        raise ValueError("feta_unet_ensemble_member_duplicate")
    if len({item.checkpoint_sha256 for item in validated}) != len(validated):
        raise ValueError("feta_unet_ensemble_checkpoint_duplicate")
    if len({item.compatibility_identity() for item in validated}) != 1:
        raise ValueError("feta_unet_ensemble_member_incompatible")
    return validated


def equal_weight_specification(
    ensemble_id: str,
    members: Sequence[EnsembleMember],
    *,
    selection_rule: str,
) -> EnsembleSpecification:
    validated = validate_compatible_members(members)
    weight = 1.0 / len(validated)
    return EnsembleSpecification(
        ensemble_id=ensemble_id,
        members=validated,
        weights=tuple(weight for _ in validated),
        selection_rule=selection_rule,
    )


def _validated_probability_tensor(value: Any):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional FeTA dependency
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    tensor = np.asarray(value, dtype=np.float32)
    if tensor.ndim != 4 or tensor.shape[0] != 8:
        raise ValueError("feta_unet_ensemble_probability_shape_invalid")
    if not bool(np.isfinite(tensor).all()):
        raise ValueError("feta_unet_ensemble_probability_non_finite")
    if float(tensor.min()) < -1e-6 or float(tensor.max()) > 1.0 + 1e-6:
        raise ValueError("feta_unet_ensemble_probability_range_invalid")
    totals = tensor.sum(axis=0)
    if not bool(np.allclose(totals, 1.0, rtol=0.0, atol=1e-3)):
        raise ValueError("feta_unet_ensemble_probability_sum_invalid")
    return tensor


def aggregate_probabilities(
    probabilities: Sequence[Any],
    weights: Sequence[float],
):
    """Return the deterministic, normalised per-class weighted probability mean."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional FeTA dependency
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    tensors = tuple(_validated_probability_tensor(item) for item in probabilities)
    if len(tensors) < 2 or len(tensors) > 4 or len(weights) != len(tensors):
        raise ValueError("feta_unet_ensemble_probability_member_count_invalid")
    if len({item.shape for item in tensors}) != 1:
        raise ValueError("feta_unet_ensemble_probability_shape_mismatch")
    numeric_weights = tuple(float(item) for item in weights)
    specification_check = sum(numeric_weights)
    if (
        any(not np.isfinite(item) or item < 0.0 for item in numeric_weights)
        or not np.isclose(specification_check, 1.0, rtol=0.0, atol=1e-9)
    ):
        raise ValueError("feta_unet_ensemble_weight_invalid")
    result = np.zeros_like(tensors[0], dtype=np.float32)
    for weight, tensor in zip(numeric_weights, tensors, strict=True):
        result += np.float32(weight) * tensor
    normaliser = result.sum(axis=0, keepdims=True)
    if not bool(np.isfinite(normaliser).all()) or bool((normaliser <= 0).any()):
        raise ValueError("feta_unet_ensemble_probability_normalisation_invalid")
    result /= normaliser
    return result


def predicted_labels(probabilities: Any):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional FeTA dependency
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    tensor = _validated_probability_tensor(probabilities)
    return np.argmax(tensor, axis=0).astype(np.uint8, copy=False)


def member_identity(member: EnsembleMember) -> str:
    return payload_hash(member.model_dump(mode="json"))


__all__ = [
    "aggregate_probabilities",
    "equal_weight_specification",
    "member_identity",
    "predicted_labels",
    "validate_compatible_members",
]
