from __future__ import annotations

import os

import numpy as np
import pytest
from pydantic import ValidationError

from auto_researcher.tasks.feta_unet_ensemble.aggregation import (
    aggregate_probabilities,
    equal_weight_specification,
    member_identity,
    predicted_labels,
    validate_compatible_members,
)
from auto_researcher.tasks.feta_unet_ensemble.cache import (
    load_probability_cache,
    write_probability_cache,
)
from auto_researcher.tasks.feta_unet_ensemble.models import (
    EnsembleMember,
    EnsembleSpecification,
)


def _member(index: int, **changes) -> EnsembleMember:
    payload = {
        "experiment_id": f"experiment-{index}",
        "checkpoint_sha256": f"{index + 1:064x}",
        "configuration_identity": f"{index + 11:064x}",
        "architecture_identity": f"architecture-{index}",
        "dataset_manifest_hash": "dataset",
        "split_hash": "split",
        "fold_hash": "fold",
        "preprocessing_identity": "preprocessing",
        "label_mapping_identity": "labels-0-through-7",
        "inference_identity": "native-probability-v1",
    }
    payload.update(changes)
    return EnsembleMember(**payload)


def _probabilities(label: int) -> np.ndarray:
    values = np.full((8, 3, 4, 5), 0.02, dtype=np.float32)
    values[label] = 0.86
    return values


def test_equal_weight_specification_is_compatible_and_holdout_safe():
    specification = equal_weight_specification(
        "ensemble-v4-v5",
        (_member(0), _member(1)),
        selection_rule="pre-specified verified champions",
    )
    assert specification.weights == (0.5, 0.5)
    assert specification.sealed_holdout_evaluations == 0
    assert specification.protected_development_only is True


def test_member_compatibility_rejects_split_drift():
    with pytest.raises(ValueError, match="member_incompatible"):
        validate_compatible_members((_member(0), _member(1, split_hash="other")))


def test_specification_rejects_invalid_weights_and_duplicate_checkpoints():
    with pytest.raises(ValidationError, match="weight_sum_invalid"):
        EnsembleSpecification(
            ensemble_id="invalid",
            members=(_member(0), _member(1)),
            weights=(0.8, 0.8),
            selection_rule="test",
        )
    with pytest.raises(ValidationError, match="checkpoint_duplicate"):
        EnsembleSpecification(
            ensemble_id="duplicate",
            members=(
                _member(0),
                _member(1, checkpoint_sha256=_member(0).checkpoint_sha256),
            ),
            weights=(0.5, 0.5),
            selection_rule="test",
        )


def test_probability_mean_is_deterministic_and_shape_safe():
    first = _probabilities(1)
    second = _probabilities(2)
    combined = aggregate_probabilities((first, second), (0.75, 0.25))
    assert combined.dtype == np.float32
    assert np.allclose(combined.sum(axis=0), 1.0)
    assert np.all(predicted_labels(combined) == 1)
    with pytest.raises(ValueError, match="shape_mismatch"):
        aggregate_probabilities(
            (first, np.ones((8, 2, 2, 2), dtype=np.float32) / 8),
            (0.5, 0.5),
        )


def test_protected_probability_cache_is_atomic_and_identity_checked(tmp_path):
    member = _member(0)
    path = tmp_path / "protected" / "sub-001.npy"
    record = write_probability_cache(
        path,
        _probabilities(3),
        subject_id="sub-001",
        member_identity=member_identity(member),
    )
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert np.array_equal(load_probability_cache(path, record), _probabilities(3))
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="cache_identity_mismatch"):
        load_probability_cache(path, record)


def test_probability_cache_refuses_overwrite(tmp_path):
    path = tmp_path / "sub-001.npy"
    member = _member(0)
    write_probability_cache(
        path,
        _probabilities(1),
        subject_id="sub-001",
        member_identity=member_identity(member),
    )
    with pytest.raises(ValueError, match="cache_exists"):
        write_probability_cache(
            path,
            _probabilities(1),
            subject_id="sub-001",
            member_identity=member_identity(member),
        )
