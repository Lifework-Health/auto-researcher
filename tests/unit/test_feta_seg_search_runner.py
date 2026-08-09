from __future__ import annotations

from pathlib import Path

import pytest

from auto_researcher.tasks.feta_seg.manifests import FeTASubject
from auto_researcher.tasks.feta_seg.splits import Partition, locked_partition
from auto_researcher.tasks.feta_seg_search.runner import select_fold_zero_subjects


def _methods() -> dict[str, str]:
    return {
        f"sub-{index:03d}": "mial" if index <= 40 else "irtk"
        for index in range(1, 81)
    }


def _subjects() -> tuple[FeTASubject, ...]:
    return tuple(
        FeTASubject(
            subject_id=subject_id,
            reconstruction_method=method,
            image_path=Path(f"{subject_id}_image.nii.gz"),
            segmentation_path=Path(f"{subject_id}_label.nii.gz"),
            image_sha256="a" * 64,
            segmentation_sha256="b" * 64,
            shape=(8, 8, 8),
            spacing=(0.5, 0.5, 0.5),
            labels=tuple(range(8)),
        )
        for subject_id, method in _methods().items()
    )


def test_fold_zero_membership_is_exact_54_14_and_development_only():
    partition = locked_partition(_methods())
    training, validation = select_fold_zero_subjects(_subjects(), partition)
    assert len(training) == 54
    assert len(validation) == 14
    assert {item.subject_id for item in training} == {
        subject_id
        for subject_id in partition.development
        if partition.folds[subject_id] != 0
    }
    assert {item.subject_id for item in validation} == {
        subject_id
        for subject_id in partition.development
        if partition.folds[subject_id] == 0
    }
    assert not {
        item.subject_id for item in training + validation
    } & set(partition.holdout)


def test_holdout_subject_access_fails_closed():
    original = locked_partition(_methods())
    leaked = original.holdout[0]
    partition = Partition(
        holdout=original.holdout,
        development=original.development + (leaked,),
        folds={**original.folds, leaked: 0},
    )
    with pytest.raises(ValueError, match="feta_search_holdout_accessed"):
        select_fold_zero_subjects(_subjects(), partition)
