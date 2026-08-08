from pathlib import Path

import pytest

from auto_researcher.tasks.feta_seg.configuration import FeTASegConfiguration
from auto_researcher.tasks.feta_seg.fold_resume import (
    load_fold_result,
    persist_fold_result,
)
from auto_researcher.tasks.feta_seg.manifests import FeTASubject
from auto_researcher.tasks.feta_seg.runner import FoldExecutionResult
from auto_researcher.tasks.feta_seg.trainer import checkpoint_reference


def subject(subject_id: str) -> FeTASubject:
    return FeTASubject(
        subject_id,
        "mial",
        Path(f"{subject_id}_image.nii.gz"),
        Path(f"{subject_id}_label.nii.gz"),
        "a" * 64,
        "b" * 64,
        (256, 256, 256),
        (0.5, 0.5, 0.5),
        tuple(range(8)),
    )


def completed_result(source_root: Path) -> FoldExecutionResult:
    checkpoint = source_root / "checkpoints/fold-0/best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"verified-checkpoint")

    reference = checkpoint_reference(
        checkpoint,
        fold=0,
        best_epoch=300,
        score=0.82,
        output_root=source_root / "checkpoints",
    )

    return FoldExecutionResult(
        fold=0,
        subject_metrics=(
            {"subject_id": "sub-001", "macro_dice": 0.8},
            {"subject_id": "sub-002", "macro_dice": 0.84},
        ),
        best_epoch=300,
        validation_score=0.82,
        training_duration_seconds=100.0,
        total_duration_seconds=120.0,
        peak_gpu_memory_bytes=1024,
        checkpoint=reference,
        seed=20260807,
        source_runner_version="feta-five-fold-oof-runner-v2",
        source_data_loader_version="legacy-loader",
    )


def test_completed_fold_can_be_verified_and_reused(tmp_path):
    configuration = FeTASegConfiguration()
    validation = (subject("sub-001"), subject("sub-002"))

    source_root = tmp_path / "source"
    target_root = tmp_path / "target"

    result = completed_result(source_root)

    persist_fold_result(
        source_root,
        result,
        configuration,
        validation,
    )

    reused = load_fold_result(
        source_root,
        target_root,
        FoldExecutionResult,
        configuration,
        0,
        validation,
    )

    assert reused is not None
    assert reused.reused_fold_result is True
    assert reused.fold == 0
    assert reused.best_epoch == 300
    assert reused.validation_score == pytest.approx(0.82)
    assert len(reused.subject_metrics) == 2

    copied = target_root / "checkpoints/fold-0/best.pt"
    assert copied.read_bytes() == b"verified-checkpoint"
    assert reused.checkpoint["sha256"] == checkpoint_reference(
        copied,
        fold=0,
        best_epoch=300,
        score=0.82,
        output_root=target_root / "checkpoints",
    )["sha256"]


def test_fold_reuse_rejects_wrong_validation_membership(tmp_path):
    configuration = FeTASegConfiguration()
    original_validation = (
        subject("sub-001"),
        subject("sub-002"),
    )

    source_root = tmp_path / "source"
    result = completed_result(source_root)

    persist_fold_result(
        source_root,
        result,
        configuration,
        original_validation,
    )

    wrong_validation = (
        subject("sub-001"),
        subject("sub-003"),
    )

    with pytest.raises(
        ValueError,
        match="feta_fold_resume_identity_mismatch",
    ):
        load_fold_result(
            source_root,
            tmp_path / "target",
            FoldExecutionResult,
            configuration,
            0,
            wrong_validation,
        )
