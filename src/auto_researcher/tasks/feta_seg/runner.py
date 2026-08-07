"""CUDA execution for the fixed FeTA SegResNet baseline and engineering smoke."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_researcher.tasks.feta_seg.configuration import FeTASegConfiguration
from auto_researcher.tasks.feta_seg.manifests import (
    EXPECTED_MANIFEST_HASH,
    FeTASubject,
    inspect_subjects,
    manifest_hash,
)
from auto_researcher.tasks.feta_seg.metrics import (
    LABELS,
    aggregate_subject_metrics,
    evaluate_subject_segmentation,
)
from auto_researcher.tasks.feta_seg.model import create_segresnet
from auto_researcher.tasks.feta_seg.splits import Partition, locked_partition
from auto_researcher.tasks.feta_seg.trainer import (
    checkpoint_reference,
    create_loss,
    create_optimizer,
    require_full_baseline_environment,
    seed_everything,
    sliding_window_predict,
)
from auto_researcher.tasks.feta_seg.transforms import create_transforms
from auto_researcher.tasks.models import TaskRuntimeContext

ENGINEERING_SMOKE_VERSION = "feta-real-data-gpu-engineering-smoke-v1"
RUNNER_VERSION = "feta-five-fold-oof-runner-v2"
DATA_LOADER_VERSION = "monai-persistent-train-uncached-validation-workers4-v2"


@dataclass(frozen=True)
class FoldExecutionResult:
    fold: int
    subject_metrics: tuple[dict[str, Any], ...]
    best_epoch: int
    validation_score: float
    training_duration_seconds: float
    total_duration_seconds: float
    peak_gpu_memory_bytes: int
    checkpoint: dict[str, Any]
    seed: int


FoldExecutor = Callable[
    [int, tuple[FeTASubject, ...], tuple[FeTASubject, ...]], FoldExecutionResult
]


def restore_prediction_to_native(
    prediction: Any,
    transformed_affine: Any,
    reference_segmentation: Path,
) -> Any:
    """Nearest-neighbour restore from cropped 0.5-mm RAS to native GT geometry."""

    try:
        import nibabel as nib
        import numpy as np
        from nibabel.processing import resample_from_to
    except ImportError as exc:
        raise RuntimeError("feta_nifti_dependencies_unavailable") from exc
    values = np.asarray(prediction)
    if values.ndim != 3 or not set(np.unique(values)).issubset(set(range(8))):
        raise ValueError("feta_prediction_labels_invalid")
    reference: Any = nib.load(str(reference_segmentation))
    source = nib.Nifti1Image(values.astype(np.int16), np.asarray(transformed_affine))
    restored = resample_from_to(
        source,
        (reference.shape, reference.affine),
        order=0,
        mode="constant",
        cval=0,
    )
    result = np.rint(np.asarray(restored.dataobj)).astype(np.uint8)
    if result.shape != reference.shape or not set(np.unique(result)).issubset(
        set(range(8))
    ):
        raise ValueError("feta_native_prediction_geometry_invalid")
    return result


def _native_label(subject: FeTASubject) -> Any:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_nifti_dependencies_unavailable") from exc
    image: Any = nib.load(str(subject.segmentation_path))
    values = np.rint(np.asarray(image.dataobj)).astype(np.uint8)
    if set(np.unique(values)) != set(range(8)):
        raise ValueError("feta_subject_tissue_absent")
    return values


def _macro_dice(actual: Any, predicted: Any) -> float:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    scores: list[float] = []
    for label in LABELS:
        actual_mask = actual == label
        predicted_mask = predicted == label
        actual_count = int(actual_mask.sum())
        if actual_count == 0:
            raise ValueError("feta_subject_tissue_absent")
        predicted_count = int(predicted_mask.sum())
        intersection = int(np.logical_and(actual_mask, predicted_mask).sum())
        scores.append(
            0.0
            if predicted_count == 0
            else 2.0 * intersection / (actual_count + predicted_count)
        )
    return sum(scores) / len(scores)


def _predict_native(
    model: Any,
    sample: dict[str, Any],
    subject: FeTASubject,
    configuration: FeTASegConfiguration,
) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    image = sample["image"]
    inputs = image.unsqueeze(0).to(device="cuda", non_blocking=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = sliding_window_predict(inputs, model, configuration)
    prediction = torch.argmax(logits, dim=1)[0].to(device="cpu").numpy()
    return restore_prediction_to_native(
        prediction,
        image.affine.detach().cpu().numpy(),
        subject.segmentation_path,
    )


def _dataset_records(subjects: Sequence[FeTASubject]) -> list[dict[str, Any]]:
    return [
        {
            "image": subject.image_path,
            "label": subject.segmentation_path,
            "subject_id": subject.subject_id,
            "reconstruction_method": subject.reconstruction_method,
        }
        for subject in subjects
    ]


def _run_cuda_fold(
    fold: int,
    training_subjects: tuple[FeTASubject, ...],
    validation_subjects: tuple[FeTASubject, ...],
    *,
    configuration: FeTASegConfiguration,
    checkpoint_root: Path,
    cache_root: Path,
) -> FoldExecutionResult:
    try:
        import torch
        from monai.data import DataLoader, Dataset, PersistentDataset
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc

    seed = seed_everything(fold)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = create_segresnet(configuration).to("cuda")
    loss_function = create_loss()
    optimizer = create_optimizer(model, configuration)
    scaler = torch.amp.GradScaler("cuda")

    training_dataset = PersistentDataset(
        _dataset_records(training_subjects),
        transform=create_transforms(training=True),
        cache_dir=cache_root / "training",
    )
    # Validation must retain MONAI MetaTensor spatial metadata because native
    # geometry restoration requires image.affine. MONAI 1.5.x PersistentDataset
    # may reload cached MetaTensors as plain torch.Tensor objects, losing affine
    # metadata. Validation transforms are deterministic, so leave them uncached.
    validation_dataset = Dataset(
        _dataset_records(validation_subjects),
        transform=create_transforms(training=False),
    )
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=configuration.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=generator,
    )

    checkpoint_path = checkpoint_root / f"fold-{fold}" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = 0
    training_started = time.perf_counter()
    for epoch in range(1, configuration.maximum_epochs + 1):
        model.train()
        for batch in training_loader:
            inputs = batch["image"].to(device="cuda", non_blocking=True)
            labels = batch["label"].to(device="cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(inputs)
                loss = loss_function(output, labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError("feta_training_loss_non_finite")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch % configuration.validation_every:
            continue
        model.eval()
        validation_scores: list[float] = []
        for index, subject in enumerate(validation_subjects):
            restored = _predict_native(
                model, validation_dataset[index], subject, configuration
            )
            validation_scores.append(_macro_dice(_native_label(subject), restored))
        score = sum(validation_scores) / len(validation_scores)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "validation_score": score,
                    "seed": seed,
                    "runner_version": RUNNER_VERSION,
                },
                checkpoint_path,
            )

    torch.cuda.synchronize()
    training_duration = time.perf_counter() - training_started
    if best_epoch == 0 or not checkpoint_path.is_file():
        raise RuntimeError("feta_best_checkpoint_missing")
    saved = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    subject_metrics: list[dict[str, Any]] = []
    for index, subject in enumerate(validation_subjects):
        restored = _predict_native(
            model, validation_dataset[index], subject, configuration
        )
        native_label = _native_label(subject)
        metrics = evaluate_subject_segmentation(native_label, restored, subject.spacing)
        subject_metrics.append(
            {
                "subject_id": subject.subject_id,
                "reconstruction_method": subject.reconstruction_method,
                "fold": fold,
                **metrics,
            }
        )
    torch.cuda.synchronize()
    total_duration = time.perf_counter() - training_started
    peak_memory = int(torch.cuda.max_memory_allocated())
    reference = checkpoint_reference(
        checkpoint_path,
        fold=fold,
        best_epoch=best_epoch,
        score=best_score,
        output_root=checkpoint_root,
    )
    del model, optimizer, scaler, training_loader, training_dataset, validation_dataset
    torch.cuda.empty_cache()
    return FoldExecutionResult(
        fold=fold,
        subject_metrics=tuple(subject_metrics),
        best_epoch=best_epoch,
        validation_score=best_score,
        training_duration_seconds=training_duration,
        total_duration_seconds=total_duration,
        peak_gpu_memory_bytes=peak_memory,
        checkpoint=reference,
        seed=seed,
    )


def orchestrate_development_folds(
    configuration: FeTASegConfiguration,
    subjects: Sequence[FeTASubject],
    partition: Partition,
    fold_executor: FoldExecutor,
) -> tuple[FoldExecutionResult, ...]:
    """Execute and validate exact OOF membership without exposing hold-out rows."""

    subject_map = {subject.subject_id: subject for subject in subjects}
    if set(subject_map) != set(partition.development) | set(partition.holdout):
        raise ValueError("feta_split_subject_identity_mismatch")
    results: list[FoldExecutionResult] = []
    observed_oof: set[str] = set()
    for fold in range(configuration.fold_count):
        training = tuple(
            subject_map[subject_id]
            for subject_id in partition.development
            if partition.folds[subject_id] != fold
        )
        validation = tuple(
            subject_map[subject_id]
            for subject_id in partition.development
            if partition.folds[subject_id] == fold
        )
        if any(
            subject.subject_id in partition.holdout for subject in training + validation
        ):
            raise ValueError("feta_holdout_accessed")
        result = fold_executor(fold, training, validation)
        if result.fold != fold:
            raise ValueError("feta_fold_result_identity_mismatch")
        result_subjects = {str(row["subject_id"]) for row in result.subject_metrics}
        expected_subjects = {subject.subject_id for subject in validation}
        if result_subjects != expected_subjects or observed_oof & result_subjects:
            raise ValueError("feta_oof_membership_invalid")
        observed_oof.update(result_subjects)
        results.append(result)
    if observed_oof != set(partition.development):
        raise ValueError("feta_oof_coverage_invalid")
    return tuple(results)


def run_full_baseline(
    context: TaskRuntimeContext,
    configuration: FeTASegConfiguration,
    experiment_id: str,
) -> dict[str, Any]:
    if configuration.mode != "full":
        raise ValueError("feta_full_runner_requires_full_configuration")
    if context.data_dir is None or context.workspace_dir is None:
        raise RuntimeError("feta_full_runner_paths_missing")
    environment = require_full_baseline_environment()
    subjects = inspect_subjects(context.data_dir, inspect_labels=False)
    identity = manifest_hash(subjects)
    if identity != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_dataset_identity_mismatch")
    partition = locked_partition(
        {subject.subject_id: subject.reconstruction_method for subject in subjects}
    )
    root = context.workspace_dir / "feta_seg" / experiment_id
    checkpoint_root = root / "checkpoints"
    cache_root = root / "cache"

    def execute(
        fold: int,
        training: tuple[FeTASubject, ...],
        validation: tuple[FeTASubject, ...],
    ) -> FoldExecutionResult:
        return _run_cuda_fold(
            fold,
            training,
            validation,
            configuration=configuration,
            checkpoint_root=checkpoint_root,
            cache_root=cache_root,
        )

    fold_results = orchestrate_development_folds(
        configuration, subjects, partition, execute
    )
    aggregate = aggregate_subject_metrics(
        [row for result in fold_results for row in result.subject_metrics]
    )
    return {
        **aggregate,
        "fold_summaries": [
            {
                "fold": result.fold,
                "seed": result.seed,
                "best_epoch": result.best_epoch,
                "validation_score": result.validation_score,
                "training_duration_seconds": result.training_duration_seconds,
                "total_duration_seconds": result.total_duration_seconds,
                "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
                "validation_subject_count": len(result.subject_metrics),
            }
            for result in fold_results
        ],
        "checkpoint_references": [result.checkpoint for result in fold_results],
        "environment": environment,
        "runner_version": RUNNER_VERSION,
        "data_loader_version": DATA_LOADER_VERSION,
        "folds_completed": len(fold_results),
        "oof_subject_count": sum(
            len(result.subject_metrics) for result in fold_results
        ),
        "holdout_subjects_evaluated": 0,
        "failed_training_folds": 0,
        "valid_prediction_labels": list(range(8)),
    }


def run_engineering_smoke(data_dir: Path) -> dict[str, Any]:
    """One real-data CUDA optimisation step; never valid as baseline evidence."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    environment = require_full_baseline_environment()
    configuration = FeTASegConfiguration()
    subjects = inspect_subjects(data_dir, inspect_labels=False)
    if manifest_hash(subjects) != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_dataset_identity_mismatch")
    partition = locked_partition(
        {subject.subject_id: subject.reconstruction_method for subject in subjects}
    )
    subject_map = {subject.subject_id: subject for subject in subjects}
    validation_subject = subject_map[
        next(item for item in partition.development if partition.folds[item] == 0)
    ]
    training_subject = subject_map[
        next(item for item in partition.development if partition.folds[item] != 0)
    ]
    seed_everything(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    training_result = create_transforms(training=True)(
        _dataset_records((training_subject,))[0]
    )
    if not isinstance(training_result, list) or not training_result:
        raise RuntimeError("feta_patch_sampling_failed")
    patch = training_result[0]
    validation = create_transforms(training=False)(
        _dataset_records((validation_subject,))[0]
    )
    model = create_segresnet(configuration).to("cuda")
    loss_function = create_loss()
    optimizer = create_optimizer(model, configuration)
    scaler = torch.amp.GradScaler("cuda")
    inputs = patch["image"].unsqueeze(0).to("cuda")
    labels = patch["label"].unsqueeze(0).to("cuda")
    step_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(inputs)
        loss_before = loss_function(output, labels)
    scaler.scale(loss_before).backward()
    scaler.step(optimizer)
    scaler.update()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        loss_after = loss_function(model(inputs), labels)
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - step_started

    validation_started = time.perf_counter()
    model.eval()
    restored = _predict_native(model, validation, validation_subject, configuration)
    metrics = evaluate_subject_segmentation(
        _native_label(validation_subject), restored, validation_subject.spacing
    )
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - validation_started
    wall_seconds = time.perf_counter() - started
    estimated_fold_seconds = (
        step_seconds * 54 * configuration.maximum_epochs
        + validation_seconds
        * 14
        * (configuration.maximum_epochs // configuration.validation_every + 1)
    )
    valid_labels = sorted(set(int(item) for item in torch.unique(labels).tolist()))
    return {
        "scientific_baseline": False,
        "reusable_as_baseline_evidence": False,
        "engineering_smoke_version": ENGINEERING_SMOKE_VERSION,
        "dataset_identity_exact": True,
        "holdout_subjects_evaluated": 0,
        "gpu_environment": environment,
        "training_subject_reconstruction": training_subject.reconstruction_method,
        "validation_subject_reconstruction": validation_subject.reconstruction_method,
        "training_patch_shape": list(inputs.shape),
        "validation_transformed_shape": list(validation["image"].shape),
        "native_prediction_shape": list(restored.shape),
        "valid_training_labels": valid_labels,
        "all_labels_valid": set(valid_labels).issubset(set(range(8))),
        "loss_before": float(loss_before.detach().cpu()),
        "loss_after_one_step": float(loss_after.detach().cpu()),
        "metric_panel_complete": all(
            key in metrics
            for key in (
                "macro_dice",
                "macro_hd95_mm",
                "macro_volume_similarity",
                "macro_euler_distance",
            )
        ),
        "validation_metrics": {
            key: metrics[key]
            for key in (
                "macro_dice",
                "macro_hd95_mm",
                "macro_volume_similarity",
                "macro_euler_distance",
                "empty_prediction_count",
            )
        },
        "step_wall_seconds": step_seconds,
        "validation_wall_seconds": validation_seconds,
        "wall_seconds": wall_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "estimated_one_fold_seconds": estimated_fold_seconds,
        "estimated_five_fold_seconds": estimated_fold_seconds * 5,
        "warnings": [
            "NON-SCIENTIFIC engineering timing extrapolation; do not use as baseline evidence."
        ],
    }
