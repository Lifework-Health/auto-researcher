"""Single-fold CUDA runner for bounded FeTA SegResNet candidates."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from auto_researcher.runtime.identity import payload_hash
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
from auto_researcher.tasks.feta_seg.runner import restore_prediction_to_native
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    FOLD_ID,
    SPLIT_ID,
    Partition,
    locked_partition,
)
from auto_researcher.tasks.feta_seg.trainer import checkpoint_reference
from auto_researcher.tasks.feta_seg_search.configuration import (
    FeTASegSearchConfiguration,
)
from auto_researcher.tasks.feta_seg_search.trainer import (
    create_loss,
    create_model,
    create_optimizer,
    require_search_environment,
    seed_everything,
    sliding_window_predict,
)
from auto_researcher.tasks.feta_seg_search.transforms import create_transforms
from auto_researcher.tasks.models import TaskRuntimeContext

RUNNER_VERSION = "feta-fold0-search-runner-v1"
DATA_LOADER_VERSION = "monai-persistent-train-spawn4-uncached-validation-search-v1"


def select_fold_zero_subjects(
    subjects: Sequence[FeTASubject], partition: Partition
) -> tuple[tuple[FeTASubject, ...], tuple[FeTASubject, ...]]:
    """Fail closed unless the exact registered 54/14 development split is selected."""

    subject_map = {subject.subject_id: subject for subject in subjects}
    if set(subject_map) != set(partition.development) | set(partition.holdout):
        raise ValueError("feta_search_split_subject_identity_mismatch")
    training = tuple(
        subject_map[subject_id]
        for subject_id in partition.development
        if partition.folds[subject_id] != 0
    )
    validation = tuple(
        subject_map[subject_id]
        for subject_id in partition.development
        if partition.folds[subject_id] == 0
    )
    selected = {subject.subject_id for subject in training + validation}
    if selected & set(partition.holdout):
        raise ValueError("feta_search_holdout_accessed")
    if len(training) != 54 or len(validation) != 14:
        raise ValueError("feta_search_fold_zero_size_mismatch")
    if selected != set(partition.development):
        raise ValueError("feta_search_fold_zero_membership_invalid")
    return training, validation


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
    configuration: FeTASegSearchConfiguration,
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


def run_search_candidate(
    context: TaskRuntimeContext,
    configuration: FeTASegSearchConfiguration,
    experiment_id: str,
) -> dict[str, Any]:
    """Train and endpoint-evaluate exactly one fold-0 development candidate."""

    if context.data_dir is None or context.workspace_dir is None:
        raise RuntimeError("feta_search_runner_paths_missing")
    environment = require_search_environment()
    subjects = inspect_subjects(context.data_dir, inspect_labels=False)
    if len(subjects) != 80 or manifest_hash(subjects) != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_search_dataset_identity_mismatch")
    partition = locked_partition(
        {subject.subject_id: subject.reconstruction_method for subject in subjects}
    )
    training_subjects, validation_subjects = select_fold_zero_subjects(
        subjects, partition
    )

    try:
        import torch
        from monai.data import DataLoader, Dataset, PersistentDataset
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc

    root = context.workspace_dir / "feta_seg_search" / experiment_id
    checkpoint_path = root / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    seed = seed_everything(configuration)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = create_model(configuration).to("cuda")
    loss_function = create_loss(configuration)
    optimizer = create_optimizer(model, configuration)
    scaler = torch.amp.GradScaler("cuda")

    training_dataset = PersistentDataset(
        _dataset_records(training_subjects),
        transform=create_transforms(configuration, training=True),
        cache_dir=root / "cache" / "training",
    )
    # Keep validation uncached so MetaTensor affine metadata survives native restore.
    validation_dataset = Dataset(
        _dataset_records(validation_subjects),
        transform=create_transforms(configuration, training=False),
    )
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=configuration.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        multiprocessing_context="spawn",
        generator=generator,
    )

    validation_schedule = configuration.validation_epochs()
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
                raise ValueError("feta_search_training_loss_non_finite")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch not in validation_schedule:
            continue
        model.eval()
        validation_scores = [
            _macro_dice(
                _native_label(subject),
                _predict_native(
                    model, validation_dataset[index], subject, configuration
                ),
            )
            for index, subject in enumerate(validation_subjects)
        ]
        score = sum(validation_scores) / len(validation_scores)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "fold": configuration.fold,
                    "epoch": epoch,
                    "validation_score": score,
                    "seed": seed,
                    "configuration_identity": payload_hash(configuration),
                    "runner_version": RUNNER_VERSION,
                },
                checkpoint_path,
            )

    torch.cuda.synchronize()
    training_duration = time.perf_counter() - training_started
    if best_epoch == 0 or not checkpoint_path.is_file():
        raise RuntimeError("feta_search_best_checkpoint_missing")
    saved = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    subject_metrics: list[dict[str, Any]] = []
    for index, subject in enumerate(validation_subjects):
        restored = _predict_native(
            model, validation_dataset[index], subject, configuration
        )
        metrics = evaluate_subject_segmentation(
            _native_label(subject), restored, subject.spacing
        )
        subject_metrics.append(
            {
                "subject_id": subject.subject_id,
                "reconstruction_method": subject.reconstruction_method,
                "fold": configuration.fold,
                **metrics,
            }
        )
    torch.cuda.synchronize()
    total_duration = time.perf_counter() - training_started
    aggregate = aggregate_subject_metrics(subject_metrics)
    reference = checkpoint_reference(
        checkpoint_path,
        fold=configuration.fold,
        best_epoch=best_epoch,
        score=best_score,
        output_root=root,
    )
    peak_memory = int(torch.cuda.max_memory_allocated())
    del model, optimizer, scaler, training_loader, training_dataset, validation_dataset
    torch.cuda.empty_cache()
    return {
        **aggregate,
        "best_epoch": best_epoch,
        "validation_score": best_score,
        "training_duration_seconds": training_duration,
        "total_duration_seconds": total_duration,
        "peak_gpu_memory_bytes": peak_memory,
        "validation_epochs": list(validation_schedule),
        "training_subject_count": 54,
        "validation_subject_count": 14,
        "holdout_subjects_evaluated": 0,
        "fold": configuration.fold,
        "configuration_identity": payload_hash(configuration),
        "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
        "split_identity": SPLIT_ID,
        "split_hash": EXPECTED_SPLIT_HASH,
        "fold_identity": FOLD_ID,
        "fold_hash": EXPECTED_FOLD_HASH,
        "checkpoint_reference": reference,
        "environment": environment,
        "environment_identity": payload_hash(environment),
        "runner_version": RUNNER_VERSION,
        "data_loader_version": DATA_LOADER_VERSION,
        "valid_prediction_labels": list(range(8)),
    }
