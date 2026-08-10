"""Throughput-oriented single-fold CUDA runner for FeTA search candidates."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
from auto_researcher.tasks.feta_seg_search.cache import (
    cache_record_hash,
    prepare_or_reuse_shared_cache,
)
from auto_researcher.tasks.feta_seg_search.configuration import (
    FeTASegSearchConfiguration,
)
from auto_researcher.tasks.feta_seg_search.continuation import (
    CONTINUATION_SEMANTICS,
    CONTINUATION_VERSION,
    build_last_checkpoint_payload,
    candidate_trajectory_identity,
    capture_rng_state,
    checkpoint_file_reference,
    load_resume_plan,
    prediction_set_identity,
    restore_rng_state,
)
from auto_researcher.tasks.feta_seg_search.metric_tiers import (
    METRIC_TIER_POLICY_VERSION,
    MetricTier,
    aggregate_screen_subject_metrics,
    evaluate_screen_subject,
    metric_tier_for_fidelity,
)
from auto_researcher.tasks.feta_seg_search.gpu_scheduler import (
    gpu_scheduler_policy,
    wait_for_gpu_admission,
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

RUNNER_VERSION = "feta-fold0-search-runner-v2"
DATA_LOADER_VERSION = "monai-shared-persistent-train-spawn4-prepared-validation-v2"


@dataclass(frozen=True)
class PreparedValidationData:
    samples: tuple[dict[str, Any], ...]
    native_labels: tuple[Any, ...]
    prepare_seconds: float


@dataclass
class RetainedBestPredictions:
    trajectory_identity: str
    validation_subject_ids: tuple[str, ...]
    epoch: int = 0
    score: float = -1.0
    predictions: tuple[Any, ...] | None = None
    prediction_identity: str | None = None

    def consider(
        self, epoch: int, score: float, predictions: Sequence[Any]
    ) -> bool:
        if len(predictions) != len(self.validation_subject_ids):
            raise ValueError("feta_search_validation_prediction_count_mismatch")
        if score <= self.score:
            return False
        self.epoch = epoch
        self.score = score
        self.predictions = tuple(predictions)
        self.prediction_identity = prediction_set_identity(
            self.trajectory_identity,
            epoch,
            score,
            self.validation_subject_ids,
        )
        return True

    def require_checkpoint_match(self, checkpoint: dict[str, Any]) -> None:
        if (
            self.predictions is None
            or self.prediction_identity is None
            or checkpoint.get("epoch") != self.epoch
            or not math.isclose(
                float(checkpoint.get("validation_score", -1.0)),
                self.score,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or checkpoint.get("trajectory_identity") != self.trajectory_identity
            or checkpoint.get("prediction_identity") != self.prediction_identity
        ):
            raise ValueError("feta_search_best_prediction_identity_mismatch")


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


def materialise_validation_data(
    validation_dataset: Any,
    validation_subjects: Sequence[FeTASubject],
    *,
    native_label_loader: Callable[[FeTASubject], Any] = _native_label,
) -> PreparedValidationData:
    started = time.perf_counter()
    samples = tuple(
        validation_dataset[index] for index in range(len(validation_subjects))
    )
    native_labels = tuple(
        native_label_loader(subject) for subject in validation_subjects
    )
    if len(samples) != 14 or len(native_labels) != 14:
        raise ValueError("feta_search_validation_preparation_incomplete")
    return PreparedValidationData(
        samples=samples,
        native_labels=native_labels,
        prepare_seconds=time.perf_counter() - started,
    )


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


def run_prepared_validation(
    model: Any,
    prepared: PreparedValidationData,
    validation_subjects: Sequence[FeTASubject],
    configuration: FeTASegSearchConfiguration,
    *,
    predictor: Callable[[Any, dict[str, Any], FeTASubject, FeTASegSearchConfiguration], Any] = _predict_native,
) -> tuple[float, tuple[Any, ...], float]:
    started = time.perf_counter()
    predictions = tuple(
        predictor(model, prepared.samples[index], subject, configuration)
        for index, subject in enumerate(validation_subjects)
    )
    scores = [
        float(evaluate_screen_subject(actual, predicted)["macro_dice"])
        for actual, predicted in zip(
            prepared.native_labels, predictions, strict=True
        )
    ]
    return sum(scores) / len(scores), predictions, time.perf_counter() - started


def endpoint_metrics_from_predictions(
    tier: MetricTier,
    validation_subjects: Sequence[FeTASubject],
    native_labels: Sequence[Any],
    predictions: Sequence[Any],
    *,
    fold: int = 0,
) -> dict[str, Any]:
    if not (
        len(validation_subjects) == len(native_labels) == len(predictions) == 14
    ):
        raise ValueError("feta_search_endpoint_prediction_count_mismatch")
    rows: list[dict[str, Any]] = []
    for subject, actual, predicted in zip(
        validation_subjects, native_labels, predictions, strict=True
    ):
        metrics = (
            evaluate_screen_subject(actual, predicted)
            if tier == "screen"
            else evaluate_subject_segmentation(actual, predicted, subject.spacing)
        )
        rows.append(
            {
                "subject_id": subject.subject_id,
                "reconstruction_method": subject.reconstruction_method,
                "fold": fold,
                **metrics,
            }
        )
    return (
        aggregate_screen_subject_metrics(rows)
        if tier == "screen"
        else aggregate_subject_metrics(rows)
    )


def _best_checkpoint_payload(
    model: Any,
    optimizer: Any,
    configuration: FeTASegSearchConfiguration,
    *,
    epoch: int,
    score: float,
    trajectory_identity: str,
    prediction_identity: str,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "fold": configuration.fold,
        "epoch": epoch,
        "validation_score": score,
        "seed": configuration.seed + configuration.fold,
        "configuration": configuration.scientific_configuration(),
        "configuration_identity": payload_hash(configuration),
        "trajectory_identity": trajectory_identity,
        "prediction_identity": prediction_identity,
        "runner_version": RUNNER_VERSION,
        "data_loader_version": DATA_LOADER_VERSION,
    }


def run_search_candidate(
    context: TaskRuntimeContext,
    configuration: FeTASegSearchConfiguration,
    experiment_id: str,
) -> dict[str, Any]:
    """Train or promote exactly one fold-0 candidate and reuse best predictions."""

    total_started = time.perf_counter()
    if context.data_dir is None or context.workspace_dir is None:
        raise RuntimeError("feta_search_runner_paths_missing")
    scheduler_policy = gpu_scheduler_policy(context)
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
        import numpy as np
        import torch
        from monai.data import DataLoader, Dataset, PersistentDataset
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc

    root = context.workspace_dir / "feta_seg_search" / experiment_id
    checkpoint_root = root / "checkpoints"
    best_checkpoint_path = checkpoint_root / "best.pt"
    last_checkpoint_path = checkpoint_root / "last.pt"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    trajectory_identity = candidate_trajectory_identity(configuration)
    validation_subject_ids = tuple(
        subject.subject_id for subject in validation_subjects
    )
    retained = RetainedBestPredictions(
        trajectory_identity, validation_subject_ids
    )

    resume_plan = None
    resume_value = context.task_options.get("resume_candidate_root")
    if resume_value is not None:
        if not isinstance(resume_value, str) or not resume_value.strip():
            raise ValueError("feta_search_resume_candidate_root_invalid")
        resume_plan = load_resume_plan(
            Path(resume_value),
            configuration,
            expected_runner_version=RUNNER_VERSION,
            expected_data_loader_version=DATA_LOADER_VERSION,
            map_location="cpu",
        )
        if resume_plan.source_candidate_root == root.resolve():
            raise ValueError("feta_search_resume_candidate_root_invalid")

    cache_started = time.perf_counter()
    def populate_shared_cache(preparation: Any) -> None:
        deterministic_training_dataset = PersistentDataset(
            _dataset_records(training_subjects),
            transform=create_transforms(configuration, training=False),
            cache_dir=preparation.training_cache_dir,
            hash_func=cache_record_hash,
        )
        for index in range(len(deterministic_training_dataset)):
            deterministic_training_dataset[index]
        del deterministic_training_dataset

    shared_cache, cache_reused = prepare_or_reuse_shared_cache(
        context.workspace_dir,
        configuration,
        training_subjects,
        populate=populate_shared_cache,
    )
    cache_prepare_seconds = time.perf_counter() - cache_started

    validation_dataset = Dataset(
        _dataset_records(validation_subjects),
        transform=create_transforms(configuration, training=False),
    )
    prepared_validation = materialise_validation_data(
        validation_dataset, validation_subjects
    )
    del validation_dataset

    admission = wait_for_gpu_admission(
        scheduler_policy, maximum_epochs=configuration.maximum_epochs
    )
    admission_metrics = {} if admission is None else admission.as_metrics()
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
        cache_dir=shared_cache.training_cache_dir,
        hash_func=cache_record_hash,
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

    resumed = False
    resumed_from_epoch: int | None = None
    source_checkpoint_sha256: str | None = None
    start_epoch = 1
    validation_inference_seconds = 0.0
    if resume_plan is not None:
        resumed = True
        resumed_from_epoch = resume_plan.completed_epoch
        source_checkpoint_sha256 = resume_plan.source_checkpoint_sha256
        start_epoch = resume_plan.start_epoch

        try:
            model.load_state_dict(resume_plan.best_payload["model_state_dict"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("feta_search_resume_best_model_state_invalid") from exc
        model.eval()
        source_score, source_predictions, inference_seconds = run_prepared_validation(
            model, prepared_validation, validation_subjects, configuration
        )
        validation_inference_seconds += inference_seconds
        if not math.isclose(
            source_score,
            float(resume_plan.best_payload["validation_score"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("feta_search_resume_best_score_mismatch")
        retained.consider(
            int(resume_plan.best_payload["epoch"]),
            float(resume_plan.best_payload["validation_score"]),
            source_predictions,
        )
        assert retained.prediction_identity is not None
        if (
            retained.prediction_identity
            != resume_plan.best_payload["prediction_identity"]
        ):
            raise ValueError("feta_search_resume_best_prediction_identity_mismatch")
        torch.save(
            _best_checkpoint_payload(
                model,
                optimizer,
                configuration,
                epoch=retained.epoch,
                score=retained.score,
                trajectory_identity=trajectory_identity,
                prediction_identity=retained.prediction_identity,
            ),
            best_checkpoint_path,
        )
        try:
            model.load_state_dict(resume_plan.last_payload["model_state_dict"])
            optimizer.load_state_dict(
                resume_plan.last_payload["optimizer_state_dict"]
            )
            scaler.load_state_dict(resume_plan.last_payload["scaler_state_dict"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("feta_search_resume_optimisation_state_invalid") from exc
        restore_rng_state(
            resume_plan.last_payload["rng_state"], torch, np, generator
        )
        del resume_plan

    validation_schedule = configuration.validation_epochs()
    training_seconds = 0.0
    for epoch in range(start_epoch, configuration.maximum_epochs + 1):
        epoch_started = time.perf_counter()
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
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - epoch_started

        if epoch not in validation_schedule:
            continue
        model.eval()
        score, predictions, inference_seconds = run_prepared_validation(
            model, prepared_validation, validation_subjects, configuration
        )
        validation_inference_seconds += inference_seconds
        if retained.consider(epoch, score, predictions):
            assert retained.prediction_identity is not None
            torch.save(
                _best_checkpoint_payload(
                    model,
                    optimizer,
                    configuration,
                    epoch=epoch,
                    score=score,
                    trajectory_identity=trajectory_identity,
                    prediction_identity=retained.prediction_identity,
                ),
                best_checkpoint_path,
            )

    torch.cuda.synchronize()
    if retained.predictions is None or not best_checkpoint_path.is_file():
        raise RuntimeError("feta_search_best_checkpoint_missing")
    saved_best = torch.load(
        best_checkpoint_path, map_location="cpu", weights_only=True
    )
    retained.require_checkpoint_match(saved_best)
    del saved_best
    best_reference = checkpoint_reference(
        best_checkpoint_path,
        fold=configuration.fold,
        best_epoch=retained.epoch,
        score=retained.score,
        output_root=root,
    )

    last_payload = build_last_checkpoint_payload(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scaler_state_dict=scaler.state_dict(),
        completed_epoch=configuration.maximum_epochs,
        configuration=configuration,
        trajectory_identity=trajectory_identity,
        runner_version=RUNNER_VERSION,
        data_loader_version=DATA_LOADER_VERSION,
        best_epoch=retained.epoch,
        best_score=retained.score,
        best_checkpoint_sha256=best_reference["sha256"],
        best_prediction_identity=retained.prediction_identity or "",
        rng_state=capture_rng_state(torch, np, generator),
    )
    torch.save(last_payload, last_checkpoint_path)
    last_reference = checkpoint_file_reference(
        last_checkpoint_path,
        output_root=root,
        checkpoint_type="continuation-last",
        completed_epoch=configuration.maximum_epochs,
        trajectory_identity=trajectory_identity,
    )

    metric_tier = metric_tier_for_fidelity(configuration.maximum_epochs)
    endpoint_started = time.perf_counter()
    aggregate = endpoint_metrics_from_predictions(
        metric_tier,
        validation_subjects,
        prepared_validation.native_labels,
        retained.predictions,
        fold=configuration.fold,
    )
    endpoint_metric_seconds = time.perf_counter() - endpoint_started
    total_duration = time.perf_counter() - total_started
    peak_memory = int(torch.cuda.max_memory_allocated())
    del model, optimizer, scaler, training_loader, training_dataset
    torch.cuda.empty_cache()
    return {
        **aggregate,
        **admission_metrics,
        "metric_tier": metric_tier,
        "metric_tier_policy_version": METRIC_TIER_POLICY_VERSION,
        "best_epoch": retained.epoch,
        "validation_score": retained.score,
        "cache_prepare_seconds": cache_prepare_seconds,
        "validation_prepare_seconds": prepared_validation.prepare_seconds,
        "training_seconds": training_seconds,
        "training_duration_seconds": training_seconds,
        "validation_inference_seconds": validation_inference_seconds,
        "endpoint_metric_seconds": endpoint_metric_seconds,
        "total_duration_seconds": total_duration,
        "peak_gpu_memory_bytes": peak_memory,
        "duplicate_endpoint_inference_avoided": True,
        "validation_epochs": list(validation_schedule),
        "training_subject_count": 54,
        "validation_subject_count": 14,
        "holdout_subjects_evaluated": 0,
        "fold": configuration.fold,
        "configuration_identity": payload_hash(configuration),
        "trajectory_identity": trajectory_identity,
        "cache_identity": shared_cache.identity,
        "cache_identity_version": shared_cache.identity_version,
        "cache_reused": cache_reused,
        "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
        "split_identity": SPLIT_ID,
        "split_hash": EXPECTED_SPLIT_HASH,
        "fold_identity": FOLD_ID,
        "fold_hash": EXPECTED_FOLD_HASH,
        "checkpoint_reference": best_reference,
        "last_checkpoint_reference": last_reference,
        "environment": environment,
        "environment_identity": payload_hash(environment),
        "runner_version": RUNNER_VERSION,
        "data_loader_version": DATA_LOADER_VERSION,
        "valid_prediction_labels": list(range(8)),
        "resumed": resumed,
        "resumed_from_epoch": resumed_from_epoch,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "continuation_version": CONTINUATION_VERSION,
        "continuation_semantics": CONTINUATION_SEMANTICS,
    }
