"""Host-owned CUDA runner interpreting bounded FeTA TrainingPolicy candidates."""

from __future__ import annotations

import time
from typing import Any

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_seg.manifests import (
    EXPECTED_MANIFEST_HASH,
    inspect_subjects,
    manifest_hash,
)
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    FOLD_ID,
    SPLIT_ID,
    locked_partition,
)
from auto_researcher.tasks.feta_seg.trainer import checkpoint_reference
from auto_researcher.tasks.feta_seg_evolve.configuration import (
    FeTASegEvolveConfiguration,
)
from auto_researcher.tasks.feta_seg_evolve.training_policy import (
    TRAINING_POLICY_VERSION,
)
from auto_researcher.tasks.feta_seg_evolve.transforms import create_evolve_transforms
from auto_researcher.tasks.feta_seg_search.cache import (
    cache_record_hash,
    prepare_or_reuse_shared_cache,
)
from auto_researcher.tasks.feta_seg_search.gpu_scheduler import (
    gpu_scheduler_policy,
    wait_for_gpu_admission,
)
from auto_researcher.tasks.feta_seg_search.metric_tiers import (
    METRIC_TIER_POLICY_VERSION,
    metric_tier_for_fidelity,
)
from auto_researcher.tasks.feta_seg_search.runner import (
    RetainedBestPredictions,
    _dataset_records,
    endpoint_metrics_from_predictions,
    materialise_validation_data,
    run_prepared_validation,
    select_fold_zero_subjects,
)
from auto_researcher.tasks.feta_seg_search.trainer import (
    create_model,
    create_optimizer,
    require_search_environment,
    seed_everything,
)
from auto_researcher.tasks.feta_seg_search.configuration import validation_epochs
from auto_researcher.tasks.models import TaskRuntimeContext

EVOLVE_RUNNER_VERSION = "feta-fold0-training-policy-runner-v1"
EVOLVE_DATA_LOADER_VERSION = "monai-shared-persistent-train-spawn4-evolve-v1"
EVOLVE_LOSS_VERSION = "dice-ce-dynamic-diceweight-ce1-v1"
EVOLVE_OPTIMISER_VERSION = "adamw-host-interpreted-lr-policy-v1"


def _create_loss(dice_weight: float):
    try:
        from monai.losses import DiceCELoss
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=dice_weight,
        lambda_ce=1.0,
    )


def _set_learning_rate(optimizer: Any, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def policy_trace(configuration: FeTASegEvolveConfiguration) -> list[dict[str, Any]]:
    maximum = configuration.maximum_epochs
    epochs = tuple(sorted({1, max(1, (maximum + 1) // 2), maximum}))
    return [
        {
            "epoch": epoch,
            "learning_rate": configuration.training_policy.learning_rate_at(
                epoch, maximum, configuration.learning_rate
            ),
            "dice_weight": configuration.training_policy.dice_weight_at(epoch, maximum),
        }
        for epoch in epochs
    ]


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    configuration: FeTASegEvolveConfiguration,
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
        "configuration": configuration.model_dump(mode="json"),
        "configuration_identity": payload_hash(configuration),
        "trajectory_identity": trajectory_identity,
        "prediction_identity": prediction_identity,
        "runner_version": EVOLVE_RUNNER_VERSION,
        "training_policy_version": TRAINING_POLICY_VERSION,
    }


def run_evolve_candidate(
    context: TaskRuntimeContext,
    configuration: FeTASegEvolveConfiguration,
    experiment_id: str,
) -> dict[str, Any]:
    """Evaluate one policy while retaining all data, training and scoring authority."""

    total_started = time.perf_counter()
    if context.data_dir is None or context.workspace_dir is None:
        raise RuntimeError("feta_evolve_runner_paths_missing")
    scheduler_policy = gpu_scheduler_policy(context)
    environment = require_search_environment()
    subjects = inspect_subjects(context.data_dir, inspect_labels=False)
    if len(subjects) != 80 or manifest_hash(subjects) != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_evolve_dataset_identity_mismatch")
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

    root = context.workspace_dir / "feta_seg_evolve" / experiment_id
    checkpoint_root = root / "checkpoints"
    best_checkpoint_path = checkpoint_root / "best.pt"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    trajectory_identity = payload_hash(
        {
            "base_configuration": configuration.base_configuration,
            "training_policy": configuration.training_policy,
        }
    )
    retained = RetainedBestPredictions(
        trajectory_identity,
        tuple(subject.subject_id for subject in validation_subjects),
    )

    cache_started = time.perf_counter()

    def populate_shared_cache(preparation: Any) -> None:
        dataset = PersistentDataset(
            _dataset_records(training_subjects),
            transform=create_evolve_transforms(configuration, training=False),
            cache_dir=preparation.training_cache_dir,
            hash_func=cache_record_hash,
        )
        for index in range(len(dataset)):
            dataset[index]

    shared_cache, cache_reused = prepare_or_reuse_shared_cache(
        context.workspace_dir,
        configuration,  # type: ignore[arg-type]
        training_subjects,
        populate=populate_shared_cache,
    )
    cache_prepare_seconds = time.perf_counter() - cache_started

    validation_dataset = Dataset(
        _dataset_records(validation_subjects),
        transform=create_evolve_transforms(configuration, training=False),
    )
    prepared_validation = materialise_validation_data(
        validation_dataset, validation_subjects
    )
    del validation_dataset

    admission = wait_for_gpu_admission(
        scheduler_policy, maximum_epochs=configuration.maximum_epochs
    )
    admission_metrics = {} if admission is None else admission.as_metrics()
    seed = seed_everything(configuration)  # type: ignore[arg-type]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = create_model(configuration).to("cuda")  # type: ignore[arg-type]
    optimizer = create_optimizer(model, configuration)  # type: ignore[arg-type]
    scaler = torch.amp.GradScaler("cuda")
    training_dataset = PersistentDataset(
        _dataset_records(training_subjects),
        transform=create_evolve_transforms(configuration, training=True),
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

    validation_schedule = validation_epochs(configuration.maximum_epochs)
    training_seconds = 0.0
    validation_inference_seconds = 0.0
    for epoch in range(1, configuration.maximum_epochs + 1):
        learning_rate = configuration.training_policy.learning_rate_at(
            epoch, configuration.maximum_epochs, configuration.learning_rate
        )
        dice_weight = configuration.training_policy.dice_weight_at(
            epoch, configuration.maximum_epochs
        )
        _set_learning_rate(optimizer, learning_rate)
        loss_function = _create_loss(dice_weight)
        epoch_started = time.perf_counter()
        model.train()
        for batch in training_loader:
            inputs = batch["image"].to(device="cuda", non_blocking=True)
            labels = batch["label"].to(device="cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_function(model(inputs), labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError("feta_evolve_training_loss_non_finite")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - epoch_started
        if epoch not in validation_schedule:
            continue
        model.eval()
        score, predictions, inference_seconds = run_prepared_validation(
            model,
            prepared_validation,
            validation_subjects,
            configuration,  # type: ignore[arg-type]
        )
        validation_inference_seconds += inference_seconds
        if retained.consider(epoch, score, predictions):
            assert retained.prediction_identity is not None
            torch.save(
                _checkpoint_payload(
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
        raise RuntimeError("feta_evolve_best_checkpoint_missing")
    saved_best = torch.load(best_checkpoint_path, map_location="cpu", weights_only=True)
    retained.require_checkpoint_match(saved_best)
    del saved_best
    best_reference = checkpoint_reference(
        best_checkpoint_path,
        fold=configuration.fold,
        best_epoch=retained.epoch,
        score=retained.score,
        output_root=root,
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
        "validation_inference_seconds": validation_inference_seconds,
        "endpoint_metric_seconds": endpoint_metric_seconds,
        "total_duration_seconds": time.perf_counter() - total_started,
        "peak_gpu_memory_bytes": peak_memory,
        "validation_epochs": list(validation_schedule),
        "training_subject_count": 54,
        "validation_subject_count": 14,
        "holdout_subjects_evaluated": 0,
        "fold": configuration.fold,
        "configuration_identity": payload_hash(configuration),
        "trajectory_identity": trajectory_identity,
        "base_configuration_identity": configuration.base_configuration_identity,
        "training_policy_identity": configuration.training_policy_identity,
        "policy_trace": policy_trace(configuration),
        "cache_identity": shared_cache.identity,
        "cache_identity_version": shared_cache.identity_version,
        "cache_reused": cache_reused,
        "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
        "split_identity": SPLIT_ID,
        "split_hash": EXPECTED_SPLIT_HASH,
        "fold_identity": FOLD_ID,
        "fold_hash": EXPECTED_FOLD_HASH,
        "checkpoint_reference": best_reference,
        "environment": environment,
        "environment_identity": payload_hash(environment),
        "runner_version": EVOLVE_RUNNER_VERSION,
        "data_loader_version": EVOLVE_DATA_LOADER_VERSION,
        "valid_prediction_labels": list(range(8)),
        "candidate_provenance": configuration.candidate_provenance.model_dump(
            mode="json"
        ),
        "seeding_mode": configuration.seeding_mode,
    }
