from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_seg.manifests import FeTASubject
from auto_researcher.tasks.feta_seg.splits import locked_partition
from auto_researcher.tasks.feta_seg_search.cache import (
    CACHE_LOCK_VERSION,
    CACHE_MANIFEST_FILENAME,
    cache_record_hash,
    deterministic_cache_identity,
    mark_shared_cache_complete,
    prepare_or_reuse_shared_cache,
    prepare_shared_cache,
    shared_cache_advisory_lock,
    shared_cache_lock_path,
    shared_cache_root,
    shared_cache_is_complete,
)
from auto_researcher.tasks.feta_seg_search.configuration import (
    FeTASegSearchConfiguration,
)
from auto_researcher.tasks.feta_seg_search.continuation import (
    CONTINUATION_SEMANTICS,
    CONTINUATION_VERSION,
    build_last_checkpoint_payload,
    candidate_trajectory_identity,
    checkpoint_metadata_identity,
    load_resume_plan,
    prediction_set_identity,
    validate_resume_checkpoint_payload,
)
from auto_researcher.tasks.feta_seg_search.metric_tiers import (
    FULL_PANEL_METRIC_NAMES,
    metric_tier_for_fidelity,
)
from auto_researcher.tasks.feta_seg_search.runner import (
    DATA_LOADER_VERSION,
    RUNNER_VERSION,
    RetainedBestPredictions,
    endpoint_metrics_from_predictions,
    materialise_validation_data,
    run_prepared_validation,
    select_fold_zero_subjects,
)


def _methods() -> dict[str, str]:
    return {
        f"sub-{index:03d}": "mial" if index <= 40 else "irtk" for index in range(1, 81)
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
            shape=(2, 2, 2),
            spacing=(0.5, 0.5, 0.5),
            labels=tuple(range(8)),
        )
        for subject_id, method in _methods().items()
    )


def _fold_zero_subjects():
    return select_fold_zero_subjects(_subjects(), locked_partition(_methods()))


def _volume() -> np.ndarray:
    return np.arange(8, dtype=np.uint8).reshape(2, 2, 2)


def _last_payload(
    source_epochs: int,
    *,
    configuration: FeTASegSearchConfiguration | None = None,
) -> dict:
    source = configuration or FeTASegSearchConfiguration(maximum_epochs=source_epochs)
    trajectory = candidate_trajectory_identity(source)
    prediction_identity = prediction_set_identity(
        trajectory,
        source_epochs,
        0.5,
        tuple(f"safe-{index}" for index in range(14)),
    )
    return build_last_checkpoint_payload(
        model_state_dict={"weight": "model"},
        optimizer_state_dict={"state": "optimizer"},
        scaler_state_dict={"scale": 1.0},
        completed_epoch=source_epochs,
        configuration=source,
        trajectory_identity=trajectory,
        runner_version=RUNNER_VERSION,
        data_loader_version=DATA_LOADER_VERSION,
        best_epoch=source_epochs,
        best_score=0.5,
        best_checkpoint_sha256="a" * 64,
        best_prediction_identity=prediction_identity,
        rng_state={"state": "fixture"},
    )


def test_hpo_candidates_share_deterministic_cache_identity():
    training, _ = _fold_zero_subjects()
    first = FeTASegSearchConfiguration(maximum_epochs=25)
    second = FeTASegSearchConfiguration(
        maximum_epochs=150,
        learning_rate=3e-5,
        weight_decay=3e-4,
        dropout=0.4,
        dice_weight=1.5,
        positive_negative_ratio="3:1",
        augmentation_strength="strong",
    )
    assert deterministic_cache_identity(
        first, training
    ) == deterministic_cache_identity(second, training)


def test_cache_record_hash_is_path_free():
    first = {
        "image": Path("/first/export/sub-001_T2w.nii.gz"),
        "label": Path("/first/export/sub-001_dseg.nii.gz"),
        "subject_id": "sub-001",
        "reconstruction_method": "mial",
    }
    second = {
        **first,
        "image": Path("/second/export/sub-001_T2w.nii.gz"),
        "label": Path("/second/export/sub-001_dseg.nii.gz"),
    }
    assert cache_record_hash(first) == cache_record_hash(second)


def test_preprocessing_change_changes_cache_identity():
    training, _ = _fold_zero_subjects()
    configuration = FeTASegSearchConfiguration(maximum_epochs=25)
    assert deterministic_cache_identity(
        configuration, training
    ) != deterministic_cache_identity(
        configuration, training, preprocessing_version="changed-preprocessing"
    )


def test_shared_cache_is_outside_candidate_experiment_root(tmp_path):
    cache = shared_cache_root(tmp_path, "identity")
    candidate = tmp_path / "feta_seg_search" / "experiment-1"
    assert cache == tmp_path / "feta_seg_search" / "_shared_cache" / "identity"
    assert not cache.is_relative_to(candidate)


def test_shared_cache_manifest_mismatch_fails_closed(tmp_path):
    training, _ = _fold_zero_subjects()
    configuration = FeTASegSearchConfiguration(maximum_epochs=25)
    prepared = prepare_shared_cache(tmp_path, configuration, training)
    manifest = prepared.root / CACHE_MANIFEST_FILENAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["orientation"] = "incompatible"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="shared_cache_identity_mismatch"):
        prepare_shared_cache(tmp_path, configuration, training)


def test_complete_shared_cache_is_reused_without_repopulation(tmp_path):
    training, _ = _fold_zero_subjects()
    prepared = prepare_shared_cache(
        tmp_path, FeTASegSearchConfiguration(maximum_epochs=25), training
    )
    for index in range(54):
        (prepared.training_cache_dir / f"{index}.pt").write_bytes(b"cached")
    mark_shared_cache_complete(prepared, expected_items=54)
    assert shared_cache_is_complete(prepared, expected_items=54) is True


def test_shared_cache_population_is_locked_and_second_initializer_rechecks(tmp_path):
    training, _ = _fold_zero_subjects()
    configuration = FeTASegSearchConfiguration(maximum_epochs=25)
    population_calls = 0

    def populate(preparation):
        nonlocal population_calls
        population_calls += 1
        for index in range(54):
            (preparation.training_cache_dir / f"{index}.pt").write_bytes(b"cached")

    first, first_reused = prepare_or_reuse_shared_cache(
        tmp_path, configuration, training, populate=populate
    )
    second, second_reused = prepare_or_reuse_shared_cache(
        tmp_path,
        configuration,
        training,
        populate=lambda _preparation: pytest.fail("completed cache was rebuilt"),
    )
    assert first.identity == second.identity
    assert first_reused is False
    assert second_reused is True
    assert population_calls == 1


def test_partial_shared_cache_fails_closed_after_lock_acquisition(tmp_path):
    training, _ = _fold_zero_subjects()
    configuration = FeTASegSearchConfiguration(maximum_epochs=25)
    preparation = prepare_shared_cache(tmp_path, configuration, training)
    (preparation.training_cache_dir / "partial.pt").write_bytes(b"partial")
    with pytest.raises(ValueError, match="shared_cache_population_partial"):
        prepare_or_reuse_shared_cache(
            tmp_path,
            configuration,
            training,
            populate=lambda _preparation: pytest.fail("partial cache was reused"),
        )


def test_shared_cache_lock_is_cross_process_advisory_and_crash_safe(tmp_path):
    lock_path = shared_cache_lock_path(tmp_path, "identity")
    acquired_path = tmp_path / "child-acquired"
    child_code = """
import fcntl
import pathlib
import sys
lock = pathlib.Path(sys.argv[1])
marker = pathlib.Path(sys.argv[2])
lock.parent.mkdir(parents=True, exist_ok=True)
with lock.open('a+') as handle:
    print('READY', flush=True)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    marker.write_text('acquired', encoding='utf-8')
"""
    with shared_cache_advisory_lock(tmp_path, "identity"):
        child = subprocess.Popen(
            [sys.executable, "-c", child_code, str(lock_path), str(acquired_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        assert acquired_path.exists() is False
    stdout, stderr = child.communicate(timeout=5)
    assert child.returncode == 0, (stdout, stderr)
    assert acquired_path.read_text(encoding="utf-8") == "acquired"


def test_lock_does_not_change_shared_cache_scientific_identity(tmp_path):
    training, _ = _fold_zero_subjects()
    configuration = FeTASegSearchConfiguration(maximum_epochs=25)
    before = deterministic_cache_identity(configuration, training)
    with shared_cache_advisory_lock(tmp_path, before) as lock_path:
        assert lock_path == shared_cache_lock_path(tmp_path, before)
        assert CACHE_LOCK_VERSION == "feta-search-shared-cache-flock-v1"
    assert deterministic_cache_identity(configuration, training) == before


def test_validation_samples_and_native_labels_materialise_once_and_reuse():
    _, validation = _fold_zero_subjects()
    samples = [{"prediction": _volume()} for _ in range(14)]

    class CountingDataset:
        def __init__(self):
            self.calls = 0

        def __getitem__(self, index):
            self.calls += 1
            return samples[index]

    dataset = CountingDataset()
    label_calls = 0

    def load_label(_subject):
        nonlocal label_calls
        label_calls += 1
        return _volume()

    prepared = materialise_validation_data(
        dataset, validation, native_label_loader=load_label
    )
    predictor_calls = 0

    def predict(_model, sample, _subject, _configuration):
        nonlocal predictor_calls
        predictor_calls += 1
        return sample["prediction"]

    configuration = FeTASegSearchConfiguration(maximum_epochs=50)
    first = run_prepared_validation(
        object(), prepared, validation, configuration, predictor=predict
    )
    second = run_prepared_validation(
        object(), prepared, validation, configuration, predictor=predict
    )
    assert first[0] == second[0] == 1.0
    assert dataset.calls == 14
    assert label_calls == 14
    assert predictor_calls == 28


def test_best_predictions_drive_endpoint_without_second_inference():
    _, validation = _fold_zero_subjects()
    samples = [{"prediction": _volume()} for _ in range(14)]

    class Dataset:
        def __getitem__(self, index):
            return samples[index]

    prepared = materialise_validation_data(
        Dataset(), validation, native_label_loader=lambda _subject: _volume()
    )
    predictor_calls = 0

    def predict(_model, sample, _subject, _configuration):
        nonlocal predictor_calls
        predictor_calls += 1
        return sample["prediction"]

    configuration = FeTASegSearchConfiguration(maximum_epochs=25)
    score, predictions, _ = run_prepared_validation(
        object(), prepared, validation, configuration, predictor=predict
    )
    retained = RetainedBestPredictions(
        candidate_trajectory_identity(configuration),
        tuple(subject.subject_id for subject in validation),
    )
    assert retained.consider(25, score, predictions) is True
    metrics = endpoint_metrics_from_predictions(
        "screen", validation, prepared.native_labels, retained.predictions or ()
    )
    assert metrics["mean_subject_macro_dice"] == 1.0
    assert predictor_calls == 14


def test_retained_predictions_correspond_to_best_epoch():
    configuration = FeTASegSearchConfiguration(maximum_epochs=50)
    subject_ids = tuple(f"safe-{index}" for index in range(14))
    retained = RetainedBestPredictions(
        candidate_trajectory_identity(configuration), subject_ids
    )
    predictions = tuple(_volume() for _ in range(14))
    assert retained.consider(25, 0.7, predictions) is True
    assert retained.consider(50, 0.6, predictions) is False
    retained.require_checkpoint_match(
        {
            "epoch": 25,
            "validation_score": 0.7,
            "trajectory_identity": retained.trajectory_identity,
            "prediction_identity": retained.prediction_identity,
        }
    )
    with pytest.raises(ValueError, match="best_prediction_identity_mismatch"):
        retained.require_checkpoint_match(
            {
                "epoch": 50,
                "validation_score": 0.6,
                "trajectory_identity": retained.trajectory_identity,
                "prediction_identity": retained.prediction_identity,
            }
        )


@pytest.mark.parametrize("fidelity", [25, 50])
def test_screen_tier_contains_dice_without_full_panel(fidelity):
    _, validation = _fold_zero_subjects()
    labels = tuple(_volume() for _ in range(14))
    metrics = endpoint_metrics_from_predictions(
        metric_tier_for_fidelity(fidelity), validation, labels, labels
    )
    assert {
        "mean_subject_macro_dice",
        "per_tissue_dice",
        "subject_metrics",
        "reconstruction_macro_dice",
        "reconstruction_gap",
        "empty_prediction_count",
    }.issubset(metrics)
    assert not FULL_PANEL_METRIC_NAMES & set(metrics)


@pytest.mark.parametrize("fidelity", [100, 150, 300, 350])
def test_full_tier_contains_complete_panel(fidelity):
    _, validation = _fold_zero_subjects()
    labels = tuple(_volume() for _ in range(14))
    metrics = endpoint_metrics_from_predictions(
        metric_tier_for_fidelity(fidelity), validation, labels, labels
    )
    assert FULL_PANEL_METRIC_NAMES.issubset(metrics)


def test_last_checkpoint_contains_continuation_state_and_identity():
    payload = _last_payload(25)
    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "completed_epoch",
        "fold",
        "seed",
        "configuration",
        "trajectory_identity",
        "runner_version",
        "data_loader_version",
        "checkpoint_identity",
    }.issubset(payload)
    assert payload["completed_epoch"] == 25
    assert payload["continuation_version"] == CONTINUATION_VERSION
    assert payload["continuation_semantics"] == CONTINUATION_SEMANTICS


def test_trajectory_identity_ignores_fidelity_but_binds_hpo():
    at_25 = FeTASegSearchConfiguration(maximum_epochs=25)
    at_50 = FeTASegSearchConfiguration(maximum_epochs=50)
    changed = FeTASegSearchConfiguration(maximum_epochs=50, dropout=0.3)
    assert candidate_trajectory_identity(at_25) == candidate_trajectory_identity(at_50)
    assert candidate_trajectory_identity(at_25) != candidate_trajectory_identity(
        changed
    )


def test_300_to_350_resume_preserves_trajectory_and_starts_at_301():
    source = FeTASegSearchConfiguration(maximum_epochs=300)
    requested = FeTASegSearchConfiguration(maximum_epochs=350)
    payload = _last_payload(300, configuration=source)
    start, trajectory = validate_resume_checkpoint_payload(
        payload,
        requested,
        expected_runner_version=RUNNER_VERSION,
        expected_data_loader_version=DATA_LOADER_VERSION,
    )
    assert start == 301
    assert trajectory == payload["trajectory_identity"]
    assert candidate_trajectory_identity(source) == candidate_trajectory_identity(
        requested
    )
    assert CONTINUATION_VERSION == "feta-search-stateful-optimisation-continuation-v1"


@pytest.mark.parametrize(
    ("source_epoch", "requested_epoch", "expected_start"),
    [(25, 50, 26), (50, 100, 51), (300, 350, 301)],
)
def test_valid_resume_starts_after_completed_rung(
    source_epoch, requested_epoch, expected_start
):
    payload = _last_payload(source_epoch)
    start, trajectory = validate_resume_checkpoint_payload(
        payload,
        FeTASegSearchConfiguration(maximum_epochs=requested_epoch),
        expected_runner_version=RUNNER_VERSION,
        expected_data_loader_version=DATA_LOADER_VERSION,
    )
    assert start == expected_start
    assert trajectory == payload["trajectory_identity"]


def test_resume_rejects_different_trajectory():
    payload = _last_payload(25)
    with pytest.raises(ValueError, match="resume_trajectory_mismatch"):
        validate_resume_checkpoint_payload(
            payload,
            FeTASegSearchConfiguration(maximum_epochs=50, dropout=0.3),
            expected_runner_version=RUNNER_VERSION,
            expected_data_loader_version=DATA_LOADER_VERSION,
        )


@pytest.mark.parametrize(
    ("source_epoch", "requested_epoch"),
    [(25, 25), (50, 25), (350, 350), (350, 300)],
)
def test_resume_rejects_same_or_lower_fidelity(source_epoch, requested_epoch):
    payload = _last_payload(source_epoch)
    with pytest.raises(ValueError, match="resume_fidelity_not_higher"):
        validate_resume_checkpoint_payload(
            payload,
            FeTASegSearchConfiguration(maximum_epochs=requested_epoch),
            expected_runner_version=RUNNER_VERSION,
            expected_data_loader_version=DATA_LOADER_VERSION,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fold", 1, "resume_fold_mismatch"),
        ("seed", 1, "resume_seed_mismatch"),
    ],
)
def test_resume_rejects_wrong_fold_or_seed(field, value, message):
    payload = _last_payload(25)
    payload[field] = value
    payload["checkpoint_identity"] = checkpoint_metadata_identity(payload)
    with pytest.raises(ValueError, match=message):
        validate_resume_checkpoint_payload(
            payload,
            FeTASegSearchConfiguration(maximum_epochs=50),
            expected_runner_version=RUNNER_VERSION,
            expected_data_loader_version=DATA_LOADER_VERSION,
        )


def test_resume_rejects_wrong_checkpoint_identity():
    payload = _last_payload(25)
    payload["checkpoint_identity"] = payload_hash({"wrong": True})
    with pytest.raises(ValueError, match="resume_checkpoint_identity_mismatch"):
        validate_resume_checkpoint_payload(
            payload,
            FeTASegSearchConfiguration(maximum_epochs=50),
            expected_runner_version=RUNNER_VERSION,
            expected_data_loader_version=DATA_LOADER_VERSION,
        )


def test_missing_resume_checkpoint_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="resume_checkpoint_missing"):
        load_resume_plan(
            tmp_path / "missing-candidate",
            FeTASegSearchConfiguration(maximum_epochs=50),
            expected_runner_version=RUNNER_VERSION,
            expected_data_loader_version=DATA_LOADER_VERSION,
            map_location="cpu",
        )


@pytest.mark.parametrize(
    ("source_epoch", "requested_epoch", "expected_start"),
    [(25, 50, 26), (300, 350, 301)],
)
def test_load_resume_plan_verifies_files_and_returns_next_epoch(
    tmp_path, source_epoch, requested_epoch, expected_start
):
    import torch

    checkpoint_root = tmp_path / "source" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    source = FeTASegSearchConfiguration(maximum_epochs=source_epoch)
    trajectory = candidate_trajectory_identity(source)
    prediction_identity = prediction_set_identity(
        trajectory,
        source_epoch,
        0.5,
        tuple(f"safe-{index}" for index in range(14)),
    )
    best_payload = {
        "model_state_dict": {},
        "fold": 0,
        "epoch": source_epoch,
        "validation_score": 0.5,
        "seed": 20260807,
        "trajectory_identity": trajectory,
        "prediction_identity": prediction_identity,
    }
    best_path = checkpoint_root / "best.pt"
    torch.save(best_payload, best_path)
    best_sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
    last_payload = build_last_checkpoint_payload(
        model_state_dict={},
        optimizer_state_dict={},
        scaler_state_dict={},
        completed_epoch=source_epoch,
        configuration=source,
        trajectory_identity=trajectory,
        runner_version=RUNNER_VERSION,
        data_loader_version=DATA_LOADER_VERSION,
        best_epoch=source_epoch,
        best_score=0.5,
        best_checkpoint_sha256=best_sha,
        best_prediction_identity=prediction_identity,
        rng_state={},
    )
    torch.save(last_payload, checkpoint_root / "last.pt")
    plan = load_resume_plan(
        tmp_path / "source",
        FeTASegSearchConfiguration(maximum_epochs=requested_epoch),
        expected_runner_version=RUNNER_VERSION,
        expected_data_loader_version=DATA_LOADER_VERSION,
        map_location="cpu",
    )
    assert plan.completed_epoch == source_epoch
    assert plan.start_epoch == expected_start
    assert plan.source_best_checkpoint_sha256 == best_sha
    assert plan.source_last_checkpoint.name == "last.pt"
    assert plan.source_best_checkpoint.name == "best.pt"
    assert plan.last_payload["model_state_dict"] == {}
    assert plan.last_payload["optimizer_state_dict"] == {}
    assert plan.last_payload["scaler_state_dict"] == {}
    assert plan.last_payload["rng_state"] == {}
