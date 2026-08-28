"""One-time sealed-holdout evaluation for one frozen five-fold FeTA model family."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.artifacts import atomic_json_write
from auto_researcher.tasks.feta_seg.manifests import (
    EXPECTED_MANIFEST_HASH,
    FeTASubject,
    inspect_subjects,
    manifest_hash,
)
from auto_researcher.tasks.feta_seg.runner import restore_prediction_to_native
from auto_researcher.tasks.feta_seg.splits import locked_partition
from auto_researcher.tasks.feta_seg.transforms import (
    PREPROCESSING_VERSION,
    create_transforms,
)
from auto_researcher.tasks.feta_unet_direct.model import create_unet_model
from auto_researcher.tasks.feta_unet_direct.trainer import (
    require_full_baseline_environment,
    sliding_window_predict,
)
from auto_researcher.tasks.feta_unet_ensemble.aggregation import (
    member_identity,
    predicted_labels,
)
from auto_researcher.tasks.feta_unet_ensemble.cache import (
    load_probability_cache,
    write_probability_cache,
)
from auto_researcher.tasks.feta_unet_ensemble.cross_validation import (
    CrossValidationMemberSource,
    load_cross_validation_member_source,
)
from auto_researcher.tasks.feta_unet_ensemble.evaluation import (
    _dice_row,
    _native_label,
    _public_summary,
    _read_json,
)
from auto_researcher.tasks.feta_unet_ensemble.models import ProbabilityCacheRecord

FINAL_HOLDOUT_MANIFEST_SCHEMA = "feta-unet-final-holdout-manifest-v1"
FINAL_HOLDOUT_REPORT_SCHEMA = "feta-unet-final-holdout-report-v1"
FINAL_HOLDOUT_SELECTION_RULE = (
    "one frozen V8 DynUNet family; equal probability mean across its five "
    "fold-specific checkpoints; no post-processing"
)
BOOTSTRAP_SEED = 20260828
BOOTSTRAP_SAMPLES = 20_000


def subject_bootstrap_interval(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Return a deterministic percentile interval over subject macro-Dice."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    values = np.asarray([float(row["macro_dice"]) for row in rows], dtype=np.float64)
    if len(values) != 12 or not bool(np.isfinite(values).all()):
        raise ValueError("feta_unet_final_holdout_bootstrap_input_invalid")
    if samples < 1_000:
        raise ValueError("feta_unet_final_holdout_bootstrap_samples_invalid")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    lower, upper = np.percentile(estimates, (2.5, 97.5))
    return {
        "method": "subject-level percentile bootstrap",
        "confidence_level": 0.95,
        "seed": seed,
        "samples": samples,
        "lower": float(lower),
        "upper": float(upper),
    }


def _average_fold_probabilities(values: Sequence[Any]):
    """Average exactly five fold predictions without broadening member ensembles."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    arrays = tuple(np.asarray(value, dtype=np.float32) for value in values)
    if len(arrays) != 5 or not arrays or arrays[0].ndim != 4:
        raise ValueError("feta_unet_final_holdout_fold_probability_count_invalid")
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("feta_unet_final_holdout_fold_probability_shape_invalid")
    combined = np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float32)
    if not bool(np.isfinite(combined).all()) or not bool(
        np.allclose(combined.sum(axis=0), 1.0, atol=1e-4)
    ):
        raise ValueError("feta_unet_final_holdout_probability_invalid")
    return combined


def _validate_release_manifest(
    manifest: dict[str, Any],
) -> CrossValidationMemberSource:
    if set(manifest) != {"schema_version", "evaluation_id", "decision", "member"}:
        raise ValueError("feta_unet_final_holdout_manifest_invalid")
    decision = manifest.get("decision")
    if (
        manifest.get("schema_version") != FINAL_HOLDOUT_MANIFEST_SCHEMA
        or not isinstance(manifest.get("evaluation_id"), str)
        or not manifest["evaluation_id"]
        or not isinstance(decision, dict)
        or set(decision)
        != {
            "scope",
            "candidate_family",
            "inference_rule",
            "post_processing",
            "selection_frozen_before_holdout",
            "result_feedback_prohibited",
        }
        or decision.get("scope") != "one-time-final-sealed-holdout"
        or decision.get("candidate_family") != "v8-dynunet"
        or decision.get("inference_rule") != "equal-probability-mean-five-fold"
        or decision.get("post_processing") != "none"
        or decision.get("selection_frozen_before_holdout") is not True
        or decision.get("result_feedback_prohibited") is not True
        or not isinstance(manifest.get("member"), dict)
    ):
        raise ValueError("feta_unet_final_holdout_manifest_invalid")
    source = load_cross_validation_member_source(manifest["member"])
    configuration = source.configuration
    if (
        configuration.profile != "five_fold_confirmation"
        or configuration.fold_count != 5
        or configuration.maximum_epochs != 150
        or configuration.model_variant != "dynunet"
        or configuration.feature_width != "v8_dyn_balanced_5"
        or configuration.architecture_budget != "dynunet-15m-150m-v1"
    ):
        raise ValueError("feta_unet_final_holdout_candidate_not_frozen_v8_dynunet")
    return source


def _prepare_release_output(
    output_root: Path, manifest: dict[str, Any], manifest_identity: str
) -> None:
    """Create a release root or admit only an exact incomplete same-run retry."""

    release_path = output_root / "release-manifest.json"
    if output_root.exists():
        if (
            not output_root.is_dir()
            or not release_path.is_file()
            or (output_root / "final-holdout-report.json").exists()
            or (output_root / "protected-subject-metrics.json").exists()
            or _read_json(release_path).get("manifest_identity") != manifest_identity
        ):
            raise ValueError("feta_unet_final_holdout_output_exists")
        return
    output_root.mkdir(parents=True, mode=0o700)
    os.chmod(output_root, 0o700)
    atomic_json_write(
        release_path,
        {**manifest, "manifest_identity": manifest_identity},
    )
    os.chmod(release_path, 0o600)


def _fold_cache_paths(
    cache_root: Path,
    source: CrossValidationMemberSource,
    fold: int,
    subject: FeTASubject,
) -> tuple[Path, Path, str]:
    fold_identity = payload_hash(
        {
            "member_identity": member_identity(source.member),
            "fold": fold,
            "checkpoint_sha256": source.checkpoint_sha256s[fold],
        }
    )
    root = cache_root / fold_identity
    return (
        root / f"{subject.subject_id}.npy",
        root / f"{subject.subject_id}.json",
        fold_identity,
    )


def _predict_holdout_fold(
    source: CrossValidationMemberSource,
    *,
    fold: int,
    subjects: Sequence[FeTASubject],
    dataset: Any,
    cache_root: Path,
) -> None:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    model = create_unet_model(source.configuration)
    saved = torch.load(
        source.checkpoint_paths[fold], map_location="cpu", weights_only=True
    )
    if (
        not isinstance(saved, dict)
        or saved.get("configuration_identity")
        != source.member.configuration_identity
        or saved.get("architecture_identity") != source.member.architecture_identity
        or int(saved.get("fold", -1)) != fold
        or not isinstance(saved.get("model_state_dict"), dict)
    ):
        raise ValueError("feta_unet_final_holdout_checkpoint_payload_invalid")
    try:
        model.load_state_dict(saved["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError("feta_unet_final_holdout_checkpoint_model_mismatch") from exc
    model = model.to("cuda")
    model.eval()
    for index, subject in enumerate(subjects):
        probability_path, record_path, fold_identity = _fold_cache_paths(
            cache_root, source, fold, subject
        )
        if probability_path.exists() or record_path.exists():
            if not probability_path.is_file() or not record_path.is_file():
                raise ValueError("feta_unet_final_holdout_cache_partial")
            record = ProbabilityCacheRecord.model_validate(_read_json(record_path))
            if (
                record.subject_id != subject.subject_id
                or record.member_identity != fold_identity
            ):
                raise ValueError("feta_unet_final_holdout_cache_identity_mismatch")
            load_probability_cache(probability_path, record)
            continue
        inputs = dataset[index]["image"].unsqueeze(0).to("cuda", non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            probability = torch.softmax(
                sliding_window_predict(inputs, model, source.configuration), dim=1
            )[0]
        tensor = probability.to(device="cpu", dtype=torch.float32).numpy()
        if not bool(np.isfinite(tensor).all()):
            raise ValueError("feta_unet_final_holdout_probability_non_finite")
        record = write_probability_cache(
            probability_path,
            tensor,
            subject_id=subject.subject_id,
            member_identity=fold_identity,
        )
        atomic_json_write(record_path, record)
        os.chmod(record_path, 0o600)
        print(
            "FETA_UNET_FINAL_HOLDOUT_CACHE "
            f"fold={fold} subject={index + 1}/{len(subjects)}",
            flush=True,
        )
    del model, saved
    torch.cuda.empty_cache()


def _load_fold_probability(
    cache_root: Path,
    source: CrossValidationMemberSource,
    fold: int,
    subject: FeTASubject,
):
    probability_path, record_path, fold_identity = _fold_cache_paths(
        cache_root, source, fold, subject
    )
    record = ProbabilityCacheRecord.model_validate(_read_json(record_path))
    if record.subject_id != subject.subject_id or record.member_identity != fold_identity:
        raise ValueError("feta_unet_final_holdout_cache_identity_mismatch")
    return load_probability_cache(probability_path, record)


def evaluate_final_holdout(
    manifest_path: Path,
    *,
    data_dir: Path,
    output_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    """Evaluate the frozen V8 DynUNet once on the locked 12-subject holdout."""

    manifest = _read_json(manifest_path)
    source = _validate_release_manifest(manifest)
    manifest_identity = payload_hash(manifest)
    _prepare_release_output(output_root, manifest, manifest_identity)
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(cache_root, 0o700)
    if source.member.dataset_manifest_hash != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_unet_final_holdout_dataset_identity_mismatch")
    if source.member.preprocessing_identity != PREPROCESSING_VERSION:
        raise ValueError("feta_unet_final_holdout_preprocessing_identity_mismatch")
    require_full_baseline_environment()
    subjects = inspect_subjects(data_dir, inspect_labels=False)
    if manifest_hash(subjects) != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_unet_final_holdout_dataset_identity_mismatch")
    partition = locked_partition(
        {item.subject_id: item.reconstruction_method for item in subjects}
    )
    subject_map = {item.subject_id: item for item in subjects}
    holdout_subjects = tuple(subject_map[item] for item in partition.holdout)
    if (
        len(holdout_subjects) != 12
        or {item.subject_id for item in holdout_subjects} != set(partition.holdout)
        or {item.subject_id for item in holdout_subjects} & set(partition.development)
    ):
        raise ValueError("feta_unet_final_holdout_membership_invalid")
    try:
        from monai.data import Dataset
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    dataset = Dataset(
        [
            {"image": item.image_path, "label": item.segmentation_path}
            for item in holdout_subjects
        ],
        transform=create_transforms(training=False),
    )
    for fold in range(5):
        _predict_holdout_fold(
            source,
            fold=fold,
            subjects=holdout_subjects,
            dataset=dataset,
            cache_root=cache_root,
        )
    rows: list[dict[str, Any]] = []
    for index, subject in enumerate(holdout_subjects):
        sample = dataset[index]
        affine = sample["image"].affine.detach().cpu().numpy()
        probabilities = tuple(
            _load_fold_probability(cache_root, source, fold, subject)
            for fold in range(5)
        )
        combined = _average_fold_probabilities(probabilities)
        prediction = restore_prediction_to_native(
            predicted_labels(combined), affine, subject.segmentation_path
        )
        rows.append(_dice_row(subject, _native_label(subject), prediction, fold=-1))
    metrics = _public_summary(rows)
    score = float(metrics["mean_subject_macro_dice"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("feta_unet_final_holdout_score_invalid")
    report = {
        "schema_version": FINAL_HOLDOUT_REPORT_SCHEMA,
        "evaluation_id": manifest["evaluation_id"],
        "release_manifest_identity": manifest_identity,
        "selection_rule": FINAL_HOLDOUT_SELECTION_RULE,
        "candidate_family": "v8-dynunet",
        "source_experiment_id": source.member.experiment_id,
        "configuration_identity": source.member.configuration_identity,
        "architecture_identity": source.member.architecture_identity,
        "checkpoint_sha256s": list(source.checkpoint_sha256s),
        "dataset_manifest_hash": source.member.dataset_manifest_hash,
        "split_hash": source.member.split_hash,
        "fold_hash": source.member.fold_hash,
        "preprocessing_identity": source.member.preprocessing_identity,
        "inference_identity": source.member.inference_identity,
        "test_subject_count": 12,
        "sealed_holdout_evaluations": 1,
        "contains_subject_identifiers": False,
        "post_processing": "none",
        "fold_weights": [0.2] * 5,
        "primary_score": score,
        "primary_score_95_percent_ci": subject_bootstrap_interval(rows),
        "metrics": metrics,
        "result_feedback_prohibited": True,
    }
    atomic_json_write(output_root / "final-holdout-report.json", report)
    atomic_json_write(
        output_root / "protected-subject-metrics.json",
        {
            "schema_version": FINAL_HOLDOUT_REPORT_SCHEMA,
            "contains_subject_identifiers": True,
            "subject_rows": rows,
        },
    )
    for path in output_root.iterdir():
        os.chmod(path, 0o600)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate_final_holdout(
        arguments.manifest,
        data_dir=arguments.data_dir,
        output_root=arguments.output_dir,
        cache_root=arguments.cache_dir,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_final_holdout", "subject_bootstrap_interval"]
