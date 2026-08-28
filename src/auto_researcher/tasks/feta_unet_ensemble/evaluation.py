"""Development-only sequential inference for deterministic FeTA ensembles."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
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
from auto_researcher.tasks.feta_seg.metrics import (
    LABELS,
    aggregate_subject_metrics,
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
    ProbabilityCacheRecord,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    FeTAUNetSearchConfiguration,
)

ENSEMBLE_EVALUATION_SCHEMA_VERSION = "feta-unet-ensemble-evaluation-v1"
LABEL_MAPPING_IDENTITY = "feta-labels-0-through-7-v1"
PRIMARY_SELECTION_RULE = (
    "pre-specified equal-weight probability mean of the verified V4 champion, "
    "V5 champion and two completed V6 finalists"
)
REPRODUCTION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class MemberSource:
    member: EnsembleMember
    checkpoint_path: Path
    configuration: FeTAUNetSearchConfiguration
    source_score: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("feta_unet_ensemble_json_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("feta_unet_ensemble_json_invalid")
    return value


def _normalise_configuration(value: Any) -> FeTAUNetSearchConfiguration:
    """Load historical V4-V6 configurations into the current inference schema."""

    if not isinstance(value, dict):
        raise ValueError("feta_unet_ensemble_configuration_invalid")
    allowed = set(FeTAUNetSearchConfiguration.model_fields)
    payload = {key: item for key, item in value.items() if key in allowed}
    # V4 used augmentation_strength; augmentation is training-only and the
    # current reference_light default has no effect on inference identity.
    payload.setdefault("augmentation_policy", "reference_light")
    try:
        return FeTAUNetSearchConfiguration.model_validate(payload)
    except Exception as exc:
        raise ValueError("feta_unet_ensemble_configuration_invalid") from exc


def _checkpoint_reference_sha(metrics: dict[str, Any]) -> str:
    references = metrics.get("checkpoint_references")
    if not isinstance(references, list):
        raise ValueError("feta_unet_ensemble_checkpoint_reference_invalid")
    matches = [
        item
        for item in references
        if isinstance(item, dict)
        and int(item.get("fold", -1)) == 0
        and str(item.get("relative_path", "")).endswith("fold-0/best.pt")
    ]
    if len(matches) != 1:
        raise ValueError("feta_unet_ensemble_checkpoint_reference_invalid")
    value = matches[0].get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("feta_unet_ensemble_checkpoint_reference_invalid")
    return value


def load_member_source(value: dict[str, Any]) -> MemberSource:
    required = {
        "experiment_id",
        "checkpoint_path",
        "experiment_spec_path",
        "evaluation_result_path",
    }
    if set(value) != required:
        raise ValueError("feta_unet_ensemble_member_source_invalid")
    experiment_id = str(value["experiment_id"])
    checkpoint_path = Path(str(value["checkpoint_path"])).expanduser().resolve()
    specification_path = Path(str(value["experiment_spec_path"])).expanduser().resolve()
    result_path = Path(str(value["evaluation_result_path"])).expanduser().resolve()
    if not all(
        path.is_file() for path in (checkpoint_path, specification_path, result_path)
    ):
        raise ValueError("feta_unet_ensemble_member_source_missing")
    specification = _read_json(specification_path)
    result = _read_json(result_path)
    metrics = result.get("metrics")
    if (
        result.get("success") is not True
        or result.get("experiment_id") != experiment_id
        or specification.get("experiment_id") != experiment_id
        or not isinstance(metrics, dict)
        or int(metrics.get("holdout_subjects_evaluated", -1)) != 0
        or metrics.get("contains_subject_identifiers") is not False
        or int(metrics.get("oof_subject_count", -1)) != 14
    ):
        raise ValueError("feta_unet_ensemble_member_evidence_invalid")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != _checkpoint_reference_sha(metrics):
        raise ValueError("feta_unet_ensemble_checkpoint_identity_mismatch")
    configuration_identity = metrics.get("configuration_identity")
    architecture_identity = metrics.get("architecture_identity")
    identities = {
        "configuration_identity": configuration_identity,
        "dataset_manifest_hash": metrics.get("dataset_manifest_hash"),
        "split_hash": metrics.get("split_hash"),
        "fold_hash": metrics.get("fold_hash"),
        "preprocessing_identity": metrics.get("preprocessing_version"),
        "inference_identity": metrics.get("inference_identity"),
    }
    if any(not isinstance(item, str) or not item for item in identities.values()):
        raise ValueError("feta_unet_ensemble_member_evidence_invalid")
    member = EnsembleMember(
        experiment_id=experiment_id,
        checkpoint_sha256=checkpoint_sha256,
        configuration_identity=str(configuration_identity),
        architecture_identity=str(architecture_identity),
        dataset_manifest_hash=str(identities["dataset_manifest_hash"]),
        split_hash=str(identities["split_hash"]),
        fold_hash=str(identities["fold_hash"]),
        preprocessing_identity=str(identities["preprocessing_identity"]),
        label_mapping_identity=LABEL_MAPPING_IDENTITY,
        inference_identity=str(identities["inference_identity"]),
    )
    source_score = float(result.get("primary_score"))
    if not math.isfinite(source_score) or not 0.0 <= source_score <= 1.0:
        raise ValueError("feta_unet_ensemble_member_score_invalid")
    return MemberSource(
        member=member,
        checkpoint_path=checkpoint_path,
        configuration=_normalise_configuration(metrics.get("configuration")),
        source_score=source_score,
    )


def candidate_subsets(member_ids: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    values = tuple(member_ids)
    if not 2 <= len(values) <= 4 or len(set(values)) != len(values):
        raise ValueError("feta_unet_ensemble_member_count_invalid")
    return tuple(
        combination
        for size in range(2, len(values) + 1)
        for combination in itertools.combinations(values, size)
    )


def select_best_exploratory_ensemble(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Select only a non-primary candidate for exploratory reporting."""

    exploratory = tuple(
        item for item in candidates if item.get("primary_pre_specified") is False
    )
    if not exploratory:
        raise ValueError("feta_unet_ensemble_exploratory_candidate_missing")
    return max(
        exploratory,
        key=lambda item: float(item["metrics"]["mean_subject_macro_dice"]),
    )


def _native_label(subject: FeTASubject):
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


def _dice_row(
    subject: FeTASubject, actual: Any, predicted: Any, *, fold: int = 0
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    reference = np.asarray(actual)
    estimate = np.asarray(predicted)
    if reference.shape != estimate.shape:
        raise ValueError("feta_metric_shape_mismatch")
    scores: dict[str, float] = {}
    for label in LABELS:
        actual_mask = reference == label
        predicted_mask = estimate == label
        actual_count = int(actual_mask.sum())
        predicted_count = int(predicted_mask.sum())
        if actual_count == 0:
            raise ValueError("feta_subject_tissue_absent")
        intersection = int(np.logical_and(actual_mask, predicted_mask).sum())
        scores[str(label)] = (
            0.0
            if predicted_count == 0
            else 2.0 * intersection / (actual_count + predicted_count)
        )
    return {
        "subject_id": subject.subject_id,
        "reconstruction_method": subject.reconstruction_method,
        "fold": fold,
        "dice": scores,
        "macro_dice": sum(scores.values()) / len(scores),
    }


def _public_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_subject_metrics(rows)
    aggregate.pop("subject_metrics", None)
    return aggregate


def _cache_paths(
    cache_root: Path,
    source: MemberSource,
    subject: FeTASubject,
) -> tuple[Path, Path]:
    identity = member_identity(source.member)
    root = cache_root / identity
    return root / f"{subject.subject_id}.npy", root / f"{subject.subject_id}.json"


def _load_record(path: Path) -> ProbabilityCacheRecord:
    return ProbabilityCacheRecord.model_validate(_read_json(path))


def _predict_and_cache(
    source: MemberSource,
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
    saved = torch.load(source.checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(saved, dict)
        or saved.get("configuration_identity") != source.member.configuration_identity
        or saved.get("architecture_identity") != source.member.architecture_identity
        or int(saved.get("fold", -1)) != 0
        or not isinstance(saved.get("model_state_dict"), dict)
    ):
        raise ValueError("feta_unet_ensemble_checkpoint_payload_invalid")
    try:
        model.load_state_dict(saved["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError("feta_unet_ensemble_checkpoint_model_mismatch") from exc
    model = model.to("cuda")
    model.eval()
    identity = member_identity(source.member)
    for index, subject in enumerate(subjects):
        probability_path, record_path = _cache_paths(cache_root, source, subject)
        if probability_path.exists() or record_path.exists():
            if not probability_path.is_file() or not record_path.is_file():
                raise ValueError("feta_unet_probability_cache_partial")
            record = _load_record(record_path)
            if (
                record.subject_id != subject.subject_id
                or record.member_identity != identity
            ):
                raise ValueError("feta_unet_probability_cache_identity_mismatch")
            load_probability_cache(probability_path, record)
            continue
        image = dataset[index]["image"]
        inputs = image.unsqueeze(0).to(device="cuda", non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = sliding_window_predict(inputs, model, source.configuration)
            probabilities = torch.softmax(logits, dim=1)[0]
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("feta_unet_ensemble_probability_non_finite")
        tensor = probabilities.to(device="cpu", dtype=torch.float32).numpy()
        if not bool(np.isfinite(tensor).all()):
            raise ValueError("feta_unet_ensemble_probability_non_finite")
        record = write_probability_cache(
            probability_path,
            tensor,
            subject_id=subject.subject_id,
            member_identity=identity,
        )
        atomic_json_write(record_path, record)
        print(
            "FETA_UNET_ENSEMBLE_CACHE "
            f"experiment={source.member.experiment_id} "
            f"subject={index + 1}/{len(subjects)}",
            flush=True,
        )
    del model, saved
    torch.cuda.empty_cache()


def _load_probability(
    cache_root: Path,
    source: MemberSource,
    subject: FeTASubject,
):
    probability_path, record_path = _cache_paths(cache_root, source, subject)
    record = _load_record(record_path)
    if (
        record.subject_id != subject.subject_id
        or record.member_identity != member_identity(source.member)
    ):
        raise ValueError("feta_unet_probability_cache_identity_mismatch")
    return load_probability_cache(probability_path, record)


def evaluate_manifest(
    manifest_path: Path,
    *,
    data_dir: Path,
    output_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise ValueError("feta_unet_ensemble_output_exists")
    output_root.mkdir(parents=True, mode=0o700)
    os.chmod(output_root, 0o700)
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(cache_root, 0o700)
    manifest = _read_json(manifest_path)
    raw_members = manifest.get("members")
    if (
        manifest.get("schema_version") != "feta-unet-ensemble-run-manifest-v1"
        or not isinstance(raw_members, list)
        or not 2 <= len(raw_members) <= 4
        or set(manifest) != {"schema_version", "ensemble_id", "members"}
    ):
        raise ValueError("feta_unet_ensemble_manifest_invalid")
    sources = tuple(load_member_source(item) for item in raw_members)
    members = validate_compatible_members([item.member for item in sources])
    primary_specification = equal_weight_specification(
        str(manifest["ensemble_id"]),
        members,
        selection_rule=PRIMARY_SELECTION_RULE,
    )
    if members[0].dataset_manifest_hash != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_unet_ensemble_dataset_identity_mismatch")
    if members[0].preprocessing_identity != PREPROCESSING_VERSION:
        raise ValueError("feta_unet_ensemble_preprocessing_identity_mismatch")
    require_full_baseline_environment()
    subjects = inspect_subjects(data_dir, inspect_labels=False)
    if manifest_hash(subjects) != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_unet_ensemble_dataset_identity_mismatch")
    partition = locked_partition(
        {item.subject_id: item.reconstruction_method for item in subjects}
    )
    subject_map = {item.subject_id: item for item in subjects}
    validation_subjects = tuple(
        subject_map[subject_id]
        for subject_id in partition.development
        if partition.folds[subject_id] == 0
    )
    if len(validation_subjects) != 14 or any(
        item.subject_id in partition.holdout for item in validation_subjects
    ):
        raise ValueError("feta_unet_ensemble_fold_membership_invalid")
    try:
        from monai.data import Dataset
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    dataset = Dataset(
        [
            {"image": item.image_path, "label": item.segmentation_path}
            for item in validation_subjects
        ],
        transform=create_transforms(training=False),
    )
    for source in sources:
        _predict_and_cache(source, validation_subjects, dataset, cache_root)

    member_by_id = {item.member.experiment_id: item for item in sources}
    member_ids = tuple(member_by_id)
    combinations = candidate_subsets(member_ids)
    rows: dict[str, list[dict[str, Any]]] = {member_id: [] for member_id in member_ids}
    combo_keys = {items: "+".join(items) for items in combinations}
    rows.update({key: [] for key in combo_keys.values()})
    for index, subject in enumerate(validation_subjects):
        sample = dataset[index]
        affine = sample["image"].affine.detach().cpu().numpy()
        actual = _native_label(subject)
        probabilities = {
            member_id: _load_probability(cache_root, member_by_id[member_id], subject)
            for member_id in member_ids
        }
        for member_id, tensor in probabilities.items():
            prediction = restore_prediction_to_native(
                predicted_labels(tensor), affine, subject.segmentation_path
            )
            rows[member_id].append(_dice_row(subject, actual, prediction))
        for items, key in combo_keys.items():
            weight = 1.0 / len(items)
            combined = aggregate_probabilities(
                [probabilities[item] for item in items],
                [weight for _ in items],
            )
            prediction = restore_prediction_to_native(
                predicted_labels(combined), affine, subject.segmentation_path
            )
            rows[key].append(_dice_row(subject, actual, prediction))
        print(
            f"FETA_UNET_ENSEMBLE_SCORE subject={index + 1}/{len(validation_subjects)}",
            flush=True,
        )

    individual_results = []
    for source in sources:
        summary = _public_summary(rows[source.member.experiment_id])
        reproduced = float(summary["mean_subject_macro_dice"])
        delta = reproduced - source.source_score
        if abs(delta) > REPRODUCTION_TOLERANCE:
            raise ValueError("feta_unet_ensemble_member_score_reproduction_failed")
        individual_results.append(
            {
                "experiment_id": source.member.experiment_id,
                "architecture_identity": source.member.architecture_identity,
                "configuration_identity": source.member.configuration_identity,
                "checkpoint_sha256": source.member.checkpoint_sha256,
                "source_score": source.source_score,
                "reproduced_score": reproduced,
                "reproduction_delta": delta,
                "metrics": summary,
            }
        )
    ensemble_results = []
    for items, key in combo_keys.items():
        summary = _public_summary(rows[key])
        ensemble_results.append(
            {
                "ensemble_id": payload_hash({"members": items, "weights": "equal"}),
                "member_experiment_ids": list(items),
                "weights": [1.0 / len(items) for _ in items],
                "primary_pre_specified": items == member_ids,
                "metrics": summary,
            }
        )
    best_single = max(
        individual_results, key=lambda item: float(item["reproduced_score"])
    )
    primary = next(item for item in ensemble_results if item["primary_pre_specified"])
    best_exploratory = select_best_exploratory_ensemble(ensemble_results)
    report = {
        "schema_version": ENSEMBLE_EVALUATION_SCHEMA_VERSION,
        "ensemble_id": primary_specification.ensemble_id,
        "selection_rule": primary_specification.selection_rule,
        "dataset_manifest_hash": members[0].dataset_manifest_hash,
        "split_hash": members[0].split_hash,
        "fold_hash": members[0].fold_hash,
        "preprocessing_identity": members[0].preprocessing_identity,
        "inference_identity": members[0].inference_identity,
        "development_fold": 0,
        "development_subject_count": len(validation_subjects),
        "reconstruction_method_counts": dict(
            sorted(
                Counter(
                    item.reconstruction_method for item in validation_subjects
                ).items()
            )
        ),
        "sealed_holdout_evaluations": 0,
        "individual_models": individual_results,
        "ensemble_candidates": ensemble_results,
        "best_single_model": best_single,
        "primary_ensemble": {
            **primary,
            "delta_vs_best_single": float(primary["metrics"]["mean_subject_macro_dice"])
            - float(best_single["reproduced_score"]),
        },
        "best_exploratory_ensemble": {
            **best_exploratory,
            "delta_vs_best_single": float(
                best_exploratory["metrics"]["mean_subject_macro_dice"]
            )
            - float(best_single["reproduced_score"]),
        },
    }
    protected = {
        "schema_version": ENSEMBLE_EVALUATION_SCHEMA_VERSION,
        "contains_subject_identifiers": True,
        "subject_rows": rows,
    }
    atomic_json_write(output_root / "ensemble-report.json", report)
    atomic_json_write(output_root / "protected-subject-metrics.json", protected)
    atomic_json_write(
        output_root / "ensemble-specification.json", primary_specification
    )
    for path in output_root.iterdir():
        os.chmod(path, 0o600)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a protected deterministic FeTA probability ensemble."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate_manifest(
        arguments.manifest,
        data_dir=arguments.data_dir,
        output_root=arguments.output_dir,
        cache_root=arguments.cache_dir,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "candidate_subsets",
    "evaluate_manifest",
    "load_member_source",
    "select_best_exploratory_ensemble",
]
