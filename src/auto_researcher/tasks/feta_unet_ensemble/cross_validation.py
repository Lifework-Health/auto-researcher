"""Protected five-fold OOF probability ensemble evaluation for V11."""

from __future__ import annotations

import argparse
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
from auto_researcher.tasks.feta_unet_ensemble.evaluation import (
    LABEL_MAPPING_IDENTITY,
    REPRODUCTION_TOLERANCE,
    _dice_row,
    _native_label,
    _normalise_configuration,
    _public_summary,
    _read_json,
    _sha256,
    candidate_subsets,
    select_best_exploratory_ensemble,
)
from auto_researcher.tasks.feta_unet_ensemble.models import (
    EnsembleMember,
    ProbabilityCacheRecord,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    FeTAUNetSearchConfiguration,
)

CV_ENSEMBLE_SCHEMA_VERSION = "feta-unet-five-fold-ensemble-evaluation-v1"
CV_MANIFEST_SCHEMA_VERSION = "feta-unet-five-fold-ensemble-run-manifest-v1"
CV_PRIMARY_SELECTION_RULE = (
    "pre-specified equal-weight probability mean of the two frozen V11 "
    "confirmation members"
)


def _novel_confirmation_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = tuple(row for row in rows if int(row.get("fold", -1)) in {1, 2, 3, 4})
    if len(values) != 54:
        raise ValueError("feta_unet_cv_ensemble_confirmation_coverage_invalid")
    return _public_summary(values)


@dataclass(frozen=True)
class CrossValidationMemberSource:
    member: EnsembleMember
    checkpoint_paths: tuple[Path, ...]
    checkpoint_sha256s: tuple[str, ...]
    configuration: FeTAUNetSearchConfiguration
    source_score: float


def _checkpoint_references(metrics: dict[str, Any]) -> dict[int, str]:
    references = metrics.get("checkpoint_references")
    if not isinstance(references, list):
        raise TypeError("feta_unet_cv_ensemble_checkpoint_reference_invalid")
    result: dict[int, str] = {}
    for item in references:
        if not isinstance(item, dict):
            continue
        fold = item.get("fold")
        sha = item.get("sha256")
        relative = item.get("relative_path")
        if (
            isinstance(fold, int)
            and fold in range(5)
            and isinstance(sha, str)
            and len(sha) == 64
            and isinstance(relative, str)
            and relative.endswith(f"fold-{fold}/best.pt")
        ):
            if fold in result:
                raise ValueError("feta_unet_cv_ensemble_checkpoint_reference_invalid")
            result[fold] = sha
    if set(result) != set(range(5)):
        raise ValueError("feta_unet_cv_ensemble_checkpoint_reference_invalid")
    return result


def load_cross_validation_member_source(
    value: dict[str, Any],
) -> CrossValidationMemberSource:
    if set(value) != {
        "experiment_id",
        "checkpoint_paths",
        "experiment_spec_path",
        "evaluation_result_path",
    }:
        raise ValueError("feta_unet_cv_ensemble_member_source_invalid")
    experiment_id = str(value["experiment_id"])
    raw_paths = value["checkpoint_paths"]
    if not isinstance(raw_paths, dict) or set(raw_paths) != {str(i) for i in range(5)}:
        raise ValueError("feta_unet_cv_ensemble_member_source_invalid")
    checkpoint_paths = tuple(
        Path(str(raw_paths[str(fold)])).expanduser().resolve() for fold in range(5)
    )
    specification_path = Path(str(value["experiment_spec_path"])).expanduser().resolve()
    result_path = Path(str(value["evaluation_result_path"])).expanduser().resolve()
    if not all(
        path.is_file() for path in (*checkpoint_paths, specification_path, result_path)
    ):
        raise ValueError("feta_unet_cv_ensemble_member_source_missing")
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
        or int(metrics.get("oof_subject_count", -1)) != 68
        or int(metrics.get("folds_completed", -1)) != 5
        or metrics.get("validation_scope") != "five-fold-confirmation-oof"
    ):
        raise ValueError("feta_unet_cv_ensemble_member_evidence_invalid")
    expected = _checkpoint_references(metrics)
    checkpoint_sha256s = tuple(_sha256(path) for path in checkpoint_paths)
    if checkpoint_sha256s != tuple(expected[fold] for fold in range(5)):
        raise ValueError("feta_unet_cv_ensemble_checkpoint_identity_mismatch")
    configuration = _normalise_configuration(metrics.get("configuration"))
    if (
        configuration.profile != "five_fold_confirmation"
        or configuration.fold_count != 5
    ):
        raise ValueError("feta_unet_cv_ensemble_member_evidence_invalid")
    identities = {
        "configuration_identity": metrics.get("configuration_identity"),
        "architecture_identity": metrics.get("architecture_identity"),
        "dataset_manifest_hash": metrics.get("dataset_manifest_hash"),
        "split_hash": metrics.get("split_hash"),
        "fold_hash": metrics.get("fold_hash"),
        "preprocessing_identity": metrics.get("preprocessing_version"),
        "inference_identity": metrics.get("inference_identity"),
    }
    if any(not isinstance(item, str) or not item for item in identities.values()):
        raise ValueError("feta_unet_cv_ensemble_member_evidence_invalid")
    score = float(result.get("primary_score"))
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("feta_unet_cv_ensemble_member_score_invalid")
    member = EnsembleMember(
        experiment_id=experiment_id,
        checkpoint_sha256=payload_hash({"fold_sha256s": checkpoint_sha256s}),
        configuration_identity=str(identities["configuration_identity"]),
        architecture_identity=str(identities["architecture_identity"]),
        dataset_manifest_hash=str(identities["dataset_manifest_hash"]),
        split_hash=str(identities["split_hash"]),
        fold_hash=str(identities["fold_hash"]),
        preprocessing_identity=str(identities["preprocessing_identity"]),
        label_mapping_identity=LABEL_MAPPING_IDENTITY,
        inference_identity=str(identities["inference_identity"]),
    )
    return CrossValidationMemberSource(
        member=member,
        checkpoint_paths=checkpoint_paths,
        checkpoint_sha256s=checkpoint_sha256s,
        configuration=configuration,
        source_score=score,
    )


def _cache_paths(
    cache_root: Path,
    source: CrossValidationMemberSource,
    subject: FeTASubject,
) -> tuple[Path, Path]:
    root = cache_root / member_identity(source.member)
    return root / f"{subject.subject_id}.npy", root / f"{subject.subject_id}.json"


def _load_record(path: Path) -> ProbabilityCacheRecord:
    return ProbabilityCacheRecord.model_validate(_read_json(path))


def _predict_fold(
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
        or saved.get("configuration_identity") != source.member.configuration_identity
        or saved.get("architecture_identity") != source.member.architecture_identity
        or int(saved.get("fold", -1)) != fold
        or not isinstance(saved.get("model_state_dict"), dict)
    ):
        raise ValueError("feta_unet_cv_ensemble_checkpoint_payload_invalid")
    try:
        model.load_state_dict(saved["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError("feta_unet_cv_ensemble_checkpoint_model_mismatch") from exc
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
        inputs = dataset[index]["image"].unsqueeze(0).to("cuda", non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            probabilities = torch.softmax(
                sliding_window_predict(inputs, model, source.configuration), dim=1
            )[0]
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
    del model, saved
    torch.cuda.empty_cache()


def _load_probability(
    cache_root: Path,
    source: CrossValidationMemberSource,
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


def evaluate_cross_validation_manifest(
    manifest_path: Path,
    *,
    data_dir: Path,
    output_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise ValueError("feta_unet_cv_ensemble_output_exists")
    output_root.mkdir(parents=True, mode=0o700)
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = _read_json(manifest_path)
    raw_members = manifest.get("members")
    if (
        manifest.get("schema_version") != CV_MANIFEST_SCHEMA_VERSION
        or not isinstance(raw_members, list)
        or len(raw_members) != 4
        or set(manifest) != {"schema_version", "ensemble_id", "members"}
    ):
        raise ValueError("feta_unet_cv_ensemble_manifest_invalid")
    sources = tuple(load_cross_validation_member_source(item) for item in raw_members)
    members = validate_compatible_members([source.member for source in sources])
    specification = equal_weight_specification(
        str(manifest["ensemble_id"]),
        members,
        selection_rule=CV_PRIMARY_SELECTION_RULE,
    )
    if members[0].dataset_manifest_hash != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_unet_cv_ensemble_dataset_identity_mismatch")
    if members[0].preprocessing_identity != PREPROCESSING_VERSION:
        raise ValueError("feta_unet_cv_ensemble_preprocessing_identity_mismatch")
    require_full_baseline_environment()
    subjects = inspect_subjects(data_dir, inspect_labels=False)
    if manifest_hash(subjects) != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_unet_cv_ensemble_dataset_identity_mismatch")
    partition = locked_partition(
        {item.subject_id: item.reconstruction_method for item in subjects}
    )
    subject_map = {item.subject_id: item for item in subjects}
    fold_subjects = {
        fold: tuple(
            subject_map[identifier]
            for identifier in partition.development
            if partition.folds[identifier] == fold
        )
        for fold in range(5)
    }
    observed = {item.subject_id for values in fold_subjects.values() for item in values}
    if observed != set(partition.development) or observed & set(partition.holdout):
        raise ValueError("feta_unet_cv_ensemble_fold_membership_invalid")
    try:
        from monai.data import Dataset
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    datasets = {
        fold: Dataset(
            [
                {"image": item.image_path, "label": item.segmentation_path}
                for item in values
            ],
            transform=create_transforms(training=False),
        )
        for fold, values in fold_subjects.items()
    }
    for source in sources:
        for fold in range(5):
            _predict_fold(
                source,
                fold=fold,
                subjects=fold_subjects[fold],
                dataset=datasets[fold],
                cache_root=cache_root,
            )

    member_by_id = {item.member.experiment_id: item for item in sources}
    member_ids = tuple(member_by_id)
    combinations = candidate_subsets(member_ids)
    keys = {items: "+".join(items) for items in combinations}
    rows: dict[str, list[dict[str, Any]]] = {
        **{identifier: [] for identifier in member_ids},
        **{key: [] for key in keys.values()},
    }
    for fold in range(5):
        dataset = datasets[fold]
        for index, subject in enumerate(fold_subjects[fold]):
            sample = dataset[index]
            affine = sample["image"].affine.detach().cpu().numpy()
            actual = _native_label(subject)
            probabilities = {
                identifier: _load_probability(
                    cache_root, member_by_id[identifier], subject
                )
                for identifier in member_ids
            }
            for identifier, tensor in probabilities.items():
                prediction = restore_prediction_to_native(
                    predicted_labels(tensor), affine, subject.segmentation_path
                )
                rows[identifier].append(
                    _dice_row(subject, actual, prediction, fold=fold)
                )
            for items, key in keys.items():
                weight = 1.0 / len(items)
                combined = aggregate_probabilities(
                    [probabilities[item] for item in items],
                    [weight] * len(items),
                )
                prediction = restore_prediction_to_native(
                    predicted_labels(combined), affine, subject.segmentation_path
                )
                rows[key].append(_dice_row(subject, actual, prediction, fold=fold))

    individual = []
    for source in sources:
        summary = _public_summary(rows[source.member.experiment_id])
        reproduced = float(summary["mean_subject_macro_dice"])
        delta = reproduced - source.source_score
        if abs(delta) > REPRODUCTION_TOLERANCE:
            raise ValueError("feta_unet_cv_ensemble_member_score_reproduction_failed")
        individual.append(
            {
                "experiment_id": source.member.experiment_id,
                "source_score": source.source_score,
                "reproduced_score": reproduced,
                "reproduction_delta": delta,
                "checkpoint_sha256s": list(source.checkpoint_sha256s),
                "metrics": summary,
                "novel_confirmation_metrics": _novel_confirmation_summary(
                    rows[source.member.experiment_id]
                ),
            }
        )
    ensembles = []
    for items, key in keys.items():
        summary = _public_summary(rows[key])
        ensembles.append(
            {
                "ensemble_id": payload_hash({"members": items, "weights": "equal"}),
                "member_experiment_ids": list(items),
                "weights": [1.0 / len(items)] * len(items),
                "primary_pre_specified": items == member_ids,
                "metrics": summary,
                "novel_confirmation_metrics": _novel_confirmation_summary(rows[key]),
            }
        )
    best_single = max(individual, key=lambda item: item["reproduced_score"])
    primary = next(item for item in ensembles if item["primary_pre_specified"])
    exploratory = select_best_exploratory_ensemble(ensembles)
    report = {
        "schema_version": CV_ENSEMBLE_SCHEMA_VERSION,
        "ensemble_id": specification.ensemble_id,
        "selection_rule": specification.selection_rule,
        "development_folds": list(range(5)),
        "selection_fold": 0,
        "novel_confirmation_folds": [1, 2, 3, 4],
        "development_subject_count": 68,
        "fold_subject_counts": {
            str(fold): len(values) for fold, values in fold_subjects.items()
        },
        "reconstruction_method_counts": dict(
            sorted(
                Counter(
                    item.reconstruction_method
                    for item in subject_map.values()
                    if item.subject_id in observed
                ).items()
            )
        ),
        "sealed_holdout_evaluations": 0,
        "individual_models": individual,
        "ensemble_candidates": ensembles,
        "best_single_model": best_single,
        "primary_ensemble": {
            **primary,
            "delta_vs_best_single": float(primary["metrics"]["mean_subject_macro_dice"])
            - float(best_single["reproduced_score"]),
        },
        "best_exploratory_ensemble": {
            **exploratory,
            "delta_vs_best_single": float(
                exploratory["metrics"]["mean_subject_macro_dice"]
            )
            - float(best_single["reproduced_score"]),
        },
    }
    atomic_json_write(output_root / "ensemble-report.json", report)
    atomic_json_write(
        output_root / "protected-subject-metrics.json",
        {
            "schema_version": CV_ENSEMBLE_SCHEMA_VERSION,
            "contains_subject_identifiers": True,
            "subject_rows": rows,
        },
    )
    atomic_json_write(output_root / "ensemble-specification.json", specification)
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
    report = evaluate_cross_validation_manifest(
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
    "evaluate_cross_validation_manifest",
    "load_cross_validation_member_source",
]
