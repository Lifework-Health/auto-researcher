"""Identity-bound completed-fold restart for the frozen BasicUNet task."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.artifacts import atomic_json_write
from auto_researcher.tasks.feta_seg.manifests import (
    EXPECTED_MANIFEST_HASH,
    FeTASubject,
)
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
)
from auto_researcher.tasks.feta_unet_direct.configuration import (
    FeTAUNetDirectConfiguration,
)
from auto_researcher.tasks.feta_unet_direct.identities import (
    DATA_LOADER_ID,
    runner_id,
)
from auto_researcher.tasks.feta_unet_direct.model import ARCHITECTURE_ID
from auto_researcher.tasks.feta_unet_direct.trainer import checkpoint_reference

FOLD_RESULT_SCHEMA_VERSION = "feta-unet-direct-fold-result-v1"
FOLD_SCIENTIFIC_IDENTITY_VERSION = "feta-unet-direct-fold-science-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(
    configuration: FeTAUNetDirectConfiguration,
    fold: int,
    validation_subjects: tuple[FeTASubject, ...],
) -> dict[str, Any]:
    return {
        "scientific_identity_version": FOLD_SCIENTIFIC_IDENTITY_VERSION,
        "architecture_identity": ARCHITECTURE_ID,
        "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
        "split_hash": EXPECTED_SPLIT_HASH,
        "fold_hash": EXPECTED_FOLD_HASH,
        "configuration_hash": payload_hash(configuration),
        "profile": configuration.profile,
        "runner_id": runner_id(configuration.profile),
        "data_loader_id": DATA_LOADER_ID,
        "fold": fold,
        "seed": configuration.seed + fold,
        "validation_subject_ids": sorted(
            subject.subject_id for subject in validation_subjects
        ),
    }


def persist_fold_result(
    experiment_root: Path,
    result: Any,
    configuration: FeTAUNetDirectConfiguration,
    validation_subjects: tuple[FeTASubject, ...],
) -> Path:
    path = experiment_root / "fold-results" / f"fold-{result.fold}.json"
    atomic_json_write(
        path,
        {
            "schema_version": FOLD_RESULT_SCHEMA_VERSION,
            "identity": _identity(configuration, result.fold, validation_subjects),
            "result": asdict(result),
        },
    )
    return path


def load_fold_result(
    source_root: Path,
    target_root: Path,
    result_type: type,
    configuration: FeTAUNetDirectConfiguration,
    fold: int,
    validation_subjects: tuple[FeTASubject, ...],
):
    path = source_root / "fold-results" / f"fold-{fold}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FOLD_RESULT_SCHEMA_VERSION:
        raise ValueError("feta_unet_fold_restart_schema_mismatch")
    if payload.get("identity") != _identity(configuration, fold, validation_subjects):
        raise ValueError("feta_unet_fold_restart_identity_mismatch")

    result_payload = payload.get("result")
    if not isinstance(result_payload, dict):
        raise ValueError("feta_unet_fold_restart_payload_invalid")
    subject_metrics = result_payload.get("subject_metrics")
    if not isinstance(subject_metrics, list):
        raise ValueError("feta_unet_fold_restart_payload_invalid")
    expected_subjects = {subject.subject_id for subject in validation_subjects}
    observed_subjects = {
        str(row.get("subject_id")) for row in subject_metrics if isinstance(row, dict)
    }
    if observed_subjects != expected_subjects:
        raise ValueError("feta_unet_fold_restart_oof_membership_invalid")
    if (
        int(result_payload.get("fold", -1)) != fold
        or int(result_payload.get("seed", -1)) != configuration.seed + fold
        or result_payload.get("source_runner_id") != runner_id(configuration.profile)
        or result_payload.get("source_data_loader_id") != DATA_LOADER_ID
    ):
        raise ValueError("feta_unet_fold_restart_identity_mismatch")

    checkpoint = result_payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("feta_unet_fold_restart_checkpoint_invalid")
    relative_path = checkpoint.get("relative_path")
    expected_sha = checkpoint.get("sha256")
    expected_size = checkpoint.get("size_bytes")
    if (
        not isinstance(relative_path, str)
        or relative_path != f"fold-{fold}/best.pt"
        or not isinstance(expected_sha, str)
        or not isinstance(expected_size, int)
    ):
        raise ValueError("feta_unet_fold_restart_checkpoint_invalid")

    source_checkpoint = source_root / "checkpoints" / relative_path
    if (
        not source_checkpoint.is_file()
        or source_checkpoint.stat().st_size != expected_size
        or _sha256(source_checkpoint) != expected_sha
    ):
        raise ValueError("feta_unet_fold_restart_checkpoint_identity_mismatch")
    target_checkpoint = target_root / "checkpoints" / relative_path
    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if source_checkpoint.resolve() != target_checkpoint.resolve():
        shutil.copy2(source_checkpoint, target_checkpoint)
    result_payload["checkpoint"] = checkpoint_reference(
        target_checkpoint,
        fold=fold,
        best_epoch=int(result_payload["best_epoch"]),
        score=float(result_payload["validation_score"]),
        output_root=target_root / "checkpoints",
    )
    result_payload["subject_metrics"] = tuple(subject_metrics)
    result_payload["reused_fold_result"] = True
    return result_type(**result_payload)
