"""Identity-bound loading of protected FeTA fold evidence for diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_researcher.diagnostics.models import DiagnosticCheckpoint
from auto_researcher.tasks.feta_unet_direct.fold_resume import (
    FOLD_RESULT_SCHEMA_VERSION,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProtectedExperimentEvidence:
    """Protected evidence that must not be copied into shareable model context."""

    checkpoint: DiagnosticCheckpoint
    dataset_manifest_hash: str
    split_hash: str
    fold_hash: str
    subject_metrics: tuple[dict[str, Any], ...]
    validation_history: tuple[dict[str, Any], ...]


def load_protected_experiment_evidence(
    experiment_root: Path,
) -> ProtectedExperimentEvidence:
    root = experiment_root.resolve()
    fold_paths = sorted((root / "fold-results").glob("fold-*.json"))
    if not fold_paths:
        raise ValueError("feta_diagnostic_fold_results_missing")

    dataset_hashes: set[str] = set()
    split_hashes: set[str] = set()
    fold_hashes: set[str] = set()
    configuration_hashes: set[str] = set()
    architecture_identities: set[str] = set()
    checkpoint_hashes: list[str] = []
    best_epochs: list[int] = []
    subject_metrics: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    observed_subjects: set[str] = set()

    for path in fold_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != FOLD_RESULT_SCHEMA_VERSION:
            raise ValueError("feta_diagnostic_fold_result_schema_invalid")
        identity = payload.get("identity")
        result = payload.get("result")
        if not isinstance(identity, dict) or not isinstance(result, dict):
            raise ValueError("feta_diagnostic_fold_result_invalid")
        fold = identity.get("fold")
        if not isinstance(fold, int) or path.name != f"fold-{fold}.json":
            raise ValueError("feta_diagnostic_fold_identity_invalid")
        dataset_hashes.add(str(identity.get("dataset_manifest_hash")))
        split_hashes.add(str(identity.get("split_hash")))
        fold_hashes.add(str(identity.get("fold_hash")))
        configuration_hashes.add(str(identity.get("configuration_hash")))
        architecture = result.get("architecture_identity")
        rows = result.get("subject_metrics")
        history = result.get("validation_history")
        checkpoint = result.get("checkpoint")
        best_epoch = result.get("best_epoch")
        if (
            not isinstance(architecture, str)
            or not architecture
            or not isinstance(rows, list)
            or not isinstance(history, list)
            or not isinstance(checkpoint, dict)
            or not isinstance(best_epoch, int)
            or best_epoch < 1
        ):
            raise ValueError("feta_diagnostic_fold_result_invalid")
        architecture_identities.add(architecture)
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("feta_diagnostic_subject_metrics_invalid")
            subject_id = row.get("subject_id")
            if (
                not isinstance(subject_id, str)
                or not subject_id
                or subject_id in observed_subjects
            ):
                raise ValueError("feta_diagnostic_subject_metrics_invalid")
            observed_subjects.add(subject_id)
            subject_metrics.append(row)
        if any(not isinstance(item, dict) for item in history):
            raise ValueError("feta_diagnostic_learning_curve_invalid")
        validation_history.extend(history)

        relative_path = checkpoint.get("relative_path")
        expected_hash = checkpoint.get("sha256")
        expected_size = checkpoint.get("size_bytes")
        if (
            not isinstance(relative_path, str)
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not isinstance(expected_size, int)
            or expected_size < 1
        ):
            raise ValueError("feta_diagnostic_checkpoint_reference_invalid")
        checkpoint_path = (root / "checkpoints" / relative_path).resolve()
        try:
            checkpoint_path.relative_to((root / "checkpoints").resolve())
        except ValueError as exc:
            raise ValueError("feta_diagnostic_checkpoint_reference_invalid") from exc
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size != expected_size
            or _sha256(checkpoint_path) != expected_hash
        ):
            raise ValueError("feta_diagnostic_checkpoint_identity_mismatch")
        checkpoint_hashes.append(expected_hash)
        best_epochs.append(best_epoch)

    singleton_sets = (
        dataset_hashes,
        split_hashes,
        fold_hashes,
        configuration_hashes,
        architecture_identities,
    )
    if any(len(values) != 1 or "None" in values for values in singleton_sets):
        raise ValueError("feta_diagnostic_cross_fold_identity_mismatch")
    return ProtectedExperimentEvidence(
        checkpoint=DiagnosticCheckpoint(
            experiment_id=root.name,
            checkpoint_sha256s=tuple(checkpoint_hashes),
            architecture_identity=next(iter(architecture_identities)),
            configuration_identity=next(iter(configuration_hashes)),
            best_epochs=tuple(best_epochs),
        ),
        dataset_manifest_hash=next(iter(dataset_hashes)),
        split_hash=next(iter(split_hashes)),
        fold_hash=next(iter(fold_hashes)),
        subject_metrics=tuple(subject_metrics),
        validation_history=tuple(validation_history),
    )
