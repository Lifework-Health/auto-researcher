"""Safe iCCA dataset fingerprints containing no patient-level content."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from auto_researcher.tasks.models import DatasetManifest, TaskRuntimeContext

ICCA_DATA_FILES = ("Combined_binary_matrix.csv", "Combined_clinical.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_icca_dataset_manifest(
    context: TaskRuntimeContext,
    *,
    loader_version: str,
) -> DatasetManifest:
    if context.data_dir is None:
        raise FileNotFoundError("iCCA runtime requires a data_dir")
    paths = [context.data_dir / filename for filename in ICCA_DATA_FILES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"iCCA data_dir is missing required files: {', '.join(missing)}"
        )
    hashes = {path.name: _sha256(path) for path in paths}
    combined = hashlib.sha256(
        "\n".join(f"{name}:{hashes[name]}" for name in sorted(hashes)).encode()
    ).hexdigest()
    created_at = context.manifest_created_at or datetime.now(UTC)
    return DatasetManifest(
        task_id="icca_nbs",
        dataset_version=f"icca-nbs:{combined[:12]}",
        files=tuple(path.name for path in paths),
        hashes=hashes,
        loader_version=loader_version,
        created_at=created_at,
        metadata={
            "file_sizes": {path.name: path.stat().st_size for path in paths},
            "combined_dataset_fingerprint": combined,
            "objective_version": str(
                context.task_options.get("objective_version", "unspecified")
            ),
            "contains_patient_identifiers": False,
        },
    )
