"""Bind verified V8 parent artefacts into a fresh V9 runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "feta-unet-v9-parent-checkpoints-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("feta_unet_v9_parent_manifest_invalid") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("development_fold") != 0
        or raw.get("sealed_holdout_evaluations") != 0
    ):
        raise ValueError("feta_unet_v9_parent_manifest_invalid")
    parents = raw.get("parents")
    if not isinstance(parents, list) or len(parents) != 2:
        raise ValueError("feta_unet_v9_parent_manifest_invalid")
    return raw


def bind_v9_parent_checkpoints(
    *,
    manifest_path: Path,
    source_workspace: Path,
    source_result_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Verify and hard-link/copy the exact fold-0 parent artefacts once."""

    manifest = _load_manifest(manifest_path)
    if destination.exists():
        raise ValueError("feta_unet_v9_parent_destination_not_fresh")
    staged = destination.with_name(f".{destination.name}.staged")
    if staged.exists():
        raise ValueError("feta_unet_v9_parent_destination_not_fresh")
    verified: list[dict[str, Any]] = []
    staged.mkdir(parents=True, mode=0o700)
    try:
        for parent in manifest["parents"]:
            if not isinstance(parent, dict):
                raise ValueError("feta_unet_v9_parent_manifest_invalid")
            experiment_id = parent.get("experiment_id")
            files = parent.get("files")
            if not isinstance(experiment_id, str) or not isinstance(files, dict):
                raise ValueError("feta_unet_v9_parent_manifest_invalid")
            parent_destination = staged / experiment_id
            for relative_name, expected_hash in files.items():
                if not isinstance(relative_name, str) or not isinstance(
                    expected_hash, str
                ):
                    raise ValueError("feta_unet_v9_parent_manifest_invalid")
                base = (
                    source_result_root
                    if relative_name in {"experiment_spec.json", "evaluation_result.json"}
                    else source_workspace
                )
                source = base / experiment_id / relative_name
                if not source.is_file() or _sha256(source) != expected_hash:
                    raise ValueError("feta_unet_v9_parent_source_identity_invalid")
                target = parent_destination / relative_name
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
                if _sha256(target) != expected_hash:
                    raise ValueError("feta_unet_v9_parent_copy_identity_invalid")
            verified.append(
                {
                    "role": parent.get("role"),
                    "experiment_id": experiment_id,
                    "file_count": len(files),
                }
            )
        staged.replace(destination)
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return {
        "schema_version": SCHEMA_VERSION,
        "source_run_id": manifest["source_run_id"],
        "parents": verified,
        "development_fold": 0,
        "sealed_holdout_evaluations": 0,
    }
