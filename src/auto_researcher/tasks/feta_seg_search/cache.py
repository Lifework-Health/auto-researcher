"""Shared deterministic preprocessing cache identity and preparation."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fcntl

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
from auto_researcher.tasks.feta_seg_search.configuration import (
    FeTASegSearchConfiguration,
)
from auto_researcher.tasks.feta_seg_search.transforms import PREPROCESSING_VERSION

CACHE_IDENTITY_VERSION = "feta-search-shared-deterministic-cache-v1"
MONAI_CACHE_SCHEMA_VERSION = "monai-persistent-dataset-1.5.1-v1"
CACHE_RECORD_HASH_VERSION = "feta-path-free-record-sha256-v1"
CACHE_MANIFEST_FILENAME = "cache_identity.json"
CACHE_COMPLETION_FILENAME = "cache_complete.json"
CACHE_LOCK_VERSION = "feta-search-shared-cache-flock-v1"
CACHE_LOCK_DIRECTORY = "_locks"


@dataclass(frozen=True)
class SharedCachePreparation:
    identity: str
    identity_version: str
    root: Path
    training_cache_dir: Path
    prepare_seconds: float


def cache_record_hash(record: Any) -> bytes:
    if not isinstance(record, dict):
        raise ValueError("feta_search_shared_cache_record_invalid")
    try:
        canonical = {
            "hash_version": CACHE_RECORD_HASH_VERSION,
            "image": Path(record["image"]).name,
            "label": Path(record["label"]).name,
            "subject_id": str(record["subject_id"]),
            "reconstruction_method": str(record["reconstruction_method"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("feta_search_shared_cache_record_invalid") from exc
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().encode("utf-8")


def shared_cache_is_complete(
    preparation: SharedCachePreparation, *, expected_items: int
) -> bool:
    marker = preparation.root / CACHE_COMPLETION_FILENAME
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("feta_search_shared_cache_completion_invalid") from exc
    expected = {
        "cache_identity": preparation.identity,
        "cache_identity_version": preparation.identity_version,
        "cached_training_items": expected_items,
    }
    if payload != expected:
        raise ValueError("feta_search_shared_cache_completion_mismatch")
    if len(tuple(preparation.training_cache_dir.glob("*.pt"))) != expected_items:
        raise ValueError("feta_search_shared_cache_population_mismatch")
    return True


def mark_shared_cache_complete(
    preparation: SharedCachePreparation, *, expected_items: int
) -> None:
    if len(tuple(preparation.training_cache_dir.glob("*.pt"))) != expected_items:
        raise ValueError("feta_search_shared_cache_population_incomplete")
    atomic_json_write(
        preparation.root / CACHE_COMPLETION_FILENAME,
        {
            "cache_identity": preparation.identity,
            "cache_identity_version": preparation.identity_version,
            "cached_training_items": expected_items,
        },
    )


def deterministic_cache_payload(
    configuration: FeTASegSearchConfiguration,
    training_subjects: Sequence[FeTASubject],
    *,
    preprocessing_version: str = PREPROCESSING_VERSION,
) -> dict[str, Any]:
    """Return the path-free identity for the cacheable transform prefix only."""

    return {
        "cache_identity_version": CACHE_IDENTITY_VERSION,
        "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
        "split_hash": EXPECTED_SPLIT_HASH,
        "fold_hash": EXPECTED_FOLD_HASH,
        "fold": configuration.fold,
        "training_subject_ids": sorted(
            subject.subject_id for subject in training_subjects
        ),
        "preprocessing_version": preprocessing_version,
        "spacing_mm": list(configuration.spacing_mm),
        "orientation": "RAS",
        "normalisation": "nonzero-per-volume-channel-zscore",
        "monai_cache_schema_version": MONAI_CACHE_SCHEMA_VERSION,
        "cache_record_hash_version": CACHE_RECORD_HASH_VERSION,
    }


def deterministic_cache_identity(
    configuration: FeTASegSearchConfiguration,
    training_subjects: Sequence[FeTASubject],
    *,
    preprocessing_version: str = PREPROCESSING_VERSION,
) -> str:
    return payload_hash(
        deterministic_cache_payload(
            configuration,
            training_subjects,
            preprocessing_version=preprocessing_version,
        )
    )


def shared_cache_root(workspace_dir: Path, identity: str) -> Path:
    return workspace_dir / "feta_seg_search" / "_shared_cache" / identity


def shared_cache_lock_path(workspace_dir: Path, identity: str) -> Path:
    return (
        workspace_dir
        / "feta_seg_search"
        / "_shared_cache"
        / CACHE_LOCK_DIRECTORY
        / f"{identity}.lock"
    )


@contextmanager
def shared_cache_advisory_lock(
    workspace_dir: Path, identity: str
) -> Iterator[Path]:
    """Serialise population with a crash-safe OS advisory lock outside git."""

    lock_path = shared_cache_lock_path(workspace_dir, identity)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield lock_path
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise RuntimeError("feta_search_shared_cache_lock_failed") from exc


def prepare_shared_cache(
    workspace_dir: Path,
    configuration: FeTASegSearchConfiguration,
    training_subjects: Sequence[FeTASubject],
) -> SharedCachePreparation:
    """Validate or initialise one cache namespace without storing paths in identity."""

    started = time.perf_counter()
    payload = deterministic_cache_payload(configuration, training_subjects)
    identity = payload_hash(payload)
    root = shared_cache_root(workspace_dir, identity)
    manifest_path = root / CACHE_MANIFEST_FILENAME
    existed = root.exists()
    root.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        try:
            observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("feta_search_shared_cache_manifest_invalid") from exc
        if observed != payload or payload_hash(observed) != identity:
            raise ValueError("feta_search_shared_cache_identity_mismatch")
    else:
        if existed and any(root.iterdir()):
            raise ValueError("feta_search_shared_cache_manifest_missing")
        atomic_json_write(manifest_path, payload)
    training_cache_dir = root / "training"
    training_cache_dir.mkdir(exist_ok=True)
    return SharedCachePreparation(
        identity=identity,
        identity_version=CACHE_IDENTITY_VERSION,
        root=root,
        training_cache_dir=training_cache_dir,
        prepare_seconds=time.perf_counter() - started,
    )


def prepare_or_reuse_shared_cache(
    workspace_dir: Path,
    configuration: FeTASegSearchConfiguration,
    training_subjects: Sequence[FeTASubject],
    *,
    populate: Callable[[SharedCachePreparation], None],
) -> tuple[SharedCachePreparation, bool]:
    """Populate once under ``flock`` and re-check completion after lock acquisition."""

    identity = deterministic_cache_identity(configuration, training_subjects)
    with shared_cache_advisory_lock(workspace_dir, identity):
        preparation = prepare_shared_cache(
            workspace_dir, configuration, training_subjects
        )
        expected_items = len(training_subjects)
        reused = shared_cache_is_complete(
            preparation, expected_items=expected_items
        )
        if reused:
            return preparation, True
        if any(preparation.training_cache_dir.iterdir()):
            raise ValueError("feta_search_shared_cache_population_partial")
        populate(preparation)
        mark_shared_cache_complete(preparation, expected_items=expected_items)
        return preparation, False
