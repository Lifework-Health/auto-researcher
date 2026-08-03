"""Strict JSON and transactional task artefact bundle helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from auto_researcher.contracts.models import EvaluationResult, ExperimentSpec
from auto_researcher.tasks.models import DatasetManifest, TaskRuntimeContext
from auto_researcher.tasks.scientific_json import SCIENTIFIC_JSON_ENCODING_VERSION

ARTEFACT_BUNDLE_SCHEMA_VERSION = "experiment-bundle-v2"
ARTEFACT_BUNDLE_METADATA_KEY = "artefact_bundle"
ARTEFACT_FILENAMES = (
    "experiment_spec.json",
    "evaluation_result.json",
    "dataset_manifest.json",
    "evaluator_manifest.json",
)

FaultInjector = Callable[[str], None]


class ArtefactBundleError(RuntimeError):
    """Base error for a bundle that was not safely published."""


class ArtefactBundleConflictError(ArtefactBundleError):
    """A completed bundle exists at the identity with different content."""


@dataclass(frozen=True)
class ArtefactBundleReceipt:
    references: tuple[str, ...]
    payload_sha256: dict[str, str]
    bundle_sha256: str
    replayed: bool


@dataclass(frozen=True)
class ArtefactBundleIntegrity:
    complete: bool
    untampered: bool
    payload_sha256: dict[str, str]
    bundle_sha256: str | None
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArtefactBundleIdentity:
    """Verified identity of one completely published experiment bundle."""

    bundle_sha256: str
    schema_version: str
    result_encoding_version: str
    references: tuple[str, ...]
    evaluator_manifest_payload_hash: str


def json_safe(value: Any) -> Any:
    """Convert scientific scalar/container types without importing domain packages."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [json_safe(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if hasattr(value, "tolist") and callable(value.tolist):
        return json_safe(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return json_safe(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported scientific result type: {type(value).__name__}")


def strict_json_bytes(value: Any) -> bytes:
    """Fully validate and serialize one standards-compliant JSON payload."""

    payload = json.dumps(
        json_safe(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return f"{payload}\n".encode("utf-8")


def safe_segment(value: str, field: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or any(character in value for character in ("/", "\\", "\0"))
    ):
        raise ValueError(f"{field} must be a non-empty path-safe segment")
    return value


def artefact_references(
    context: TaskRuntimeContext,
    experiment_id: str,
) -> tuple[str, ...]:
    if context.output_dir is None or not context.run_id:
        return ()
    run_id = safe_segment(context.run_id, "run_id")
    safe_experiment_id = safe_segment(experiment_id, "experiment_id")
    prefix = Path("runs") / run_id / safe_experiment_id
    return tuple((prefix / name).as_posix() for name in ARTEFACT_FILENAMES)


def atomic_json_write(path: Path, value: Any) -> None:
    """Write a standalone strict-JSON artefact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = strict_json_bytes(value)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bundle_hash(payload_hashes: dict[str, str]) -> str:
    canonical = json.dumps(
        payload_hashes,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(canonical)


def _serialise_bundle(
    experiment: ExperimentSpec,
    evaluation: EvaluationResult,
    dataset_manifest: DatasetManifest,
    evaluator_manifest: dict[str, Any],
) -> tuple[dict[str, bytes], dict[str, str], str]:
    if ARTEFACT_BUNDLE_METADATA_KEY in evaluator_manifest:
        raise ValueError(
            f"{ARTEFACT_BUNDLE_METADATA_KEY!r} is reserved for bundle integrity"
        )
    base_values = {
        "experiment_spec.json": experiment,
        "evaluation_result.json": evaluation,
        "dataset_manifest.json": dataset_manifest,
        "evaluator_manifest.json": evaluator_manifest,
    }
    # Validate and serialize all four values before any filesystem write occurs.
    base_payloads = {
        name: strict_json_bytes(value) for name, value in base_values.items()
    }
    payload_hashes = {name: _sha256(payload) for name, payload in base_payloads.items()}
    bundle_sha256 = _bundle_hash(payload_hashes)
    enriched_manifest = {
        **evaluator_manifest,
        ARTEFACT_BUNDLE_METADATA_KEY: {
            "schema_version": ARTEFACT_BUNDLE_SCHEMA_VERSION,
            "result_encoding_version": SCIENTIFIC_JSON_ENCODING_VERSION,
            "expected_filenames": list(ARTEFACT_FILENAMES),
            "payload_sha256": payload_hashes,
            "bundle_sha256": bundle_sha256,
            "completed": True,
        },
    }
    published_payloads = {
        **base_payloads,
        "evaluator_manifest.json": strict_json_bytes(enriched_manifest),
    }
    return published_payloads, payload_hashes, bundle_sha256


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _published_payloads_match(
    directory: Path,
    payloads: dict[str, bytes],
) -> bool:
    if not directory.is_dir():
        return False
    if {item.name for item in directory.iterdir()} != set(ARTEFACT_FILENAMES):
        return False
    return all(
        (directory / name).read_bytes() == payload for name, payload in payloads.items()
    )


def write_artefact_bundle(
    context: TaskRuntimeContext,
    experiment: ExperimentSpec,
    evaluation: EvaluationResult,
    dataset_manifest: DatasetManifest,
    evaluator_manifest: dict[str, Any],
    *,
    fault_injector: FaultInjector | None = None,
) -> ArtefactBundleReceipt:
    """Publish the complete four-file experiment bundle as one directory rename."""

    references = artefact_references(context, experiment.experiment_id)
    payloads, payload_hashes, bundle_sha256 = _serialise_bundle(
        experiment,
        evaluation,
        dataset_manifest,
        evaluator_manifest,
    )
    if not references:
        return ArtefactBundleReceipt((), payload_hashes, bundle_sha256, False)

    assert context.output_dir is not None
    run_id = safe_segment(context.run_id or "", "run_id")
    experiment_id = safe_segment(experiment.experiment_id, "experiment_id")
    parent = context.output_dir / "runs" / run_id
    final_directory = parent / experiment_id
    parent.mkdir(parents=True, exist_ok=True)

    if final_directory.exists():
        if _published_payloads_match(final_directory, payloads):
            return ArtefactBundleReceipt(
                references, payload_hashes, bundle_sha256, True
            )
        raise ArtefactBundleConflictError(
            "a conflicting artefact bundle already exists for this experiment"
        )

    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{experiment_id}.", suffix=".tmp", dir=parent)
    )
    try:
        if fault_injector is not None:
            fault_injector("before_any_file_write")
        for name in ARTEFACT_FILENAMES:
            payload = payloads[name]
            destination = temporary_directory / name
            with destination.open("xb") as handle:
                midpoint = max(1, len(payload) // 2)
                handle.write(payload[:midpoint])
                if fault_injector is not None:
                    fault_injector(f"during_write:{name}")
                handle.write(payload[midpoint:])
                handle.flush()
                os.fsync(handle.fileno())
        if {item.name for item in temporary_directory.iterdir()} != set(
            ARTEFACT_FILENAMES
        ):
            raise ArtefactBundleError("temporary artefact bundle is incomplete")
        _fsync_directory(temporary_directory)
        if fault_injector is not None:
            fault_injector("before_directory_publication")
        os.replace(temporary_directory, final_directory)
        _fsync_directory(parent)
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)

    return ArtefactBundleReceipt(references, payload_hashes, bundle_sha256, False)


def _reject_non_standard_json(token: str) -> None:
    raise ValueError(f"non-standard JSON token: {token}")


def verify_artefact_bundle(
    context: TaskRuntimeContext,
    experiment_id: str,
) -> ArtefactBundleIntegrity:
    """Verify completeness and hashes recorded inside a published bundle."""

    if context.output_dir is None or not context.run_id:
        return ArtefactBundleIntegrity(False, False, {}, None, ("output_disabled",))
    run_id = safe_segment(context.run_id, "run_id")
    safe_experiment_id = safe_segment(experiment_id, "experiment_id")
    directory = context.output_dir / "runs" / run_id / safe_experiment_id
    if not directory.is_dir():
        return ArtefactBundleIntegrity(False, False, {}, None, ("bundle_missing",))
    if {item.name for item in directory.iterdir()} != set(ARTEFACT_FILENAMES):
        return ArtefactBundleIntegrity(
            False, False, {}, None, ("bundle_file_set_mismatch",)
        )
    try:
        manifest = json.loads(
            (directory / "evaluator_manifest.json").read_text(encoding="utf-8"),
            parse_constant=_reject_non_standard_json,
        )
        integrity = manifest.pop(ARTEFACT_BUNDLE_METADATA_KEY)
        if integrity["schema_version"] != ARTEFACT_BUNDLE_SCHEMA_VERSION:
            raise ValueError("bundle schema mismatch")
        if integrity["result_encoding_version"] != SCIENTIFIC_JSON_ENCODING_VERSION:
            raise ValueError("result encoding mismatch")
        if tuple(integrity["expected_filenames"]) != ARTEFACT_FILENAMES:
            raise ValueError("expected filename mismatch")
        if integrity["completed"] is not True:
            raise ValueError("bundle is not marked complete")
        actual_hashes = {
            name: _sha256((directory / name).read_bytes())
            for name in ARTEFACT_FILENAMES[:-1]
        }
        actual_hashes["evaluator_manifest.json"] = _sha256(strict_json_bytes(manifest))
        stored_hashes = {
            str(name): str(digest)
            for name, digest in integrity["payload_sha256"].items()
        }
        actual_bundle_hash = _bundle_hash(actual_hashes)
        untampered = (
            stored_hashes == actual_hashes
            and integrity["bundle_sha256"] == actual_bundle_hash
        )
        return ArtefactBundleIntegrity(
            True,
            untampered,
            actual_hashes,
            actual_bundle_hash,
            () if untampered else ("bundle_hash_mismatch",),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return ArtefactBundleIntegrity(
            True, False, {}, None, ("bundle_integrity_metadata_invalid",)
        )


def artefact_bundle_identity(
    context: TaskRuntimeContext,
    experiment_id: str,
) -> ArtefactBundleIdentity:
    """Return a fail-closed identity for a complete, compatible bundle."""

    references = artefact_references(context, experiment_id)
    if context.output_dir is None or not references:
        raise ArtefactBundleError("completed_evaluation_artefact_bundle_missing")
    evaluator_reference = next(
        (
            reference
            for reference in references
            if Path(reference).name == "evaluator_manifest.json"
        ),
        None,
    )
    if evaluator_reference is None:
        raise ArtefactBundleError("completed_evaluation_artefact_bundle_missing")
    try:
        manifest = json.loads(
            (context.output_dir / evaluator_reference).read_text(encoding="utf-8"),
            parse_constant=_reject_non_standard_json,
        )
        metadata = manifest[ARTEFACT_BUNDLE_METADATA_KEY]
        schema_version = str(metadata["schema_version"])
        encoding_version = str(metadata["result_encoding_version"])
        expected_filenames = tuple(metadata["expected_filenames"])
        completed = metadata["completed"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtefactBundleError(
            "completed_evaluation_artefact_bundle_tampered"
        ) from exc
    if schema_version != ARTEFACT_BUNDLE_SCHEMA_VERSION:
        raise ArtefactBundleError("artefact_bundle_schema_incompatible")
    if encoding_version != SCIENTIFIC_JSON_ENCODING_VERSION:
        raise ArtefactBundleError("artefact_result_encoding_incompatible")
    if expected_filenames != ARTEFACT_FILENAMES or completed is not True:
        raise ArtefactBundleError("completed_evaluation_artefact_bundle_tampered")
    integrity = verify_artefact_bundle(context, experiment_id)
    if not integrity.complete:
        raise ArtefactBundleError("completed_evaluation_artefact_bundle_missing")
    if not integrity.untampered or integrity.bundle_sha256 is None:
        raise ArtefactBundleError("completed_evaluation_artefact_bundle_tampered")
    manifest_hash = integrity.payload_sha256.get("evaluator_manifest.json")
    if manifest_hash is None:
        raise ArtefactBundleError("completed_evaluation_artefact_bundle_tampered")
    return ArtefactBundleIdentity(
        bundle_sha256=integrity.bundle_sha256,
        schema_version=schema_version,
        result_encoding_version=encoding_version,
        references=references,
        evaluator_manifest_payload_hash=manifest_hash,
    )
