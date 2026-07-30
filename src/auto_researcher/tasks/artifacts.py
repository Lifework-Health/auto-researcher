"""Safe, atomic task artefact bundle helpers."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from auto_researcher.contracts.models import EvaluationResult, ExperimentSpec
from auto_researcher.tasks.models import DatasetManifest, TaskRuntimeContext

ARTEFACT_FILENAMES = (
    "experiment_spec.json",
    "evaluation_result.json",
    "dataset_manifest.json",
    "evaluator_manifest.json",
)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_artefact_bundle(
    context: TaskRuntimeContext,
    experiment: ExperimentSpec,
    evaluation: EvaluationResult,
    dataset_manifest: DatasetManifest,
    evaluator_manifest: dict[str, Any],
) -> None:
    references = artefact_references(context, experiment.experiment_id)
    if not references:
        return
    assert context.output_dir is not None
    values = (
        experiment,
        evaluation,
        dataset_manifest,
        evaluator_manifest,
    )
    for relative, value in zip(references, values, strict=True):
        atomic_json_write(context.output_dir / relative, value)
