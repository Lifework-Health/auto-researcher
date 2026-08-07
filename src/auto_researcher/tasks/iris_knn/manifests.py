"""Immutable Iris data and fold identities."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from auto_researcher.tasks.models import DatasetManifest, TaskRuntimeContext

DATASET_ID = "uci-iris-bezdek"
DATASET_VERSION = "uci-iris-bezdek-2023-05-22-v1"
DATA_FILE_NAME = "bezdekIris.data"
DATA_SHA256 = "0fed2a99db77ec533a62dc66894d3ec6df3b58b6a8f3cf4a6b47e4086b7f97dc"
FOLD_FILE_NAME = "folds-v1.json"
FOLD_SHA256 = "8f31b0bcb1cadea5599ceb389142acfff41c9fbb057215bf861c5af37fe3a831"
FOLD_VERSION = "iris-stratified-5fold-v1"
LOADER_VERSION = "iris-csv-loader-v1"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"
SOURCE_DOI = "10.24432/C56C76"
FEATURE_NAMES = (
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
)
CLASS_NAMES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
CLASS_COUNTS = {name: 50 for name in CLASS_NAMES}
DATA_DIR = Path(__file__).with_name("data")


@dataclass(frozen=True)
class IrisRow:
    index: int
    features: tuple[float, float, float, float]
    label: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_paths(context: TaskRuntimeContext | None = None) -> tuple[Path, Path]:
    root = context.data_dir if context and context.data_dir is not None else DATA_DIR
    return root / DATA_FILE_NAME, root / FOLD_FILE_NAME


def verify_data_files(context: TaskRuntimeContext | None = None) -> tuple[bool, bool]:
    data_path, fold_path = data_paths(context)
    return (
        data_path.is_file() and _sha256(data_path) == DATA_SHA256,
        fold_path.is_file() and _sha256(fold_path) == FOLD_SHA256,
    )


def load_iris_rows(context: TaskRuntimeContext | None = None) -> tuple[IrisRow, ...]:
    data_path, _ = data_paths(context)
    if _sha256(data_path) != DATA_SHA256:
        raise ValueError("iris_dataset_hash_mismatch")
    rows: list[IrisRow] = []
    with data_path.open(newline="", encoding="ascii") as handle:
        for index, fields in enumerate(csv.reader(handle)):
            if not fields:
                continue
            if len(fields) != 5 or fields[4] not in CLASS_NAMES:
                raise ValueError("iris_dataset_schema_mismatch")
            values = tuple(float(item) for item in fields[:4])
            if len(values) != 4:
                raise ValueError("iris_dataset_schema_mismatch")
            rows.append(IrisRow(index, values, fields[4]))  # type: ignore[arg-type]
    if (
        len(rows) != 150
        or {name: sum(row.label == name for row in rows) for name in CLASS_NAMES}
        != CLASS_COUNTS
    ):
        raise ValueError("iris_dataset_shape_mismatch")
    return tuple(rows)


def load_fold_assignments(
    context: TaskRuntimeContext | None = None,
) -> tuple[int, ...]:
    _, fold_path = data_paths(context)
    if _sha256(fold_path) != FOLD_SHA256:
        raise ValueError("iris_fold_hash_mismatch")
    payload = json.loads(fold_path.read_text(encoding="utf-8"))
    assignments = tuple(payload.get("assignments", ()))
    if (
        payload.get("fold_version") != FOLD_VERSION
        or payload.get("dataset_sha256") != DATA_SHA256
        or len(assignments) != 150
        or set(assignments) != set(range(5))
        or any(assignments.count(fold) != 30 for fold in range(5))
    ):
        raise ValueError("iris_fold_schema_mismatch")
    rows = load_iris_rows(context)
    for fold in range(5):
        counts = {
            name: sum(
                assignment == fold and row.label == name
                for row, assignment in zip(rows, assignments, strict=True)
            )
            for name in CLASS_NAMES
        }
        if counts != {name: 10 for name in CLASS_NAMES}:
            raise ValueError("iris_fold_stratification_mismatch")
    return assignments


def build_dataset_manifest(context: TaskRuntimeContext) -> DatasetManifest:
    created_at = context.manifest_created_at or datetime.now(UTC)
    class_counts: dict[str, JsonValue] = dict(CLASS_COUNTS)
    metadata: dict[str, JsonValue] = {
        "dataset_id": DATASET_ID,
        "source": "UCI Machine Learning Repository",
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "licence": "CC-BY-4.0",
        "row_count": 150,
        "feature_count": 4,
        "class_count": 3,
        "feature_names": list(FEATURE_NAMES),
        "class_names": list(CLASS_NAMES),
        "class_counts": class_counts,
        "fold_version": FOLD_VERSION,
        "fold_count": 5,
        "fold_class_counts": [{name: 10 for name in CLASS_NAMES} for _ in range(5)],
        "evaluator_input_schema_version": "iris-knn-configuration-v1",
        "contains_patient_data": False,
    }
    return DatasetManifest(
        task_id="iris_knn",
        dataset_version=DATASET_VERSION,
        files=(DATA_FILE_NAME, FOLD_FILE_NAME),
        hashes={DATA_FILE_NAME: DATA_SHA256, FOLD_FILE_NAME: FOLD_SHA256},
        loader_version=LOADER_VERSION,
        created_at=created_at,
        metadata=metadata,
    )
