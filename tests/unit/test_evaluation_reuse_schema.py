from __future__ import annotations

from datetime import UTC, datetime
import json
import sqlite3
from threading import Thread

import pytest

from auto_researcher.contracts.enums import ProvenanceKind
from auto_researcher.contracts.models import EvaluationResult
from auto_researcher.provenance.reuse import EvaluationReuseRecord
from auto_researcher.provenance.sqlite_store import (
    EVALUATION_REUSE_STORE_SCHEMA,
    SQLiteProvenanceStore,
)


def _record() -> EvaluationReuseRecord:
    references = (
        "runs/run-1/experiment-1/experiment_spec.json",
        "runs/run-1/experiment-1/evaluation_result.json",
        "runs/run-1/experiment-1/dataset_manifest.json",
        "runs/run-1/experiment-1/evaluator_manifest.json",
    )
    result = EvaluationResult(
        experiment_id="experiment-1",
        success=True,
        primary_score=0.8,
        metrics={"objective": 0.8},
        constraint_results={"valid": True},
        artefact_references=references,
        evaluator_version="evaluator-v1",
        provenance=ProvenanceKind.SIMULATED,
    )
    return EvaluationReuseRecord(
        run_id="run-1",
        experiment_id="experiment-1",
        scientific_identity_hash="a" * 64,
        experiment_payload_hash="b" * 64,
        result_payload_hash="c" * 64,
        evaluator_version="evaluator-v1",
        dataset_version="dataset-v1",
        code_version="code-v1",
        artefact_bundle_hash="d" * 64,
        artefact_bundle_schema_version="experiment-bundle-v2",
        result_encoding_version="scientific-json-v1",
        expected_artefact_references=references,
        evaluator_manifest_payload_hash="e" * 64,
        completed_at=datetime(2026, 8, 3, tzinfo=UTC),
        result=result,
    )


def test_fresh_store_uses_v2_and_round_trips_record(tmp_path):
    store = SQLiteProvenanceStore(tmp_path / "reuse.sqlite")
    record = _record()
    try:
        assert store.evaluation_reuse_store_schema() == EVALUATION_REUSE_STORE_SCHEMA
        store.append_evaluation_reuse(record)
        assert store.get_evaluation_reuse("run-1", "experiment-1") == record
    finally:
        store.close()


def test_legacy_v1_record_is_actionable_and_never_silently_migrated(tmp_path):
    path = tmp_path / "legacy.sqlite"
    store = SQLiteProvenanceStore(path)
    store.close()
    legacy = {
        "protocol_version": "evaluation-reuse-v1",
        "run_id": "run-1",
        "experiment_id": "experiment-1",
        "scientific_identity_hash": "a" * 64,
    }
    payload = json.dumps(legacy, sort_keys=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO evaluation_reuse_records(run_id, experiment_id, payload) "
        "VALUES (?, ?, ?)",
        ("run-1", "experiment-1", payload),
    )
    connection.commit()
    connection.close()

    reopened = SQLiteProvenanceStore(path)
    try:
        with pytest.raises(
            ValueError,
            match="legacy_evaluation_reuse_record_not_reusable",
        ):
            reopened.get_evaluation_reuse("run-1", "experiment-1")
    finally:
        reopened.close()
    connection = sqlite3.connect(path)
    try:
        stored = connection.execute(
            "SELECT payload FROM evaluation_reuse_records"
        ).fetchone()[0]
    finally:
        connection.close()
    assert stored == payload
    assert "artefact_bundle_hash" not in json.loads(stored)


def test_duplicate_concurrent_identical_insert_is_idempotent(tmp_path):
    store = SQLiteProvenanceStore(tmp_path / "concurrent.sqlite")
    record = _record()
    errors: list[Exception] = []

    def append():
        try:
            store.append_evaluation_reuse(record)
        except Exception as exc:  # pragma: no cover - assertion reports the detail
            errors.append(exc)

    threads = [Thread(target=append) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        assert errors == []
        count = store._connection.execute(
            "SELECT COUNT(*) FROM evaluation_reuse_records"
        ).fetchone()[0]
        assert count == 1
    finally:
        store.close()


def test_conflicting_insert_is_rejected(tmp_path):
    store = SQLiteProvenanceStore(tmp_path / "conflict.sqlite")
    record = _record()
    try:
        store.append_evaluation_reuse(record)
        with pytest.raises(ValueError, match="conflicting_result_reuse_record"):
            store.append_evaluation_reuse(
                record.model_copy(update={"artefact_bundle_hash": "f" * 64})
            )
        assert store.get_evaluation_reuse("run-1", "experiment-1") == record
    finally:
        store.close()
