"""Append-only storage for changing external knowledge reads."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Protocol

from auto_researcher.contracts.enums import KnowledgeRetrievalStatus
from auto_researcher.knowledge.models import KnowledgeRetrievalRecord


class KnowledgeRetrievalStore(Protocol):
    def append(self, record: KnowledgeRetrievalRecord) -> None: ...
    def records_for_retrieval(
        self, retrieval_id: str
    ) -> tuple[KnowledgeRetrievalRecord, ...]: ...
    def latest(self, retrieval_id: str) -> KnowledgeRetrievalRecord | None: ...
    def list_records(
        self, run_id: str | None = None
    ) -> tuple[KnowledgeRetrievalRecord, ...]: ...
    def create_retry(
        self, retrieval_id: str, *, created_at: datetime
    ) -> KnowledgeRetrievalRecord: ...


class InMemoryKnowledgeRetrievalStore:
    def __init__(self) -> None:
        self._records: list[KnowledgeRetrievalRecord] = []
        self._lock = RLock()

    def append(self, record: KnowledgeRetrievalRecord) -> None:
        with self._lock:
            existing = next(
                (item for item in self._records if item.record_id == record.record_id),
                None,
            )
            if existing is not None:
                if existing != record:
                    raise ValueError("knowledge retrieval records are immutable")
                return
            self._records.append(record)

    def records_for_retrieval(
        self, retrieval_id: str
    ) -> tuple[KnowledgeRetrievalRecord, ...]:
        with self._lock:
            return tuple(
                item for item in self._records if item.retrieval_id == retrieval_id
            )

    def latest(self, retrieval_id: str) -> KnowledgeRetrievalRecord | None:
        records = self.records_for_retrieval(retrieval_id)
        return records[-1] if records else None

    def list_records(
        self, run_id: str | None = None
    ) -> tuple[KnowledgeRetrievalRecord, ...]:
        with self._lock:
            return tuple(
                item
                for item in self._records
                if run_id is None or item.run_id == run_id
            )

    def create_retry(
        self, retrieval_id: str, *, created_at: datetime
    ) -> KnowledgeRetrievalRecord:
        return _create_retry(self, retrieval_id, created_at=created_at)


class SQLiteKnowledgeRetrievalStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_retrieval_records (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL UNIQUE,
                retrieval_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS knowledge_retrieval_id_idx "
            "ON knowledge_retrieval_records(retrieval_id)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS knowledge_retrieval_run_idx "
            "ON knowledge_retrieval_records(run_id)"
        )
        self._connection.commit()

    def append(self, record: KnowledgeRetrievalRecord) -> None:
        payload = record.model_dump_json()
        with self._lock:
            existing = self._connection.execute(
                "SELECT payload FROM knowledge_retrieval_records WHERE record_id = ?",
                (record.record_id,),
            ).fetchone()
            if existing:
                if KnowledgeRetrievalRecord.model_validate_json(existing[0]) != record:
                    raise ValueError("knowledge retrieval records are immutable")
                return
            self._connection.execute(
                "INSERT INTO knowledge_retrieval_records"
                "(record_id, retrieval_id, run_id, payload) VALUES (?, ?, ?, ?)",
                (record.record_id, record.retrieval_id, record.run_id, payload),
            )
            self._connection.commit()

    def records_for_retrieval(
        self, retrieval_id: str
    ) -> tuple[KnowledgeRetrievalRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM knowledge_retrieval_records "
                "WHERE retrieval_id = ? ORDER BY sequence",
                (retrieval_id,),
            ).fetchall()
        return tuple(
            KnowledgeRetrievalRecord.model_validate_json(row[0]) for row in rows
        )

    def latest(self, retrieval_id: str) -> KnowledgeRetrievalRecord | None:
        records = self.records_for_retrieval(retrieval_id)
        return records[-1] if records else None

    def list_records(
        self, run_id: str | None = None
    ) -> tuple[KnowledgeRetrievalRecord, ...]:
        query = "SELECT payload FROM knowledge_retrieval_records"
        params: tuple[str, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            params = (run_id,)
        query += " ORDER BY sequence"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return tuple(
            KnowledgeRetrievalRecord.model_validate_json(row[0]) for row in rows
        )

    def create_retry(
        self, retrieval_id: str, *, created_at: datetime
    ) -> KnowledgeRetrievalRecord:
        return _create_retry(self, retrieval_id, created_at=created_at)

    def close(self) -> None:
        self._connection.close()


def _create_retry(
    store: KnowledgeRetrievalStore,
    retrieval_id: str,
    *,
    created_at: datetime,
) -> KnowledgeRetrievalRecord:
    original = store.latest(retrieval_id)
    if original is None:
        raise KeyError(f"unknown knowledge retrieval {retrieval_id!r}")
    if original.status != KnowledgeRetrievalStatus.INDETERMINATE:
        raise ValueError("only an INDETERMINATE knowledge retrieval may be retried")
    children = tuple(
        dict.fromkeys(
            item.retrieval_id
            for item in store.list_records(original.run_id)
            if item.retry_of_retrieval_id == retrieval_id
        )
    )
    latest_children = tuple(
        item for child_id in children if (item := store.latest(child_id)) is not None
    )
    if any(
        item.status == KnowledgeRetrievalStatus.COMPLETED for item in latest_children
    ):
        raise ValueError("knowledge retry lineage already completed")
    unused = [
        item
        for item in latest_children
        if item.status == KnowledgeRetrievalStatus.RESERVED
        and not item.provider_request_started
    ]
    if len(latest_children) == 1 and unused:
        return unused[0]
    if latest_children:
        raise ValueError("knowledge retry lineage is already active")
    attempt_id = (
        f"{retrieval_id}-retry-"
        + hashlib.sha256(f"{retrieval_id}:1".encode()).hexdigest()[:8]
    )
    request = original.request.model_copy(update={"retrieval_id": attempt_id})
    record = original.model_copy(
        update={
            "record_id": f"{attempt_id}:authorized",
            "retrieval_id": attempt_id,
            "status": KnowledgeRetrievalStatus.RESERVED,
            "request": request,
            "bundle": None,
            "errors": (),
            "retry_of_retrieval_id": retrieval_id,
            "provider_request_started": False,
            "created_at": created_at.astimezone(UTC),
        }
    )
    store.append(record)
    return record


def retrieval_record_id(
    retrieval_id: str,
    status: KnowledgeRetrievalStatus,
    ordinal: int,
) -> str:
    return f"{retrieval_id}:{ordinal}:{status.value.lower()}"
