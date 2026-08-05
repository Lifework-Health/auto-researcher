"""Append-only durable storage for nondeterministic model call side effects."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Protocol

from auto_researcher.agents.models import AgentCallRecord
from auto_researcher.contracts.enums import AgentCallStatus


class AgentCallStore(Protocol):
    def append(self, record: AgentCallRecord) -> None: ...
    def records_for_call(self, call_id: str) -> tuple[AgentCallRecord, ...]: ...
    def latest(self, call_id: str) -> AgentCallRecord | None: ...
    def list_records(
        self, run_id: str | None = None
    ) -> tuple[AgentCallRecord, ...]: ...
    def create_retry(
        self, call_id: str, *, created_at: datetime
    ) -> AgentCallRecord: ...
    def reserve(
        self,
        record: AgentCallRecord,
        *,
        maximum_calls: int,
        maximum_total_cost: float,
    ) -> tuple[AgentCallRecord, bool]: ...
    def transition(
        self,
        record: AgentCallRecord,
        *,
        expected_status: AgentCallStatus,
    ) -> bool: ...


class InMemoryAgentCallStore:
    def __init__(self) -> None:
        self._records: list[AgentCallRecord] = []
        self._lock = RLock()

    def append(self, record: AgentCallRecord) -> None:
        with self._lock:
            existing = next(
                (item for item in self._records if item.record_id == record.record_id),
                None,
            )
            if existing is not None:
                if existing != record:
                    raise ValueError(
                        f"agent call record {record.record_id!r} is immutable"
                    )
                return
            self._records.append(record)

    def records_for_call(self, call_id: str) -> tuple[AgentCallRecord, ...]:
        with self._lock:
            return tuple(item for item in self._records if item.call_id == call_id)

    def latest(self, call_id: str) -> AgentCallRecord | None:
        records = self.records_for_call(call_id)
        return records[-1] if records else None

    def list_records(self, run_id: str | None = None) -> tuple[AgentCallRecord, ...]:
        with self._lock:
            return tuple(
                item
                for item in self._records
                if run_id is None or item.run_id == run_id
            )

    def create_retry(self, call_id: str, *, created_at: datetime) -> AgentCallRecord:
        return _create_retry(self, call_id, created_at=created_at)

    def reserve(
        self,
        record: AgentCallRecord,
        *,
        maximum_calls: int,
        maximum_total_cost: float,
    ) -> tuple[AgentCallRecord, bool]:
        with self._lock:
            return _reserve_locked(
                self._records,
                record,
                maximum_calls=maximum_calls,
                maximum_total_cost=maximum_total_cost,
                append=self._records.append,
            )

    def transition(
        self,
        record: AgentCallRecord,
        *,
        expected_status: AgentCallStatus,
    ) -> bool:
        with self._lock:
            latest = next(
                (
                    item
                    for item in reversed(self._records)
                    if item.call_id == record.call_id
                ),
                None,
            )
            if latest == record:
                return True
            if latest is not None and latest.record_id == record.record_id:
                raise ValueError("model_call_completion_conflict")
            if latest is None or latest.status != expected_status:
                return False
            self.append(record)
            return True


class SQLiteAgentCallStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_call_records (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL UNIQUE,
                call_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        columns = {
            row[1]
            for row in self._connection.execute(
                "PRAGMA table_info(agent_call_records)"
            ).fetchall()
        }
        if "semantic_key" not in columns:
            self._connection.execute(
                "ALTER TABLE agent_call_records ADD COLUMN semantic_key TEXT"
            )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS agent_call_id_idx ON agent_call_records(call_id)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS agent_call_run_idx ON agent_call_records(run_id)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS agent_call_semantic_idx "
            "ON agent_call_records(semantic_key)"
        )
        self._connection.commit()

    def append(self, record: AgentCallRecord) -> None:
        payload = record.model_dump_json()
        with self._lock:
            existing = self._connection.execute(
                "SELECT payload FROM agent_call_records WHERE record_id = ?",
                (record.record_id,),
            ).fetchone()
            if existing is not None:
                if AgentCallRecord.model_validate_json(existing[0]) != record:
                    raise ValueError(
                        f"agent call record {record.record_id!r} is immutable"
                    )
                return
            self._connection.execute(
                """
                INSERT INTO agent_call_records(record_id, call_id, run_id, payload)
                VALUES (?, ?, ?, ?)
                """,
                (record.record_id, record.call_id, record.run_id, payload),
            )
            if record.semantic_key is not None:
                self._connection.execute(
                    "UPDATE agent_call_records SET semantic_key = ? WHERE record_id = ?",
                    (record.semantic_key, record.record_id),
                )
            self._connection.commit()

    def records_for_call(self, call_id: str) -> tuple[AgentCallRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM agent_call_records WHERE call_id = ? ORDER BY sequence",
                (call_id,),
            ).fetchall()
        return tuple(AgentCallRecord.model_validate_json(row[0]) for row in rows)

    def latest(self, call_id: str) -> AgentCallRecord | None:
        records = self.records_for_call(call_id)
        return records[-1] if records else None

    def list_records(self, run_id: str | None = None) -> tuple[AgentCallRecord, ...]:
        query = "SELECT payload FROM agent_call_records"
        params: tuple[str, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            params = (run_id,)
        query += " ORDER BY sequence"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return tuple(AgentCallRecord.model_validate_json(row[0]) for row in rows)

    def create_retry(self, call_id: str, *, created_at: datetime) -> AgentCallRecord:
        return _create_retry(self, call_id, created_at=created_at)

    def reserve(
        self,
        record: AgentCallRecord,
        *,
        maximum_calls: int,
        maximum_total_cost: float,
    ) -> tuple[AgentCallRecord, bool]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                rows = self._connection.execute(
                    "SELECT payload FROM agent_call_records ORDER BY sequence"
                ).fetchall()
                records = [AgentCallRecord.model_validate_json(row[0]) for row in rows]

                def insert(item: AgentCallRecord) -> None:
                    self._connection.execute(
                        "INSERT INTO agent_call_records"
                        "(record_id, call_id, run_id, payload, semantic_key) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            item.record_id,
                            item.call_id,
                            item.run_id,
                            item.model_dump_json(),
                            item.semantic_key,
                        ),
                    )

                result = _reserve_locked(
                    records,
                    record,
                    maximum_calls=maximum_calls,
                    maximum_total_cost=maximum_total_cost,
                    append=insert,
                )
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    def transition(
        self,
        record: AgentCallRecord,
        *,
        expected_status: AgentCallStatus,
    ) -> bool:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT payload FROM agent_call_records WHERE call_id = ? "
                    "ORDER BY sequence DESC LIMIT 1",
                    (record.call_id,),
                ).fetchone()
                latest = (
                    AgentCallRecord.model_validate_json(row[0])
                    if row is not None
                    else None
                )
                if latest == record:
                    self._connection.rollback()
                    return True
                if latest is not None and latest.record_id == record.record_id:
                    self._connection.rollback()
                    raise ValueError("model_call_completion_conflict")
                if latest is None or latest.status != expected_status:
                    self._connection.rollback()
                    return False
                self._connection.execute(
                    "INSERT INTO agent_call_records"
                    "(record_id, call_id, run_id, payload, semantic_key) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        record.record_id,
                        record.call_id,
                        record.run_id,
                        record.model_dump_json(),
                        record.semantic_key,
                    ),
                )
                self._connection.commit()
                return True
            except Exception:
                self._connection.rollback()
                raise

    def close(self) -> None:
        self._connection.close()


def _create_retry(
    store: AgentCallStore,
    call_id: str,
    *,
    created_at: datetime,
) -> AgentCallRecord:
    original = store.latest(call_id)
    if original is None:
        raise KeyError(f"unknown agent call {call_id!r}")
    if original.status != AgentCallStatus.INDETERMINATE:
        raise ValueError("only an INDETERMINATE call may be explicitly retried")
    child_ids = tuple(
        dict.fromkeys(
            item.call_id
            for item in store.list_records(original.run_id)
            if item.retry_of_call_id == call_id
        )
    )
    children = tuple(
        child for child_id in child_ids if (child := store.latest(child_id)) is not None
    )
    if any(child.status == AgentCallStatus.COMPLETED for child in children):
        raise ValueError("the retry lineage already contains a completed call")
    unused_authorisations = [
        child
        for child in children
        if child.status == AgentCallStatus.RESERVED
        and not child.provider_request_started
    ]
    if len(children) == 1 and unused_authorisations:
        return unused_authorisations[0]
    if children:
        raise ValueError(
            "the retry lineage is already active; retry its indeterminate leaf"
        )
    ordinal = len(child_ids) + 1
    retry_call_id = (
        f"{call_id}-retry-"
        + hashlib.sha256(f"{call_id}:{ordinal}".encode()).hexdigest()[:8]
    )
    record = original.model_copy(
        update={
            "record_id": f"{retry_call_id}:authorized",
            "call_id": retry_call_id,
            "status": AgentCallStatus.RESERVED,
            "structured_output": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "estimated_cost": 0,
            "latency_ms": 0,
            "provider_request_id": None,
            "attempt_count": 0,
            "created_at": created_at.astimezone(UTC),
            "error_code": None,
            "response_hash": None,
            "retry_of_call_id": call_id,
            "provider_request_started": False,
        }
    )
    store.append(record)
    return record


def stable_record_id(call_id: str, status: AgentCallStatus, ordinal: int) -> str:
    return f"{call_id}:{ordinal}:{status.value.lower()}"


def _reserve_locked(
    records: list[AgentCallRecord],
    record: AgentCallRecord,
    *,
    maximum_calls: int,
    maximum_total_cost: float,
    append,
) -> tuple[AgentCallRecord, bool]:
    same_semantic = [
        item for item in records if item.semantic_key == record.semantic_key
    ]
    if same_semantic:
        existing_call_ids = {item.call_id for item in same_semantic}
        if existing_call_ids != {record.call_id}:
            raise ValueError("model_call_identity_conflict")
        latest = same_semantic[-1]
        identity_fields = (
            "approval_hash",
            "budget_identity",
            "provider",
            "model_id",
            "prompt_name",
            "prompt_version",
            "input_payload_hash",
            "response_schema_version",
        )
        if any(
            getattr(latest, field) != getattr(record, field)
            for field in identity_fields
        ):
            raise ValueError("model_call_identity_conflict")
        return latest, False
    latest_by_call: dict[str, AgentCallRecord] = {}
    for item in records:
        latest_by_call[item.call_id] = item
    budget_records = [
        item
        for item in latest_by_call.values()
        if item.budget_identity == record.budget_identity
        and item.role == record.role
        and item.status != AgentCallStatus.FAILED_BEFORE_DISPATCH
    ]
    if len(budget_records) >= maximum_calls:
        raise ValueError("model_call_budget_exhausted")
    reserved_cost = sum(item.maximum_reserved_cost for item in budget_records)
    if reserved_cost + record.maximum_reserved_cost > maximum_total_cost + 1e-12:
        raise ValueError("model_call_cost_limit_exceeded")
    append(record)
    return record, True
