"""SQLite implementation whose public surface permits inserts and reads only."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock

from auto_researcher.contracts.enums import EventType
from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.provenance.reuse import (
    EvaluationReuseRecord,
    VerificationReuseRecord,
)


class SQLiteProvenanceStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = Lock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        columns = {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(decision_events)")
        }
        if "semantic_key" not in columns:
            self._connection.execute(
                "ALTER TABLE decision_events ADD COLUMN semantic_key TEXT"
            )
        if "semantic_payload_hash" not in columns:
            self._connection.execute(
                "ALTER TABLE decision_events ADD COLUMN semantic_payload_hash TEXT"
            )
        self._connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS decision_events_semantic_key_idx
            ON decision_events(semantic_key)
            WHERE semantic_key IS NOT NULL
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_reuse_records (
                run_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(run_id, experiment_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS verification_reuse_records (
                run_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(run_id, experiment_id)
            )
            """
        )
        self._connection.commit()

    def append_event(self, event: DecisionEvent) -> None:
        try:
            with self._lock, self._connection:
                self._connection.execute(
                    """
                    INSERT INTO decision_events(event_id, run_id, event_type, timestamp, payload)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.run_id,
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        event.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"event {event.event_id!r} already exists; events are immutable"
            ) from exc

    def append_event_idempotent(self, event: DecisionEvent) -> bool:
        existing = self.get_event(event.event_id)
        if existing is not None:
            if existing != event:
                raise ValueError(
                    f"event {event.event_id!r} already exists with different content"
                )
            return False
        self.append_event(event)
        return True

    def append_semantic_event(
        self,
        event: DecisionEvent,
        semantic_key: str,
        payload_hash: str,
    ) -> tuple[DecisionEvent, bool]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload, semantic_payload_hash
                FROM decision_events WHERE semantic_key = ?
                """,
                (semantic_key,),
            ).fetchone()
            if row is not None:
                existing = DecisionEvent.model_validate_json(row[0])
                if row[1] != payload_hash:
                    raise ValueError("conflicting_semantic_provenance_event")
                return existing, False
            try:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO decision_events(
                            event_id, run_id, event_type, timestamp, payload,
                            semantic_key, semantic_payload_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_id,
                            event.run_id,
                            event.event_type.value,
                            event.timestamp.isoformat(),
                            event.model_dump_json(),
                            semantic_key,
                            payload_hash,
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                row = self._connection.execute(
                    """
                    SELECT payload, semantic_payload_hash
                    FROM decision_events WHERE semantic_key = ?
                    """,
                    (semantic_key,),
                ).fetchone()
                if row is not None and row[1] == payload_hash:
                    return DecisionEvent.model_validate_json(row[0]), False
                raise ValueError("conflicting_semantic_provenance_event") from exc
        return event, True

    def get_event(self, event_id: str) -> DecisionEvent | None:
        row = self._connection.execute(
            "SELECT payload FROM decision_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return DecisionEvent.model_validate_json(row[0]) if row else None

    def list_events(self, run_id: str) -> list[DecisionEvent]:
        rows = self._connection.execute(
            "SELECT payload FROM decision_events WHERE run_id = ? ORDER BY sequence",
            (run_id,),
        ).fetchall()
        return [DecisionEvent.model_validate_json(row[0]) for row in rows]

    def list_events_by_type(
        self,
        run_id: str,
        event_type: EventType,
    ) -> list[DecisionEvent]:
        rows = self._connection.execute(
            """
            SELECT payload FROM decision_events
            WHERE run_id = ? AND event_type = ?
            ORDER BY sequence
            """,
            (run_id, event_type.value),
        ).fetchall()
        return [DecisionEvent.model_validate_json(row[0]) for row in rows]

    def get_evaluation_reuse(
        self,
        run_id: str,
        experiment_id: str,
    ) -> EvaluationReuseRecord | None:
        row = self._connection.execute(
            """
            SELECT payload FROM evaluation_reuse_records
            WHERE run_id = ? AND experiment_id = ?
            """,
            (run_id, experiment_id),
        ).fetchone()
        return EvaluationReuseRecord.model_validate_json(row[0]) if row else None

    def append_evaluation_reuse(self, record: EvaluationReuseRecord) -> None:
        self._append_reuse_record(
            "evaluation_reuse_records",
            record.run_id,
            record.experiment_id,
            record.model_dump_json(),
        )

    def get_verification_reuse(
        self,
        run_id: str,
        experiment_id: str,
    ) -> VerificationReuseRecord | None:
        row = self._connection.execute(
            """
            SELECT payload FROM verification_reuse_records
            WHERE run_id = ? AND experiment_id = ?
            """,
            (run_id, experiment_id),
        ).fetchone()
        return VerificationReuseRecord.model_validate_json(row[0]) if row else None

    def append_verification_reuse(self, record: VerificationReuseRecord) -> None:
        self._append_reuse_record(
            "verification_reuse_records",
            record.run_id,
            record.experiment_id,
            record.model_dump_json(),
        )

    def _append_reuse_record(
        self,
        table: str,
        run_id: str,
        experiment_id: str,
        payload: str,
    ) -> None:
        with self._lock:
            row = self._connection.execute(
                f"SELECT payload FROM {table} WHERE run_id = ? AND experiment_id = ?",
                (run_id, experiment_id),
            ).fetchone()
            if row is not None:
                if row[0] != payload:
                    raise ValueError("conflicting_result_reuse_record")
                return
            try:
                with self._connection:
                    self._connection.execute(
                        f"INSERT INTO {table}(run_id, experiment_id, payload) VALUES (?, ?, ?)",
                        (run_id, experiment_id, payload),
                    )
            except sqlite3.IntegrityError as exc:
                row = self._connection.execute(
                    f"SELECT payload FROM {table} WHERE run_id = ? AND experiment_id = ?",
                    (run_id, experiment_id),
                ).fetchone()
                if row is not None and row[0] == payload:
                    return
                raise ValueError("conflicting_result_reuse_record") from exc

    def close(self) -> None:
        self._connection.close()
