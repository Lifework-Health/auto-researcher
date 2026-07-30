"""SQLite implementation whose public surface permits inserts and reads only."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock

from auto_researcher.contracts.enums import EventType
from auto_researcher.contracts.models import DecisionEvent


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
            raise ValueError(f"event {event.event_id!r} already exists; events are immutable") from exc

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

    def close(self) -> None:
        self._connection.close()
