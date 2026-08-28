"""Append-only, restart-safe SQLite persistence for Research State v1."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from threading import RLock
from typing import TypeVar

from pydantic import TypeAdapter

from auto_researcher.research_state.models import (
    CandidateNextAction,
    EvidenceReferences,
    PlannerDecision,
    PlannerInference,
    RecordType,
    ResearchExperiment,
    ResearchHypothesis,
    ResearchProgramme,
    ResearchState,
    ResearchStateRecord,
    ResearchUncertainty,
    ResearchWorkItem,
    StateRevision,
    WorkStatus,
    record_content_hash,
    record_identity,
    state_revision_for,
)

STORE_SCHEMA_VERSION = "research-state-sqlite-v1"
_RECORD_ADAPTER = TypeAdapter(ResearchStateRecord)
_IMMUTABLE_RECORD_TYPES = {
    RecordType.EXTERNAL_EVIDENCE,
    RecordType.INTERNAL_OBSERVATION,
    RecordType.DIAGNOSTIC_OBSERVATION,
    RecordType.PLANNER_INFERENCE,
    RecordType.PLANNER_DECISION,
}

RecordT = TypeVar("RecordT", bound=ResearchStateRecord)


class SQLiteResearchStateStore:
    """A record journal whose public mutations only append versions."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._lock = RLock()
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_state_metadata (
                    name TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_programmes (
                    programme_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_state_records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    programme_id TEXT NOT NULL,
                    record_type TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    record_revision INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE(programme_id, record_type, record_id, record_revision),
                    FOREIGN KEY(programme_id) REFERENCES research_programmes(programme_id)
                );
                CREATE INDEX IF NOT EXISTS research_state_record_latest_idx
                ON research_state_records(programme_id, record_type, record_id, record_revision);
                CREATE TABLE IF NOT EXISTS research_state_revisions (
                    state_revision INTEGER NOT NULL,
                    state_revision_id TEXT PRIMARY KEY,
                    programme_id TEXT NOT NULL,
                    record_type TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    record_revision INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE(programme_id, state_revision),
                    FOREIGN KEY(programme_id) REFERENCES research_programmes(programme_id)
                );
                """
            )
            row = self._connection.execute(
                "SELECT value FROM research_state_metadata WHERE name = ?",
                ("schema_version",),
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO research_state_metadata(name, value) VALUES (?, ?)",
                    ("schema_version", STORE_SCHEMA_VERSION),
                )
            elif row[0] != STORE_SCHEMA_VERSION:
                raise ValueError("research_state_store_schema_mismatch")

    def create_programme(self, programme: ResearchProgramme) -> ResearchProgramme:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT payload FROM research_programmes WHERE programme_id = ?",
                (programme.programme_id,),
            ).fetchone()
            if row is not None:
                existing = ResearchProgramme.model_validate_json(row[0])
                if existing != programme:
                    raise ValueError("research_programme_immutable_conflict")
                return existing
            self._connection.execute(
                "INSERT INTO research_programmes(programme_id, payload) VALUES (?, ?)",
                (programme.programme_id, programme.model_dump_json()),
            )
        return programme

    def get_programme(self, programme_id: str) -> ResearchProgramme | None:
        row = self._connection.execute(
            "SELECT payload FROM research_programmes WHERE programme_id = ?",
            (programme_id,),
        ).fetchone()
        return ResearchProgramme.model_validate_json(row[0]) if row else None

    def append(self, record: RecordT) -> StateRevision:
        return self.append_many((record,))[0]

    def append_many(
        self, records: tuple[ResearchStateRecord, ...] | list[ResearchStateRecord]
    ) -> tuple[StateRevision, ...]:
        records = tuple(
            _RECORD_ADAPTER.validate_python(record.model_dump(mode="python"))
            for record in records
        )
        if not records:
            return ()
        programme_ids = {record.programme_id for record in records}
        if len(programme_ids) != 1:
            raise ValueError("research state append must target one programme")
        programme_id = next(iter(programme_ids))
        with self._lock, self._connection:
            if self.get_programme(programme_id) is None:
                raise ValueError("research_state_programme_missing")
            available = self._available_identifiers(programme_id, records)
            self._validate_references(records, available)
            latest_revisions = self._latest_record_revisions(programme_id)
            state_revision = self._latest_state_revision(programme_id)
            appended: list[StateRevision] = []
            seen_keys: dict[tuple[RecordType, str, int], str] = {}
            for record in records:
                record_type = RecordType(record.record_type)
                record_id = record_identity(record)
                key = (record_type, record_id, record.revision)
                record_hash = record_content_hash(record)
                duplicate_hash = seen_keys.get(key)
                if duplicate_hash is not None:
                    if duplicate_hash != record_hash:
                        raise ValueError("research_state_record_batch_conflict")
                    raise ValueError("research_state_record_duplicated_in_batch")
                seen_keys[key] = record_hash

                existing = self._connection.execute(
                    "SELECT content_hash FROM research_state_records "
                    "WHERE programme_id = ? AND record_type = ? "
                    "AND record_id = ? AND record_revision = ?",
                    (programme_id, record_type.value, record_id, record.revision),
                ).fetchone()
                if existing is not None:
                    if existing[0] != record_hash:
                        raise ValueError("research_state_record_immutable_conflict")
                    revision_row = self._connection.execute(
                        "SELECT payload FROM research_state_revisions "
                        "WHERE programme_id = ? AND record_type = ? "
                        "AND record_id = ? AND record_revision = ?",
                        (programme_id, record_type.value, record_id, record.revision),
                    ).fetchone()
                    appended.append(StateRevision.model_validate_json(revision_row[0]))
                    continue

                latest = latest_revisions.get((record_type, record_id), 0)
                if record_type in _IMMUTABLE_RECORD_TYPES and record.revision != 1:
                    raise ValueError(
                        "immutable research state records only have revision 1"
                    )
                if record.revision != latest + 1:
                    raise ValueError("research_state_record_revision_gap")
                self._validate_revision_invariants(record)

                state_revision += 1
                revision = state_revision_for(programme_id, state_revision, record)
                self._connection.execute(
                    "INSERT INTO research_state_records"
                    "(programme_id, record_type, record_id, record_revision, "
                    "content_hash, recorded_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        programme_id,
                        record_type.value,
                        record_id,
                        record.revision,
                        record_hash,
                        record.recorded_at.isoformat(),
                        record.model_dump_json(),
                    ),
                )
                self._connection.execute(
                    "INSERT INTO research_state_revisions"
                    "(state_revision, state_revision_id, programme_id, record_type, "
                    "record_id, record_revision, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        revision.state_revision,
                        revision.state_revision_id,
                        programme_id,
                        record_type.value,
                        record_id,
                        record.revision,
                        revision.model_dump_json(),
                    ),
                )
                latest_revisions[(record_type, record_id)] = record.revision
                appended.append(revision)
            return tuple(appended)

    def get_record(
        self,
        programme_id: str,
        record_type: RecordType,
        record_id: str,
        revision: int | None = None,
    ) -> ResearchStateRecord | None:
        if revision is None:
            row = self._connection.execute(
                "SELECT payload FROM research_state_records "
                "WHERE programme_id = ? AND record_type = ? AND record_id = ? "
                "ORDER BY record_revision DESC LIMIT 1",
                (programme_id, record_type.value, record_id),
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT payload FROM research_state_records "
                "WHERE programme_id = ? AND record_type = ? AND record_id = ? "
                "AND record_revision = ?",
                (programme_id, record_type.value, record_id, revision),
            ).fetchone()
        return _RECORD_ADAPTER.validate_json(row[0]) if row else None

    def record_history(
        self, programme_id: str, record_type: RecordType, record_id: str
    ) -> tuple[ResearchStateRecord, ...]:
        rows = self._connection.execute(
            "SELECT payload FROM research_state_records "
            "WHERE programme_id = ? AND record_type = ? AND record_id = ? "
            "ORDER BY record_revision",
            (programme_id, record_type.value, record_id),
        ).fetchall()
        return tuple(_RECORD_ADAPTER.validate_json(row[0]) for row in rows)

    def load_state(self, programme_id: str) -> ResearchState:
        programme = self.get_programme(programme_id)
        if programme is None:
            raise ValueError("research_state_programme_missing")
        rows = self._connection.execute(
            "SELECT records.record_type, records.payload "
            "FROM research_state_records records "
            "JOIN (SELECT record_type, record_id, MAX(record_revision) latest_revision "
            "      FROM research_state_records WHERE programme_id = ? "
            "      GROUP BY record_type, record_id) latest "
            "ON records.record_type = latest.record_type "
            "AND records.record_id = latest.record_id "
            "AND records.record_revision = latest.latest_revision "
            "WHERE records.programme_id = ? "
            "ORDER BY records.record_type, records.record_id",
            (programme_id, programme_id),
        ).fetchall()
        grouped: dict[RecordType, list[ResearchStateRecord]] = defaultdict(list)
        for record_type, payload in rows:
            grouped[RecordType(record_type)].append(
                _RECORD_ADAPTER.validate_json(payload)
            )
        revision_rows = self._connection.execute(
            "SELECT payload FROM research_state_revisions "
            "WHERE programme_id = ? ORDER BY state_revision",
            (programme_id,),
        ).fetchall()
        revisions = tuple(
            StateRevision.model_validate_json(row[0]) for row in revision_rows
        )
        return ResearchState(
            programme=programme,
            state_revision=revisions[-1].state_revision if revisions else 0,
            revision_history=revisions,
            external_evidence=tuple(grouped[RecordType.EXTERNAL_EVIDENCE]),
            internal_observations=tuple(grouped[RecordType.INTERNAL_OBSERVATION]),
            diagnostic_observations=tuple(grouped[RecordType.DIAGNOSTIC_OBSERVATION]),
            hypotheses=tuple(grouped[RecordType.HYPOTHESIS]),
            uncertainties=tuple(grouped[RecordType.UNCERTAINTY]),
            planner_inferences=tuple(grouped[RecordType.PLANNER_INFERENCE]),
            planner_decisions=tuple(grouped[RecordType.PLANNER_DECISION]),
            experiments=tuple(grouped[RecordType.EXPERIMENT]),
            work_items=tuple(grouped[RecordType.WORK_ITEM]),
            candidate_next_actions=tuple(grouped[RecordType.NEXT_ACTION]),
        )

    def active_work(self, programme_id: str) -> tuple[ResearchStateRecord, ...]:
        state = self.load_state(programme_id)
        return tuple(
            (
                *[item for item in state.experiments if item.status.value == "ACTIVE"],
                *[
                    item
                    for item in state.work_items
                    if item.status == WorkStatus.ACTIVE
                ],
            )
        )

    def completed_work(self, programme_id: str) -> tuple[ResearchStateRecord, ...]:
        state = self.load_state(programme_id)
        return tuple(
            (
                *[
                    item
                    for item in state.experiments
                    if item.status.value == "COMPLETED"
                ],
                *[
                    item
                    for item in state.work_items
                    if item.status == WorkStatus.COMPLETED
                ],
            )
        )

    def _available_identifiers(
        self, programme_id: str, records: tuple[ResearchStateRecord, ...]
    ) -> dict[RecordType, set[str]]:
        available: dict[RecordType, set[str]] = defaultdict(set)
        rows = self._connection.execute(
            "SELECT DISTINCT record_type, record_id FROM research_state_records "
            "WHERE programme_id = ?",
            (programme_id,),
        ).fetchall()
        for record_type, record_id in rows:
            available[RecordType(record_type)].add(record_id)
        for record in records:
            available[RecordType(record.record_type)].add(record_identity(record))
        return available

    def _latest_record_revisions(
        self, programme_id: str
    ) -> dict[tuple[RecordType, str], int]:
        rows = self._connection.execute(
            "SELECT record_type, record_id, MAX(record_revision) "
            "FROM research_state_records WHERE programme_id = ? "
            "GROUP BY record_type, record_id",
            (programme_id,),
        ).fetchall()
        return {
            (RecordType(kind), record_id): revision
            for kind, record_id, revision in rows
        }

    def _latest_state_revision(self, programme_id: str) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(state_revision), 0) FROM research_state_revisions "
            "WHERE programme_id = ?",
            (programme_id,),
        ).fetchone()
        return int(row[0])

    def _validate_references(
        self,
        records: tuple[ResearchStateRecord, ...],
        available: dict[RecordType, set[str]],
    ) -> None:
        def require(ids, record_type: RecordType, message: str) -> None:
            if not set(ids).issubset(available[record_type]):
                raise ValueError(message)

        def require_evidence(refs: EvidenceReferences) -> None:
            require(
                refs.external_evidence_card_ids,
                RecordType.EXTERNAL_EVIDENCE,
                "external_evidence_card_reference_missing",
            )
            require(
                refs.internal_observation_ids,
                RecordType.INTERNAL_OBSERVATION,
                "internal_observation_reference_missing",
            )
            require(
                refs.diagnostic_observation_ids,
                RecordType.DIAGNOSTIC_OBSERVATION,
                "diagnostic_observation_reference_missing",
            )

        for record in records:
            if isinstance(record, ResearchHypothesis):
                require_evidence(record.motivating_evidence)
                require_evidence(record.supporting_evidence)
                require_evidence(record.refuting_evidence)
                require(
                    record.origin_inference_ids,
                    RecordType.PLANNER_INFERENCE,
                    "hypothesis_origin_inference_reference_missing",
                )
                require(
                    record.competing_hypothesis_ids,
                    RecordType.HYPOTHESIS,
                    "competing_hypothesis_reference_missing",
                )
            elif isinstance(record, ResearchUncertainty):
                require_evidence(record.affects_evidence)
                require_evidence(record.resolved_by_evidence)
                require(
                    record.affects_hypothesis_ids,
                    RecordType.HYPOTHESIS,
                    "uncertainty_hypothesis_reference_missing",
                )
                require(
                    record.affects_decision_ids,
                    RecordType.PLANNER_DECISION,
                    "uncertainty_decision_reference_missing",
                )
                if record.superseded_by_uncertainty_id:
                    require(
                        (record.superseded_by_uncertainty_id,),
                        RecordType.UNCERTAINTY,
                        "successor_uncertainty_reference_missing",
                    )
            elif isinstance(record, PlannerInference):
                require_evidence(record.derived_from_evidence)
                require(
                    record.hypothesis_ids,
                    RecordType.HYPOTHESIS,
                    "inference_hypothesis_reference_missing",
                )
                require(
                    record.uncertainty_ids,
                    RecordType.UNCERTAINTY,
                    "inference_uncertainty_reference_missing",
                )
            elif isinstance(record, PlannerDecision):
                require_evidence(record.supporting_evidence)
                require(
                    record.inference_ids,
                    RecordType.PLANNER_INFERENCE,
                    "decision_inference_reference_missing",
                )
                require(
                    record.hypothesis_ids,
                    RecordType.HYPOTHESIS,
                    "decision_hypothesis_reference_missing",
                )
                require(
                    record.uncertainty_ids,
                    RecordType.UNCERTAINTY,
                    "decision_uncertainty_reference_missing",
                )
                require(
                    record.experiment_ids,
                    RecordType.EXPERIMENT,
                    "decision_experiment_reference_missing",
                )
            elif isinstance(record, ResearchExperiment):
                require_evidence(record.intent.motivated_by_evidence)
                require(
                    record.intent.hypothesis_ids,
                    RecordType.HYPOTHESIS,
                    "experiment_hypothesis_reference_missing",
                )
                require(
                    record.observation_ids,
                    RecordType.INTERNAL_OBSERVATION,
                    "experiment_observation_reference_missing",
                )
            elif isinstance(record, CandidateNextAction):
                require_evidence(record.motivated_by_evidence)
                require(
                    record.hypothesis_ids,
                    RecordType.HYPOTHESIS,
                    "next_action_hypothesis_reference_missing",
                )
                require(
                    record.competing_hypothesis_ids,
                    RecordType.HYPOTHESIS,
                    "next_action_competing_hypothesis_reference_missing",
                )
                require(
                    record.uncertainty_ids,
                    RecordType.UNCERTAINTY,
                    "next_action_uncertainty_reference_missing",
                )

    def _validate_revision_invariants(self, record: ResearchStateRecord) -> None:
        """Prevent a later revision from rewriting the original research intent."""

        if record.revision == 1:
            return
        record_type = RecordType(record.record_type)
        original = self.get_record(
            record.programme_id,
            record_type,
            record_identity(record),
            revision=1,
        )
        if original is None:
            raise ValueError("research_state_original_revision_missing")
        invariant_fields: tuple[str, ...]
        if isinstance(record, ResearchHypothesis):
            invariant_fields = (
                "hypothesis_id",
                "proposition",
                "origin",
                "motivation",
                "motivating_evidence",
                "origin_inference_ids",
                "mixed_origins",
            )
        elif isinstance(record, ResearchUncertainty):
            invariant_fields = ("uncertainty_id", "question")
        elif isinstance(record, ResearchExperiment):
            invariant_fields = (
                "experiment_id",
                "experiment_spec_reference",
                "intent",
            )
        elif isinstance(record, ResearchWorkItem):
            invariant_fields = (
                "work_item_id",
                "description",
                "reference_type",
                "reference_id",
            )
        elif isinstance(record, CandidateNextAction):
            invariant_fields = (
                "next_action_id",
                "action",
                "rationale",
                "hypothesis_ids",
                "competing_hypothesis_ids",
                "uncertainty_ids",
                "motivated_by_evidence",
                "expected_information_value",
            )
        else:
            return
        changed = tuple(
            field
            for field in invariant_fields
            if getattr(record, field) != getattr(original, field)
        )
        if changed:
            raise ValueError(
                "research_state_revision_invariant_violation:" + ",".join(changed)
            )

    def close(self) -> None:
        self._connection.close()
