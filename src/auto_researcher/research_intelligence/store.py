"""Durable, restart-safe SQLite storage for external evidence."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from auto_researcher.knowledge.identity import content_hash, stable_identifier
from auto_researcher.research_intelligence.models import (
    EvidenceCard,
    EvidenceCategory,
    EvidenceQuery,
    EvidenceRelationType,
    ResearchIntelligenceRefreshRecord,
    SourceRecord,
    SynthesisResult,
)

STORE_SCHEMA_VERSION = "research-intelligence-sqlite-v1"


def _query_matches(card: EvidenceCard, query: EvidenceQuery) -> bool:
    context = card.programme_context
    return (
        (query.task_id is None or context.task_id == query.task_id)
        and (query.domain is None or query.domain in context.domains)
        and (query.dataset_id is None or query.dataset_id in context.dataset_ids)
        and (query.hypothesis_id is None or query.hypothesis_id in card.hypothesis_tags)
        and (not query.categories or card.category in query.categories)
        and card.current_applicability.score >= query.minimum_applicability_score
        and card.ranking.combined_score >= query.minimum_rank_score
    )


class SQLiteEvidenceStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._lock = RLock()
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_intelligence_metadata (
                    name TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_sources (
                    source_version_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_content_hash TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS research_sources_stable_id_idx
                ON research_sources(source_id, retrieved_at);
                CREATE TABLE IF NOT EXISTS research_evidence_cards (
                    evidence_id TEXT PRIMARY KEY,
                    evidence_content_hash TEXT NOT NULL,
                    claim_key TEXT NOT NULL,
                    category TEXT NOT NULL,
                    stance TEXT NOT NULL,
                    synthesised_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS research_evidence_claim_idx
                ON research_evidence_cards(claim_key);
                CREATE INDEX IF NOT EXISTS research_evidence_category_idx
                ON research_evidence_cards(category);
                CREATE TABLE IF NOT EXISTS research_evidence_sources (
                    evidence_id TEXT NOT NULL,
                    source_version_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    PRIMARY KEY(evidence_id, source_version_id, relation_type),
                    FOREIGN KEY(evidence_id) REFERENCES research_evidence_cards(evidence_id),
                    FOREIGN KEY(source_version_id) REFERENCES research_sources(source_version_id)
                );
                CREATE TABLE IF NOT EXISTS research_evidence_relations (
                    evidence_id TEXT NOT NULL,
                    related_evidence_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    PRIMARY KEY(evidence_id, related_evidence_id, relation_type),
                    FOREIGN KEY(evidence_id) REFERENCES research_evidence_cards(evidence_id),
                    FOREIGN KEY(related_evidence_id) REFERENCES research_evidence_cards(evidence_id)
                );
                CREATE TABLE IF NOT EXISTS research_refreshes (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    refresh_id TEXT NOT NULL UNIQUE,
                    completed_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_refresh_evidence (
                    refresh_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    PRIMARY KEY(refresh_id, evidence_id),
                    FOREIGN KEY(refresh_id) REFERENCES research_refreshes(refresh_id),
                    FOREIGN KEY(evidence_id) REFERENCES research_evidence_cards(evidence_id)
                );
                """
            )
            row = self._connection.execute(
                "SELECT value FROM research_intelligence_metadata WHERE name = ?",
                ("schema_version",),
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO research_intelligence_metadata(name, value) "
                    "VALUES (?, ?)",
                    ("schema_version", STORE_SCHEMA_VERSION),
                )
            elif row[0] != STORE_SCHEMA_VERSION:
                raise ValueError("research_intelligence_store_schema_mismatch")

    def store_synthesis(
        self, result: SynthesisResult
    ) -> ResearchIntelligenceRefreshRecord:
        source_ids = tuple(
            sorted(source.source_version_id for source in result.source_records)
        )
        evidence_ids = tuple(sorted(card.evidence_id for card in result.evidence_cards))
        refresh = ResearchIntelligenceRefreshRecord(
            refresh_id=stable_identifier(
                "research-refresh",
                content_hash(result.programme_context),
                *source_ids,
                *evidence_ids,
            ),
            programme_context=result.programme_context,
            source_version_ids=source_ids,
            evidence_card_ids=evidence_ids,
            source_count=len(source_ids),
            evidence_count=len(evidence_ids),
            completed_at=result.synthesised_at,
        )
        with self._lock, self._connection:
            existing_refresh = self._connection.execute(
                "SELECT payload FROM research_refreshes WHERE refresh_id = ?",
                (refresh.refresh_id,),
            ).fetchone()
            if existing_refresh is not None:
                existing = ResearchIntelligenceRefreshRecord.model_validate_json(
                    existing_refresh[0]
                )
                if (
                    existing.programme_context != refresh.programme_context
                    or existing.source_version_ids != refresh.source_version_ids
                    or existing.evidence_card_ids != refresh.evidence_card_ids
                ):
                    raise ValueError("research_intelligence_refresh_conflict")
                return existing

            for source in result.source_records:
                self._insert_source(source)
            for card in result.evidence_cards:
                self._insert_evidence(card)
            available_evidence = {
                row[0]
                for row in self._connection.execute(
                    "SELECT evidence_id FROM research_evidence_cards"
                ).fetchall()
            }
            for card in result.evidence_cards:
                for reference in card.source_references:
                    row = self._connection.execute(
                        "SELECT source_id FROM research_sources "
                        "WHERE source_version_id = ?",
                        (reference.source_version_id,),
                    ).fetchone()
                    if row is None or row[0] != reference.source_id:
                        raise ValueError(
                            "research_intelligence_source_reference_missing"
                        )
                    self._connection.execute(
                        "INSERT OR IGNORE INTO research_evidence_sources"
                        "(evidence_id, source_version_id, relation_type) "
                        "VALUES (?, ?, ?)",
                        (
                            card.evidence_id,
                            reference.source_version_id,
                            EvidenceRelationType.SUPPORTS.value,
                        ),
                    )
                relationships = (
                    (EvidenceRelationType.SUPPORTS, card.supporting_evidence_ids),
                    (EvidenceRelationType.CONFLICTS, card.conflicting_evidence_ids),
                )
                for relation, related_ids in relationships:
                    for related_id in related_ids:
                        if related_id not in available_evidence:
                            raise ValueError(
                                "research_intelligence_evidence_relation_missing"
                            )
                        self._connection.execute(
                            "INSERT OR IGNORE INTO research_evidence_relations"
                            "(evidence_id, related_evidence_id, relation_type) "
                            "VALUES (?, ?, ?)",
                            (card.evidence_id, related_id, relation.value),
                        )
            self._connection.execute(
                "INSERT INTO research_refreshes(refresh_id, completed_at, payload) "
                "VALUES (?, ?, ?)",
                (
                    refresh.refresh_id,
                    refresh.completed_at.astimezone(UTC).isoformat(),
                    refresh.model_dump_json(),
                ),
            )
            self._connection.executemany(
                "INSERT INTO research_refresh_evidence(refresh_id, evidence_id) "
                "VALUES (?, ?)",
                ((refresh.refresh_id, evidence_id) for evidence_id in evidence_ids),
            )
        return refresh

    def _insert_source(self, source: SourceRecord) -> None:
        existing = self._connection.execute(
            "SELECT source_content_hash, retrieved_at FROM research_sources "
            "WHERE source_version_id = ?",
            (source.source_version_id,),
        ).fetchone()
        if existing is not None:
            if existing[0] != source.source_content_hash:
                raise ValueError("research_intelligence_source_immutable_conflict")
            if source.retrieved_at.astimezone(UTC).isoformat() > existing[1]:
                self._connection.execute(
                    "UPDATE research_sources SET retrieved_at = ?, payload = ? "
                    "WHERE source_version_id = ?",
                    (
                        source.retrieved_at.astimezone(UTC).isoformat(),
                        source.model_dump_json(),
                        source.source_version_id,
                    ),
                )
            return
        self._connection.execute(
            "INSERT INTO research_sources"
            "(source_version_id, source_id, source_content_hash, retrieved_at, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                source.source_version_id,
                source.source_id,
                source.source_content_hash,
                source.retrieved_at.astimezone(UTC).isoformat(),
                source.model_dump_json(),
            ),
        )

    def _insert_evidence(self, card: EvidenceCard) -> None:
        existing = self._connection.execute(
            "SELECT evidence_content_hash FROM research_evidence_cards "
            "WHERE evidence_id = ?",
            (card.evidence_id,),
        ).fetchone()
        if existing is not None:
            if existing[0] != card.evidence_content_hash:
                raise ValueError("research_intelligence_evidence_immutable_conflict")
            self._connection.execute(
                "UPDATE research_evidence_cards SET payload = ? WHERE evidence_id = ?",
                (card.model_dump_json(), card.evidence_id),
            )
            return
        self._connection.execute(
            "INSERT INTO research_evidence_cards"
            "(evidence_id, evidence_content_hash, claim_key, category, stance, "
            "synthesised_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                card.evidence_id,
                card.evidence_content_hash,
                card.claim_key,
                card.category.value,
                card.stance.value,
                card.synthesised_at.astimezone(UTC).isoformat(),
                card.model_dump_json(),
            ),
        )

    def get_source(
        self, source_id: str, source_version_id: str | None = None
    ) -> SourceRecord | None:
        if source_version_id is not None:
            row = self._connection.execute(
                "SELECT payload FROM research_sources "
                "WHERE source_id = ? AND source_version_id = ?",
                (source_id, source_version_id),
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT payload FROM research_sources WHERE source_id = ? "
                "ORDER BY retrieved_at DESC, source_version_id DESC LIMIT 1",
                (source_id,),
            ).fetchone()
        return SourceRecord.model_validate_json(row[0]) if row else None

    def get_evidence(self, evidence_id: str) -> EvidenceCard | None:
        row = self._connection.execute(
            "SELECT payload FROM research_evidence_cards WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        return EvidenceCard.model_validate_json(row[0]) if row else None

    def query_evidence(self, query: EvidenceQuery) -> tuple[EvidenceCard, ...]:
        rows = self._connection.execute(
            "SELECT payload FROM research_evidence_cards"
        ).fetchall()
        cards = tuple(EvidenceCard.model_validate_json(row[0]) for row in rows)
        return tuple(
            sorted(
                (card for card in cards if _query_matches(card, query)),
                key=lambda card: (-card.ranking.combined_score, card.evidence_id),
            )
        )

    def strongest_known_baselines(
        self, query: EvidenceQuery, *, limit: int = 10
    ) -> tuple[EvidenceCard, ...]:
        selected = query.model_copy(
            update={"categories": frozenset({EvidenceCategory.STRONG_BASELINE})}
        )
        return self.query_evidence(selected)[:limit]

    def known_failure_modes(
        self, query: EvidenceQuery, *, limit: int = 10
    ) -> tuple[EvidenceCard, ...]:
        selected = query.model_copy(
            update={"categories": frozenset({EvidenceCategory.FAILURE_MODE})}
        )
        return self.query_evidence(selected)[:limit]

    def conflicting_evidence(self, query: EvidenceQuery) -> tuple[EvidenceCard, ...]:
        return tuple(
            card for card in self.query_evidence(query) if card.conflicting_evidence_ids
        )

    def evidence_newer_than(self, timestamp: datetime) -> tuple[EvidenceCard, ...]:
        rows = self._connection.execute(
            "SELECT DISTINCT cards.payload "
            "FROM research_evidence_cards cards "
            "JOIN research_refresh_evidence links "
            "ON links.evidence_id = cards.evidence_id "
            "JOIN research_refreshes refreshes "
            "ON refreshes.refresh_id = links.refresh_id "
            "WHERE refreshes.completed_at > ?",
            (timestamp.astimezone(UTC).isoformat(),),
        ).fetchall()
        return tuple(
            sorted(
                (EvidenceCard.model_validate_json(row[0]) for row in rows),
                key=lambda card: (-card.ranking.combined_score, card.evidence_id),
            )
        )

    def sources_supporting_claim(self, claim_key: str) -> tuple[SourceRecord, ...]:
        rows = self._connection.execute(
            "SELECT DISTINCT sources.payload "
            "FROM research_sources sources "
            "JOIN research_evidence_sources links "
            "ON links.source_version_id = sources.source_version_id "
            "JOIN research_evidence_cards cards "
            "ON cards.evidence_id = links.evidence_id "
            "WHERE cards.claim_key = ? AND cards.stance = ? "
            "AND links.relation_type = ? "
            "ORDER BY sources.source_version_id",
            (claim_key, "SUPPORTS", EvidenceRelationType.SUPPORTS.value),
        ).fetchall()
        return tuple(SourceRecord.model_validate_json(row[0]) for row in rows)

    def latest_refresh(self) -> ResearchIntelligenceRefreshRecord | None:
        row = self._connection.execute(
            "SELECT payload FROM research_refreshes ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        return (
            ResearchIntelligenceRefreshRecord.model_validate_json(row[0])
            if row
            else None
        )

    def close(self) -> None:
        self._connection.close()
