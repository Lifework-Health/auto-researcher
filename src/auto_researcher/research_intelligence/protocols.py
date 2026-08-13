"""Interfaces for offline-first Research Intelligence."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from auto_researcher.research_intelligence.models import (
    EvidenceCard,
    EvidenceQuery,
    ResearchIntelligenceRefreshRecord,
    RetrievedSourceMaterial,
    SourceRecord,
    SynthesisResult,
)


@runtime_checkable
class ResearchScout(Protocol):
    """Boundary by which already-retrieved material enters the subsystem."""

    scout_id: str
    scout_version: str

    def collect(self) -> tuple[RetrievedSourceMaterial, ...]: ...


@runtime_checkable
class EvidenceStore(Protocol):
    def store_synthesis(
        self, result: SynthesisResult
    ) -> ResearchIntelligenceRefreshRecord: ...

    def get_source(
        self, source_id: str, source_version_id: str | None = None
    ) -> SourceRecord | None: ...

    def get_evidence(self, evidence_id: str) -> EvidenceCard | None: ...

    def query_evidence(self, query: EvidenceQuery) -> tuple[EvidenceCard, ...]: ...

    def strongest_known_baselines(
        self, query: EvidenceQuery, *, limit: int = 10
    ) -> tuple[EvidenceCard, ...]: ...

    def known_failure_modes(
        self, query: EvidenceQuery, *, limit: int = 10
    ) -> tuple[EvidenceCard, ...]: ...

    def conflicting_evidence(
        self, query: EvidenceQuery
    ) -> tuple[EvidenceCard, ...]: ...

    def evidence_newer_than(self, timestamp: datetime) -> tuple[EvidenceCard, ...]: ...

    def sources_supporting_claim(self, claim_key: str) -> tuple[SourceRecord, ...]: ...

    def latest_refresh(self) -> ResearchIntelligenceRefreshRecord | None: ...
