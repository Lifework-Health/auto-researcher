"""Protocols for durable Research State implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from auto_researcher.research_state.models import (
    ResearchProgramme,
    ResearchState,
    ResearchStateRecord,
    StateRevision,
)


@runtime_checkable
class ResearchStateStore(Protocol):
    def create_programme(self, programme: ResearchProgramme) -> ResearchProgramme: ...

    def append(self, record: ResearchStateRecord) -> StateRevision: ...

    def append_many(
        self, records: tuple[ResearchStateRecord, ...] | list[ResearchStateRecord]
    ) -> tuple[StateRevision, ...]: ...

    def load_state(self, programme_id: str) -> ResearchState: ...
