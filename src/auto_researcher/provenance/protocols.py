"""The deliberately mutation-free provenance storage interface."""

from typing import Protocol, runtime_checkable

from auto_researcher.contracts.enums import EventType
from auto_researcher.contracts.models import DecisionEvent


@runtime_checkable
class ProvenanceStore(Protocol):
    def append_event(self, event: DecisionEvent) -> None: ...

    def get_event(self, event_id: str) -> DecisionEvent | None: ...

    def list_events(self, run_id: str) -> list[DecisionEvent]: ...

    def list_events_by_type(
        self,
        run_id: str,
        event_type: EventType,
    ) -> list[DecisionEvent]: ...
