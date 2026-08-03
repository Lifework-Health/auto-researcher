"""The deliberately mutation-free provenance storage interface."""

from typing import Protocol, runtime_checkable

from auto_researcher.contracts.enums import EventType
from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.provenance.reuse import (
    EvaluationReuseRecord,
    VerificationReuseRecord,
)


@runtime_checkable
class ProvenanceStore(Protocol):
    def append_event(self, event: DecisionEvent) -> None: ...

    def append_event_idempotent(self, event: DecisionEvent) -> bool: ...

    def append_semantic_event(
        self,
        event: DecisionEvent,
        semantic_key: str,
        payload_hash: str,
    ) -> tuple[DecisionEvent, bool]: ...

    def get_event(self, event_id: str) -> DecisionEvent | None: ...

    def list_events(self, run_id: str) -> list[DecisionEvent]: ...

    def list_events_by_type(
        self,
        run_id: str,
        event_type: EventType,
    ) -> list[DecisionEvent]: ...

    def get_evaluation_reuse(
        self, run_id: str, experiment_id: str
    ) -> EvaluationReuseRecord | None: ...

    def append_evaluation_reuse(self, record: EvaluationReuseRecord) -> None: ...

    def get_verification_reuse(
        self, run_id: str, experiment_id: str
    ) -> VerificationReuseRecord | None: ...

    def append_verification_reuse(self, record: VerificationReuseRecord) -> None: ...
