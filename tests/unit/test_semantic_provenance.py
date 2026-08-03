from __future__ import annotations

from datetime import UTC, datetime

import pytest

from auto_researcher.contracts.enums import EventType, ProvenanceKind
from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore


def _event(event_id: str, event_type: EventType, output: str) -> DecisionEvent:
    return DecisionEvent(
        event_id=event_id,
        run_id="run-1",
        cycle=1,
        event_type=event_type,
        actor="test",
        output_references=(output,),
        rationale="semantic identity test",
        timestamp=datetime(2026, 8, 3, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
    )


@pytest.mark.parametrize(
    "event_type",
    [
        EventType.HYPOTHESIS_PROPOSED,
        EventType.SEARCH_PLANNED,
        EventType.EXPERIMENT_PREPARED,
        EventType.EVALUATION_OBSERVED,
        EventType.EVIDENCE_VERIFIED,
    ],
)
def test_identical_lifecycle_semantic_event_is_reused(event_type):
    store = SQLiteProvenanceStore()
    first, inserted = store.append_semantic_event(
        _event("event-1", event_type, "same"),
        f"semantic:{event_type.value}",
        "a" * 64,
    )
    second, inserted_again = store.append_semantic_event(
        _event("event-2", event_type, "same"),
        f"semantic:{event_type.value}",
        "a" * 64,
    )
    assert inserted is True
    assert inserted_again is False
    assert second.event_id == first.event_id
    assert len(store.list_events("run-1")) == 1


@pytest.mark.parametrize(
    "event_type",
    [EventType.EVALUATION_OBSERVED, EventType.EVIDENCE_VERIFIED],
)
def test_conflicting_scientific_payload_fails_closed(event_type):
    store = SQLiteProvenanceStore()
    store.append_semantic_event(
        _event("event-1", event_type, "first"),
        f"semantic:{event_type.value}",
        "a" * 64,
    )
    with pytest.raises(ValueError, match="conflicting_semantic_provenance_event"):
        store.append_semantic_event(
            _event("event-2", event_type, "changed"),
            f"semantic:{event_type.value}",
            "b" * 64,
        )
