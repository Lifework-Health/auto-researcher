"""Semantic, replay-safe provenance for OpenEvolve lifecycle boundaries."""

from __future__ import annotations

from datetime import datetime

from auto_researcher.contracts.enums import EventType, ProvenanceKind
from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.provenance.protocols import ProvenanceStore
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.models import OPENEVOLVE_PROVENANCE_VERSION

CODE_VERSION = "auto-researcher-v2.1-pr6"


def append_openevolve_event(
    store: ProvenanceStore,
    *,
    run_id: str,
    cycle: int,
    search_request_id: str,
    event_type: EventType,
    actor: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    rationale: str,
    timestamp: datetime,
    provenance: ProvenanceKind,
    semantic_identity: str,
    scientific_payload,
) -> str:
    semantic_key = payload_hash(
        {
            "schema": OPENEVOLVE_PROVENANCE_VERSION,
            "run_id": run_id,
            "search_request_id": search_request_id,
            "event_type": event_type.value,
            "identity": semantic_identity,
        }
    )
    event_id = f"event-{semantic_key[:24]}"
    event = DecisionEvent(
        event_id=event_id,
        run_id=run_id,
        cycle=cycle,
        event_type=event_type,
        actor=actor,
        input_references=inputs,
        output_references=outputs,
        rationale=rationale,
        timestamp=timestamp,
        code_version=CODE_VERSION,
        provenance=provenance,
    )
    append = getattr(store, "append_semantic_event", None)
    if append is not None:
        persisted, _ = append(event, semantic_key, payload_hash(scientific_payload))
        return persisted.event_id
    existing = store.get_event(event_id)
    if existing is None:
        store.append_event(event)
    elif payload_hash(existing) != payload_hash(event):
        raise ValueError("conflicting_semantic_provenance_event")
    return event_id
