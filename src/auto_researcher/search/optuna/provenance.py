"""Deterministic, replay-safe provenance helpers for Optuna lifecycle nodes."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from auto_researcher.contracts.enums import EventType, ProvenanceKind
from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.provenance.protocols import ProvenanceStore

CODE_VERSION = "auto-researcher-v2.1-pr3"


def deterministic_event_id(
    run_id: str,
    study_name: str,
    event_type: EventType,
    *,
    trial_number: int | None = None,
) -> str:
    parts = [run_id, study_name, event_type.value]
    if trial_number is not None:
        parts.append(str(trial_number))
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]
    return f"event-{digest}"


def append_optuna_event(
    store: ProvenanceStore,
    *,
    run_id: str,
    cycle: int,
    study_name: str,
    event_type: EventType,
    actor: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    rationale: str,
    timestamp: datetime,
    provenance: ProvenanceKind,
    trial_number: int | None = None,
    safe_payload: dict[str, Any] | None = None,
) -> str:
    event = DecisionEvent(
        event_id=deterministic_event_id(
            run_id,
            study_name,
            event_type,
            trial_number=trial_number,
        ),
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
        safe_payload=safe_payload or {},
    )
    append = getattr(store, "append_event_idempotent", None)
    if append is not None:
        append(event)
    else:
        existing = store.get_event(event.event_id)
        if existing is None:
            store.append_event(event)
        elif existing != event:
            raise ValueError(
                f"event {event.event_id!r} already exists with different content"
            )
    return event.event_id
