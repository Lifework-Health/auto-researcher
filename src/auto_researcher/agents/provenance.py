"""Replay-safe, prompt-free provenance for durable model-call snapshots."""

from __future__ import annotations

import hashlib

from auto_researcher.agents.call_store import AgentCallStore
from auto_researcher.contracts.enums import (
    AgentCallStatus,
    EventType,
    ProvenanceKind,
)
from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.provenance.protocols import ProvenanceStore

CODE_VERSION = "auto-researcher-v2.1-pr5"


def _event_type(status: AgentCallStatus) -> EventType:
    if status == AgentCallStatus.RESERVED:
        return EventType.MODEL_CALL_RESERVED
    if status == AgentCallStatus.COMPLETED:
        return EventType.MODEL_CALL_COMPLETED
    return EventType.MODEL_CALL_FAILED


def append_model_call_events(
    provenance_store: ProvenanceStore,
    call_store: AgentCallStore,
    *,
    run_id: str,
    cycle: int,
) -> list[str]:
    event_ids: list[str] = []
    for record in call_store.list_records(run_id):
        if record.cycle != cycle:
            continue
        event_type = _event_type(record.status)
        digest = hashlib.sha256(record.record_id.encode()).hexdigest()[:24]
        event_id = f"event-model-{digest}"
        outputs = (
            f"status:{record.status.value}",
            f"provider:{record.provider}",
            f"model:{record.model_id}",
            f"prompt:{record.prompt_name}@{record.prompt_version}",
            f"prompt_hash:{record.prompt_hash}",
            f"context_hash:{record.context_hash}",
            f"response_hash:{record.response_hash or 'none'}",
            f"input_tokens:{record.input_tokens}",
            f"output_tokens:{record.output_tokens}",
            f"cache_creation_tokens:{record.cache_creation_input_tokens}",
            f"cache_read_tokens:{record.cache_read_input_tokens}",
            f"cost:{record.estimated_cost}",
            f"currency:{record.pricing_currency}",
            f"pricing_version:{record.pricing_version}",
            (
                "pricing_rates:"
                f"{record.pricing.input_cost_per_million_tokens}/"
                f"{record.pricing.output_cost_per_million_tokens}/"
                f"{record.pricing.cache_write_cost_per_million_tokens}/"
                f"{record.pricing.cache_read_cost_per_million_tokens}"
            ),
        )
        event = DecisionEvent(
            event_id=event_id,
            run_id=run_id,
            cycle=cycle,
            event_type=event_type,
            actor="structured_model_client",
            input_references=(record.call_id,),
            output_references=outputs,
            rationale=(
                f"{record.role.value.lower()} model call "
                f"{record.status.value.lower()} under bounded policy"
            ),
            timestamp=record.created_at,
            code_version=CODE_VERSION,
            provenance=ProvenanceKind.MOCK,
        )
        append = getattr(provenance_store, "append_event_idempotent", None)
        if append is not None:
            append(event)
        else:
            existing = provenance_store.get_event(event_id)
            if existing is None:
                provenance_store.append_event(event)
            elif existing != event:
                raise ValueError(f"conflicting model call event {event_id!r}")
        event_ids.append(event_id)
    return event_ids
