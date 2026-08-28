"""Deterministic, provenance-driven Research Director call cadence."""

from __future__ import annotations

from collections.abc import Iterable

from auto_researcher.contracts.enums import EventType
from auto_researcher.contracts.models import DecisionEvent

SCHEDULED_FIDELITIES = (10, 15, 25, 50, 100)
MAXIMUM_DIRECTOR_CALLS = 8
MAXIMUM_ANOMALY_CALLS = 2


def _maximum_epochs(event: DecisionEvent) -> int | None:
    configuration = event.safe_payload.get("configuration")
    if not isinstance(configuration, dict):
        return None
    value = configuration.get("maximum_epochs")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _score(event: DecisionEvent) -> float | None:
    for reference in event.output_references:
        if not reference.startswith("score:"):
            continue
        try:
            return float(reference.split(":", 1)[1])
        except ValueError:
            return None
    return None


def next_research_director_trigger(
    events: Iterable[DecisionEvent],
    history: tuple[str, ...],
    *,
    explicit_trigger: str | None = None,
) -> str | None:
    """Return one new trigger without depending on process-local state."""

    if len(history) >= MAXIMUM_DIRECTOR_CALLS:
        return None
    seen = set(history)
    if explicit_trigger and explicit_trigger not in seen:
        return explicit_trigger
    if "campaign_start" not in seen:
        return "campaign_start"

    verified = [event for event in events if event.event_type == EventType.EVIDENCE_VERIFIED]
    supported = [
        event
        for event in verified
        if {
            "evidence:SUPPORTED",
            "verified:true",
            "constraints:true",
        }.issubset(event.output_references)
    ]
    observed_fidelities = {_maximum_epochs(event) for event in supported}
    for fidelity in SCHEDULED_FIDELITIES:
        trigger = f"first_verified_{fidelity}ep"
        if fidelity in observed_fidelities and trigger not in seen:
            return trigger

    anomaly_history = sum(
        trigger.startswith(("verification_anomaly:", "score_stall:"))
        for trigger in history
    )
    if anomaly_history >= MAXIMUM_ANOMALY_CALLS:
        return None
    anomalous = [
        event
        for event in verified
        if "verified:false" in event.output_references
        or "constraints:false" in event.output_references
    ]
    if anomalous:
        trigger = f"verification_anomaly:{anomalous[-1].event_id}"
        if trigger not in seen:
            return trigger

    scored = [(event, _score(event)) for event in supported]
    scored = [(event, score) for event, score in scored if score is not None]
    if len(scored) >= 6:
        recent = scored[-6:]
        early_best = max(score for _, score in recent[:3])
        late_best = max(score for _, score in recent[3:])
        if late_best <= early_best + 0.001:
            trigger = f"score_stall:{recent[-1][0].event_id}"
            if trigger not in seen:
                return trigger
    return None
