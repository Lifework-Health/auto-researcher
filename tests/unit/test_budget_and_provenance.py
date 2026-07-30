from __future__ import annotations

from datetime import UTC, datetime

import pytest

from auto_researcher.contracts.enums import EventType, ProvenanceKind, RunStatus
from auto_researcher.contracts.models import BudgetState, DecisionEvent
from auto_researcher.graph.builder import build_graph
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore


def _event(event_id: str, event_type: EventType = EventType.SEARCH_PLANNED):
    return DecisionEvent(
        event_id=event_id,
        run_id="run-1",
        cycle=1,
        event_type=event_type,
        actor="test",
        rationale="test event",
        timestamp=datetime(2026, 7, 30, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.MOCK,
    )


def test_budget_state_marks_exhaustion():
    budget = BudgetState(maximum_cycles=0, maximum_experiments=1, maximum_cost=1)
    exhausted = budget.before_cycle()
    assert exhausted.exhausted is True
    assert exhausted.exhaustion_reason == "maximum_cycles_reached"


def test_budget_stops_graph_before_agents(
    contract_factory,
    deterministic_dependencies,
):
    contract = contract_factory(maximum_cycles=0)
    graph = build_graph(deterministic_dependencies)
    final = graph.invoke(
        {"run_id": "budget-run", "thread_id": "budget-thread", "contract": contract},
        {"configurable": {"thread_id": "budget-thread"}},
    )
    assert final["status"] == RunStatus.COMPLETED
    assert final["stop_reason"] == "maximum_cycles_reached"
    assert "generate_hypothesis" not in final["executed_nodes"]


def test_provenance_events_are_append_only():
    store = SQLiteProvenanceStore()
    store.append_event(_event("event-1"))
    with pytest.raises(ValueError, match="immutable"):
        store.append_event(_event("event-1", EventType.EVIDENCE_VERIFIED))
    assert store.get_event("event-1").event_type == EventType.SEARCH_PLANNED
    assert not hasattr(store, "update_event")
    assert not hasattr(store, "delete_event")


def test_provenance_queries_preserve_insert_order():
    store = SQLiteProvenanceStore()
    store.append_event(_event("event-1", EventType.SEARCH_PLANNED))
    store.append_event(_event("event-2", EventType.EVIDENCE_VERIFIED))
    assert [event.event_id for event in store.list_events("run-1")] == [
        "event-1",
        "event-2",
    ]
    assert [
        event.event_id
        for event in store.list_events_by_type("run-1", EventType.EVIDENCE_VERIFIED)
    ] == ["event-2"]


def test_model_and_evaluator_costs_combine_without_hiding_retry_calls():
    budget = BudgetState(
        maximum_cycles=1,
        maximum_experiments=2,
        maximum_cost=10,
    )
    budget = budget.record_model_usage(
        calls=2,
        input_tokens=100,
        output_tokens=50,
        cache_creation_tokens=10,
        cache_read_tokens=5,
        cost=0.25,
    )
    budget = budget.record_experiment(0.75)
    assert budget.model_calls_used == 2
    assert budget.model_cache_creation_tokens_used == 10
    assert budget.model_cache_read_tokens_used == 5
    assert budget.model_cost_used == 0.25
    assert budget.evaluator_cost_used == 0.75
    assert budget.cost_used == 1.0
