from __future__ import annotations

import json
from datetime import UTC, datetime

from langgraph.types import Command

from auto_researcher.agents.mock import MockPlannerAgent
from auto_researcher.contracts.enums import EventType, RunStatus, SearchType
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import memory_dependencies, sqlite_dependencies


class SequenceIds:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, prefix: str) -> str:
        self.value += 1
        return f"{prefix}-{self.value:04d}"


def fixed_clock() -> datetime:
    return datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


def _invoke(graph, contract, run_id="run-1", thread_id="thread-1"):
    return graph.invoke(
        {"run_id": run_id, "thread_id": thread_id, "contract": contract},
        {"configurable": {"thread_id": thread_id}},
    )


def test_complete_one_cycle_and_every_required_direct_node_executes(
    contract_factory,
    deterministic_dependencies,
):
    graph = build_graph(deterministic_dependencies)
    final = _invoke(graph, contract_factory())
    assert final["status"] == RunStatus.COMPLETED
    assert final["stop_reason"] == "maximum_cycles_reached"
    required = {
        "initialise_run",
        "supervisor_prepare",
        "generate_hypothesis",
        "plan_search",
        "approval_router",
        "search_router",
        "direct_search",
        "evaluate_experiment",
        "verify_evidence",
        "record_provenance",
        "supervisor_decide",
    }
    assert required.issubset(final["executed_nodes"])


def test_invalid_initial_state_fails_cleanly_without_provenance(
    contract_factory,
    deterministic_dependencies,
):
    graph = build_graph(deterministic_dependencies)
    final = _invoke(
        graph,
        contract_factory(),
        run_id="",
        thread_id="invalid-thread",
    )
    assert final["status"] == RunStatus.FAILED
    assert final["stop_reason"] == "invalid_initial_state"
    assert final["cycle"] == 0
    assert final["errors"] == ["run_id_and_thread_id_are_required"]
    assert final["executed_nodes"] == ["initialise_run"]
    assert deterministic_dependencies.provenance_store.list_events("") == []


def test_verifier_runs_automatically_and_provenance_is_complete(
    contract_factory,
    deterministic_dependencies,
):
    graph = build_graph(deterministic_dependencies)
    final = _invoke(graph, contract_factory(), run_id="provenance-run")
    assert final["verification_result"] is not None
    assert "verify_evidence" in final["executed_nodes"]
    events = deterministic_dependencies.provenance_store.list_events("provenance-run")
    assert [event.event_type for event in events] == [
        EventType.HYPOTHESIS_PROPOSED,
        EventType.SEARCH_PLANNED,
        EventType.EXPERIMENT_PREPARED,
        EventType.EVALUATION_OBSERVED,
        EventType.EVIDENCE_VERIFIED,
    ]
    assert final["decision_event_ids"] == [event.event_id for event in events]


def test_graph_reconstructs_and_resumes_same_sqlite_thread(contract_factory, tmp_path):
    checkpoint = tmp_path / "checkpoints.sqlite"
    provenance = tmp_path / "provenance.sqlite"
    config = {"configurable": {"thread_id": "resume-thread"}}
    initial = {
        "run_id": "resume-run",
        "thread_id": "resume-thread",
        "contract": contract_factory(),
    }
    ids = SequenceIds()
    with sqlite_dependencies(
        checkpoint,
        provenance,
        clock=fixed_clock,
        id_generator=ids,
    ) as first_dependencies:
        first_graph = build_graph(first_dependencies, interrupt_after=["plan_search"])
        paused = first_graph.invoke(initial, config)
        assert "plan_search" in paused["executed_nodes"]
        assert "verify_evidence" not in paused["executed_nodes"]

    with sqlite_dependencies(
        checkpoint,
        provenance,
        clock=fixed_clock,
        id_generator=ids,
    ) as second_dependencies:
        second_graph = build_graph(second_dependencies)
        final = second_graph.invoke(None, config)
        assert final["status"] == RunStatus.COMPLETED
        assert "verify_evidence" in final["executed_nodes"]
        assert len(second_dependencies.provenance_store.list_events("resume-run")) == 5


def test_rejected_human_approval_terminates_cleanly(
    contract_factory,
    deterministic_dependencies,
):
    contract = contract_factory(approval=frozenset({SearchType.DIRECT}))
    graph = build_graph(deterministic_dependencies)
    config = {"configurable": {"thread_id": "approval-thread"}}
    paused = graph.invoke(
        {
            "run_id": "approval-run",
            "thread_id": "approval-thread",
            "contract": contract,
        },
        config,
    )
    interrupt_payload = paused["__interrupt__"][0].value
    assert json.loads(json.dumps(interrupt_payload))["search_type"] == "DIRECT"
    final = graph.invoke(Command(resume={"approved": False}), config)
    assert final["status"] == RunStatus.STOPPED
    assert final["stop_reason"] == "human_rejected"
    assert final["experiment_spec"] is None
    assert EventType.HUMAN_DECISION in {
        event.event_type
        for event in deterministic_dependencies.provenance_store.list_events("approval-run")
    }


def test_unavailable_backend_terminates_structurally(contract_factory):
    dependencies = memory_dependencies(
        planner_agent=MockPlannerAgent(SearchType.OPTUNA),
        clock=fixed_clock,
        id_generator=SequenceIds(),
    )
    contract = contract_factory(allowed=frozenset({SearchType.OPTUNA}))
    final = _invoke(
        build_graph(dependencies),
        contract,
        run_id="optuna-run",
        thread_id="optuna-thread",
    )
    assert final["status"] == RunStatus.STOPPED
    assert final["stop_reason"] == "backend_unavailable:OPTUNA"
    assert final["search_backend_result"].code == "BACKEND_UNAVAILABLE"
    assert final["experiment_spec"] is None
    assert final["errors"] == []
