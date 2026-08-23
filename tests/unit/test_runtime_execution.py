from __future__ import annotations

from dataclasses import replace

import pytest

from auto_researcher.contracts.enums import RunStatus, SearchType
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.execution import (
    EXECUTION_ERROR_VOCABULARY_VERSION,
    RunExecutionError,
    can_resume_recoverable_planner_failure,
    inspect_terminal_run,
    resume_run,
    start_run,
)


def test_execution_error_vocabulary_is_public_and_versioned():
    assert EXECUTION_ERROR_VOCABULARY_VERSION == "run-execution-errors-v1"


def _input(contract, *, run_id="run-1", thread_id="thread-1", **extra):
    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "contract": contract,
        **extra,
    }


def _config(thread_id="thread-1"):
    return {"configurable": {"thread_id": thread_id}}


def test_start_requires_a_fresh_thread_and_inspect_is_read_only(
    contract_factory,
    deterministic_dependencies,
):
    graph = build_graph(deterministic_dependencies)
    initial = _input(contract_factory())
    final = start_run(graph, initial, _config())
    assert final["status"] == RunStatus.COMPLETED
    event_ids = [
        item.event_id
        for item in deterministic_dependencies.provenance_store.list_events("run-1")
    ]

    with pytest.raises(
        RunExecutionError,
        match="thread_already_exists_use_resume_or_inspect",
    ):
        start_run(graph, initial, _config())

    inspected = inspect_terminal_run(graph, _config())
    assert inspected == graph.get_state(_config()).values
    assert [
        item.event_id
        for item in deterministic_dependencies.provenance_store.list_events("run-1")
    ] == event_ids


def test_start_rejects_an_existing_non_terminal_thread_and_resume_continues(
    contract_factory,
    deterministic_dependencies,
):
    graph = build_graph(
        deterministic_dependencies,
        interrupt_after=["plan_search"],
    )
    initial = _input(contract_factory())
    paused = start_run(graph, initial, _config())
    assert paused["status"] == RunStatus.RUNNING

    with pytest.raises(RunExecutionError, match="thread_already_exists"):
        start_run(graph, initial, _config())

    conflicts = [
        (
            _input(contract_factory(), run_id="different-run"),
            "conflicting_run_identity",
        ),
        (
            _input(contract_factory().model_copy(update={"question": "different"})),
            "conflicting_contract_identity",
        ),
        (
            _input(contract_factory().model_copy(update={"task_id": "different-task"})),
            "conflicting_task_identity",
        ),
        (
            {**initial, "operator_request": "different"},
            "conflicting_initial_input_identity",
        ),
    ]
    for conflicting, code in conflicts:
        with pytest.raises(RunExecutionError, match=code):
            start_run(graph, conflicting, _config())

    resumed_graph = build_graph(deterministic_dependencies)
    final = resume_run(resumed_graph, _config())
    assert final["status"] == RunStatus.COMPLETED


def test_resume_unknown_and_terminal_threads_are_rejected(
    contract_factory,
    deterministic_dependencies,
):
    graph = build_graph(deterministic_dependencies)
    with pytest.raises(RunExecutionError, match="thread_not_found"):
        resume_run(graph, _config())
    start_run(graph, _input(contract_factory()), _config())
    with pytest.raises(RunExecutionError, match="thread_is_terminal_use_inspect"):
        resume_run(graph, _config())


def test_resume_recovers_exact_legacy_planner_failure_without_new_hypothesis(
    contract_factory,
    deterministic_dependencies,
):
    paused_graph = build_graph(
        deterministic_dependencies,
        interrupt_after=["generate_hypothesis"],
    )
    state = start_run(paused_graph, _input(contract_factory()), _config())
    hypothesis_id = state["active_hypothesis"].hypothesis_id
    paused_graph.update_state(
        _config(),
        {
            "status": RunStatus.FAILED,
            "stop_reason": "planner_agent_failed",
            "search_request": None,
            "errors": ["planner_agent_failed"],
            "executed_nodes": ["plan_search"],
        },
        as_node="plan_search",
    )

    final = resume_run(build_graph(deterministic_dependencies), _config())

    assert final.get("recovered_error_codes") == ["planner_agent_failed"], final.get(
        "recovered_error_codes"
    )
    assert final["status"] == RunStatus.COMPLETED, final.get("stop_reason")
    assert final["active_hypothesis"].hypothesis_id == hypothesis_id
    assert final["recovered_error_codes"] == ["planner_agent_failed"]
    assert final["last_executed_search_type"] == SearchType.DIRECT


def test_research_directive_projection_failure_is_narrowly_recoverable():
    values = {
        "status": RunStatus.FAILED,
        "stop_reason": "research_director_openevolve_context_invalid",
        "errors": ["research_director_openevolve_context_invalid"],
        "planner_failure_stage": "research_directive_projection",
        "active_research_directive": object(),
        "active_hypothesis": object(),
        "search_request": None,
        "executed_nodes": ["plan_search"],
    }

    assert can_resume_recoverable_planner_failure(values) is True
    assert (
        can_resume_recoverable_planner_failure(
            {**values, "planner_failure_stage": "portfolio_policy"}
        )
        is False
    )
    assert (
        can_resume_recoverable_planner_failure(
            {**values, "active_research_directive": None}
        )
        is False
    )


@pytest.mark.parametrize(
    ("changed", "code"),
    [
        ({"run_id": "run-2"}, "conflicting_run_identity"),
        ({"operator_request": "changed"}, "conflicting_initial_input_identity"),
    ],
)
def test_existing_thread_rejects_conflicting_identity(
    contract_factory,
    deterministic_dependencies,
    changed,
    code,
):
    graph = build_graph(deterministic_dependencies)
    contract = contract_factory()
    original = _input(contract, operator_request="original")
    start_run(graph, original, _config())
    conflicting = {**original, **changed}
    with pytest.raises(RunExecutionError, match=code):
        start_run(graph, conflicting, _config())


def test_existing_thread_rejects_conflicting_contract_and_task(
    contract_factory,
    deterministic_dependencies,
):
    graph = build_graph(deterministic_dependencies)
    contract = contract_factory()
    initial = _input(contract)
    start_run(graph, initial, _config())

    changed_contract = contract.model_copy(update={"question": "Changed question"})
    with pytest.raises(RunExecutionError, match="conflicting_contract_identity"):
        start_run(graph, _input(changed_contract), _config())

    changed_task = contract.model_copy(update={"task_id": "different-task"})
    with pytest.raises(RunExecutionError, match="conflicting_task_identity"):
        start_run(graph, _input(changed_task), _config())


def test_checkpoint_03_operator_error_is_harmless_before_any_side_effect(
    contract_factory,
    deterministic_dependencies,
):
    class CountingHypothesisAgent:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return self.inner.generate(*args, **kwargs)

    class CountingPlannerAgent:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def plan(self, *args, **kwargs):
            self.calls += 1
            return self.inner.plan(*args, **kwargs)

    class CountingEvaluator:
        def __init__(self, inner):
            self.inner = inner
            self.evaluator_id = inner.evaluator_id
            self.version = inner.version
            self.cost_per_experiment = inner.cost_per_experiment
            self.calls = 0

        def evaluate(self, experiment, contract):
            self.calls += 1
            return self.inner.evaluate(experiment, contract)

    class CountingVerifier:
        def __init__(self, inner):
            self.inner = inner
            self.verifier_id = inner.verifier_id
            self.version = inner.version
            self.calls = 0

        def verify(self, *args, **kwargs):
            self.calls += 1
            return self.inner.verify(*args, **kwargs)

    evaluator = CountingEvaluator(deterministic_dependencies.evaluator)
    verifier = CountingVerifier(deterministic_dependencies.verifier)
    hypothesis = CountingHypothesisAgent(deterministic_dependencies.hypothesis_agent)
    planner = CountingPlannerAgent(deterministic_dependencies.planner_agent)
    dependencies = replace(
        deterministic_dependencies,
        evaluator=evaluator,
        verifier=verifier,
        hypothesis_agent=hypothesis,
        planner_agent=planner,
    )
    graph = build_graph(dependencies)
    initial = _input(contract_factory())
    first = start_run(graph, initial, _config())
    event_ids = [
        item.event_id for item in dependencies.provenance_store.list_events("run-1")
    ]
    assert (hypothesis.calls, planner.calls, evaluator.calls, verifier.calls) == (
        1,
        1,
        1,
        1,
    )

    with pytest.raises(RunExecutionError, match="thread_already_exists"):
        start_run(graph, initial, _config())

    inspected = inspect_terminal_run(graph, _config())
    assert inspected == first
    assert (hypothesis.calls, planner.calls, evaluator.calls, verifier.calls) == (
        1,
        1,
        1,
        1,
    )
    assert [
        item.event_id for item in dependencies.provenance_store.list_events("run-1")
    ] == event_ids
