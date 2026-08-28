"""Run initialisation."""

from datetime import timedelta

from auto_researcher.contracts.enums import RunStatus
from auto_researcher.contracts.models import BudgetState
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def initialise_run(
    state: ResearchState, dependencies: RuntimeDependencies | None = None
) -> dict:
    contract = state["contract"]
    now = dependencies.clock() if dependencies is not None else None
    raw_duration = contract.constraints.get("campaign_duration_seconds")
    duration = (
        float(raw_duration)
        if isinstance(raw_duration, (int, float)) and not isinstance(raw_duration, bool)
        else None
    )
    if duration is not None and duration <= 0:
        duration = None
    deadline = (
        now + timedelta(seconds=duration)
        if now is not None and duration is not None
        else None
    )
    if not state.get("run_id") or not state.get("thread_id"):
        return {
            "status": RunStatus.FAILED,
            "cycle": 0,
            "budget": BudgetState(
                maximum_cycles=contract.maximum_cycles,
                maximum_experiments=contract.maximum_experiments,
                maximum_cost=contract.maximum_cost,
                started_at=now,
                deadline_at=deadline,
            ),
            "decision_event_ids": [],
            "errors": ["run_id_and_thread_id_are_required"],
            "stop_reason": "invalid_initial_state",
            "executed_nodes": ["initialise_run"],
        }
    return {
        "status": RunStatus.RUNNING,
        "cycle": 0,
        "budget": BudgetState(
            maximum_cycles=contract.maximum_cycles,
            maximum_experiments=contract.maximum_experiments,
            maximum_cost=contract.maximum_cost,
            started_at=now,
            deadline_at=deadline,
        ),
        "decision_event_ids": [],
        "errors": [],
        "knowledge_errors": [],
        "knowledge_warnings": [],
        "knowledge_bundle_reference": None,
        "executed_nodes": ["initialise_run"],
        "stop_reason": None,
    }
