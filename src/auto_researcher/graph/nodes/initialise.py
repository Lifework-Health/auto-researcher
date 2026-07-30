"""Run initialisation."""

from auto_researcher.contracts.enums import RunStatus
from auto_researcher.contracts.models import BudgetState
from auto_researcher.graph.state import ResearchState


def initialise_run(state: ResearchState) -> dict:
    contract = state["contract"]
    if not state.get("run_id") or not state.get("thread_id"):
        return {
            "status": RunStatus.FAILED,
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
        ),
        "decision_event_ids": [],
        "errors": [],
        "executed_nodes": ["initialise_run"],
        "stop_reason": None,
    }
