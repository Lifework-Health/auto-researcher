"""Deterministic lifecycle preparation owned by the supervisor."""

from auto_researcher.contracts.enums import RunStatus
from auto_researcher.graph.state import ResearchState


def supervisor_prepare(state: ResearchState) -> dict:
    if state["status"] != RunStatus.RUNNING:
        return {"executed_nodes": ["supervisor_prepare"]}
    if state["errors"]:
        return {
            "status": RunStatus.FAILED,
            "stop_reason": "fatal_error",
            "executed_nodes": ["supervisor_prepare"],
        }
    budget = state["budget"].before_cycle()
    if budget.exhausted:
        return {
            "budget": budget,
            "status": RunStatus.COMPLETED,
            "stop_reason": budget.exhaustion_reason,
            "executed_nodes": ["supervisor_prepare"],
        }
    return {
        "budget": budget,
        "cycle": budget.cycles_used,
        "active_hypothesis": None,
        "search_request": None,
        "search_backend_result": None,
        "experiment_spec": None,
        "evaluation_result": None,
        "verification_result": None,
        "pending_human_request": None,
        "human_approval_granted": None,
        "executed_nodes": ["supervisor_prepare"],
    }
