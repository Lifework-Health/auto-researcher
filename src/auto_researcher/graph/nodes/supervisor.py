"""Deterministic lifecycle preparation owned by the supervisor."""

from auto_researcher.contracts.enums import RunStatus
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def supervisor_prepare(
    state: ResearchState, dependencies: RuntimeDependencies | None = None
) -> dict:
    if state["status"] != RunStatus.RUNNING:
        return {"executed_nodes": ["supervisor_prepare"]}
    recovered = set(state.get("recovered_error_codes", ()))
    if any(code not in recovered for code in state["errors"]):
        return {
            "status": RunStatus.FAILED,
            "stop_reason": "fatal_error",
            "executed_nodes": ["supervisor_prepare"],
        }
    budget = state["budget"].before_cycle(
        dependencies.clock() if dependencies is not None else None
    )
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
        "optuna_study_spec": None,
        "optuna_study_state": None,
        "optuna_study_result": None,
        "optuna_trial_outcome": None,
        "diagnostic_experiment_spec": None,
        "diagnostic_evaluation_result": None,
        "diagnostic_verification_result": None,
        "knowledge_bundle_reference": None,
        "planner_failure_code": None,
        "planner_failure_stage": None,
        "planner_fallback_code": None,
        "hypothesis_failure_code": None,
        "hypothesis_failure_stage": None,
        "hypothesis_fallback_code": None,
        "executed_nodes": ["supervisor_prepare"],
    }
