"""Deterministic continue/stop adjudication."""

from auto_researcher.contracts.enums import RunStatus
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def supervisor_decide(
    state: ResearchState, dependencies: RuntimeDependencies | None = None
) -> dict:
    update: dict = {"executed_nodes": ["supervisor_decide"]}
    if state["status"] in {RunStatus.STOPPED, RunStatus.FAILED}:
        return update
    recovered = set(state.get("recovered_error_codes", ()))
    if any(code not in recovered for code in state["errors"]):
        update.update(status=RunStatus.FAILED, stop_reason="fatal_error")
        return update

    budget = state["budget"]
    now = dependencies.clock() if dependencies is not None else None
    if budget.deadline_at is not None and now is not None and now >= budget.deadline_at:
        update.update(
            status=RunStatus.COMPLETED, stop_reason="campaign_deadline_reached"
        )
        return update
    if budget.cycles_used >= budget.maximum_cycles:
        update.update(status=RunStatus.COMPLETED, stop_reason="maximum_cycles_reached")
        return update
    if budget.experiments_used >= budget.maximum_experiments:
        update.update(
            status=RunStatus.COMPLETED, stop_reason="maximum_experiments_reached"
        )
        return update
    if budget.cost_used >= budget.maximum_cost:
        update.update(status=RunStatus.COMPLETED, stop_reason="maximum_cost_reached")
        return update

    next_budget = budget.before_cycle(now)
    if next_budget.exhausted:
        update.update(
            budget=next_budget,
            status=RunStatus.COMPLETED,
            stop_reason=next_budget.exhaustion_reason,
        )
        return update
    update.update(
        budget=next_budget,
        cycle=next_budget.cycles_used,
        status=RunStatus.RUNNING,
        active_hypothesis=None,
        search_request=None,
        search_backend_result=None,
        experiment_spec=None,
        evaluation_result=None,
        verification_result=None,
        pending_human_request=None,
        human_approval_granted=None,
        optuna_study_spec=None,
        optuna_study_state=None,
        optuna_study_result=None,
        optuna_trial_outcome=None,
        diagnostic_experiment_spec=None,
        diagnostic_evaluation_result=None,
        diagnostic_verification_result=None,
        knowledge_bundle_reference=None,
        planner_failure_code=None,
        planner_failure_stage=None,
        planner_fallback_code=None,
        hypothesis_failure_code=None,
        hypothesis_failure_stage=None,
        hypothesis_fallback_code=None,
        stop_reason=None,
    )
    return update
