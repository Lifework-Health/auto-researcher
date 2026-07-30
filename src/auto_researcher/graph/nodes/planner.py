"""Planner agent invocation with structural output reconciliation."""

from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def plan_search(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    hypothesis = state["active_hypothesis"]
    assert hypothesis is not None
    request = dependencies.planner_agent.plan(
        state["contract"],
        hypothesis,
        cycle=state["cycle"],
    )
    errors: list[str] = []
    if request.hypothesis_id != hypothesis.hypothesis_id:
        errors.append("planner_hypothesis_mismatch")
    if request.experiment_budget > state["contract"].maximum_experiments:
        errors.append("planner_budget_exceeds_contract")
    return {
        "search_request": request,
        "errors": errors,
        "executed_nodes": ["plan_search"],
    }
