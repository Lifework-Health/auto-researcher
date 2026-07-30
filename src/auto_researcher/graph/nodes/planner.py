"""Planner agent invocation with structural output reconciliation."""

from auto_researcher.agents.live.base import LiveAgentExecutionError
from auto_researcher.agents.telemetry import (
    apply_agent_telemetry,
    consume_agent_telemetry,
)
from auto_researcher.contracts.enums import RunStatus
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def plan_search(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    hypothesis = state["active_hypothesis"]
    if state["status"] == RunStatus.FAILED or hypothesis is None:
        return {
            "search_request": None,
            "executed_nodes": ["plan_search"],
        }
    try:
        context = dependencies.agent_context_assembler.planner_context(
            state,
            dependencies.task_agent_context,
            dependencies.search_capabilities,
        )
        request = dependencies.planner_agent.plan(context)
    except Exception as exc:
        telemetry = consume_agent_telemetry(dependencies.planner_agent)
        code = (
            exc.code
            if isinstance(exc, LiveAgentExecutionError)
            else "planner_agent_failed"
        )
        return {
            "status": RunStatus.FAILED,
            "budget": apply_agent_telemetry(state["budget"], telemetry),
            "search_request": None,
            "errors": [code],
            "stop_reason": "planner_agent_failed",
            "executed_nodes": ["plan_search"],
        }
    telemetry = consume_agent_telemetry(dependencies.planner_agent)
    errors: list[str] = []
    if request.hypothesis_id != hypothesis.hypothesis_id:
        errors.append("planner_hypothesis_mismatch")
    if request.experiment_budget > state["contract"].maximum_experiments:
        errors.append("planner_budget_exceeds_contract")
    return {
        "search_request": request,
        "budget": apply_agent_telemetry(state["budget"], telemetry),
        "errors": errors,
        "executed_nodes": ["plan_search"],
    }
