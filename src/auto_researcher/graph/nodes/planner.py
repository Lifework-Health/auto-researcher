"""Planner agent invocation with structural output reconciliation."""

from auto_researcher.agents.live.base import LiveAgentExecutionError
from auto_researcher.agents.telemetry import (
    apply_agent_telemetry,
    consume_agent_telemetry,
)
from auto_researcher.contracts.enums import RunStatus, SearchType
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies
from auto_researcher.tasks.protocols import (
    CampaignDurationCapableTask,
    CampaignRequestEnrichmentCapableTask,
)


def plan_search(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    hypothesis = state["active_hypothesis"]
    if state["status"] != RunStatus.RUNNING or hypothesis is None:
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
        if isinstance(dependencies.task, CampaignRequestEnrichmentCapableTask):
            request = dependencies.task.enrich_search_request(
                request,
                context.prior_verified_findings,
            )
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
    if request.search_type == SearchType.OPENEVOLVE:
        capability = dependencies.search_capabilities[SearchType.OPENEVOLVE]
        if not capability.available or dependencies.openevolve_backend is None:
            errors.append("planner_openevolve_task_unsupported")
        else:
            try:
                dependencies.openevolve_backend.create_search_contract(
                    request,
                    state["contract"],
                )
            except ValueError as exc:
                errors.append(str(exc))
    deadline_stop_reason = None
    remaining_time = state["budget"].remaining_seconds(dependencies.clock())
    if remaining_time is not None and isinstance(
        dependencies.task, CampaignDurationCapableTask
    ):
        try:
            estimated = dependencies.task.estimate_search_duration_seconds(
                request, dependencies.runtime_context
            )
            raw_reserve = dependencies.runtime_context.task_options.get(
                "campaign_finalisation_reserve_seconds",
                state["contract"].constraints.get(
                    "campaign_finalisation_reserve_seconds", 0
                ),
            )
            if (
                isinstance(raw_reserve, bool)
                or not isinstance(raw_reserve, (int, float))
                or raw_reserve < 0
            ):
                raise ValueError("campaign_finalisation_reserve_invalid")
            reserve = float(raw_reserve)
            if estimated + reserve > remaining_time:
                deadline_stop_reason = "campaign_insufficient_time_for_proposed_block"
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    update = {
        "search_request": request,
        "budget": apply_agent_telemetry(state["budget"], telemetry),
        "errors": errors,
        "executed_nodes": ["plan_search"],
    }
    if telemetry is not None and telemetry.cost_limit_exceeded:
        update.update(
            status=RunStatus.STOPPED,
            stop_reason="maximum_agent_call_cost_exceeded",
        )
    elif deadline_stop_reason is not None:
        update.update(
            status=RunStatus.COMPLETED,
            stop_reason=deadline_stop_reason,
        )
    elif errors and request.search_type == SearchType.OPENEVOLVE:
        update.update(
            status=RunStatus.FAILED,
            stop_reason="invalid_openevolve_plan",
        )
    return update
