"""Planner agent invocation with structural output reconciliation."""

import hashlib

from auto_researcher.agents.context import AgentContextAssemblyError
from auto_researcher.agents.live.base import LiveAgentExecutionError
from auto_researcher.agents.telemetry import (
    apply_agent_telemetry,
    consume_agent_telemetry,
)
from auto_researcher.contracts.enums import ProposalSource, RunStatus, SearchType
from auto_researcher.contracts.models import SearchRequest
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies
from auto_researcher.tasks.protocols import (
    CampaignDurationCapableTask,
    CampaignRequestEnrichmentCapableTask,
)


def _safe_failure_code(exc: Exception) -> str:
    if isinstance(exc, (LiveAgentExecutionError, AgentContextAssemblyError)):
        return exc.code
    return "planner_agent_failed"


def _deterministic_direct_fallback(
    state: ResearchState,
    dependencies: RuntimeDependencies,
    *,
    failure_code: str,
) -> SearchRequest | None:
    """Execute an exact, valid hypothesis configuration when planning fails."""

    hypothesis = state["active_hypothesis"]
    if (
        hypothesis is None
        or SearchType.DIRECT not in state["contract"].allowed_search_types
        or SearchType.DIRECT
        not in dependencies.task_agent_context.available_search_types
        or dependencies.search_capabilities.get(SearchType.DIRECT) is None
        or not dependencies.search_capabilities[SearchType.DIRECT].available
        or state["budget"].experiments_used >= state["budget"].maximum_experiments
    ):
        return None
    try:
        configuration = dependencies.task.normalise_configuration(
            dict(hypothesis.predicted_subspace)
        )
    except (TypeError, ValueError):
        return None
    digest = hashlib.sha256(
        (
            state["run_id"]
            + "\x1f"
            + str(state["cycle"])
            + "\x1f"
            + hypothesis.hypothesis_id
            + "\x1f"
            + failure_code
        ).encode()
    ).hexdigest()[:20]
    return SearchRequest(
        request_id=f"search-fallback-{digest}",
        hypothesis_id=hypothesis.hypothesis_id,
        search_type=SearchType.DIRECT,
        target=state["contract"].primary_metric,
        search_space=configuration,
        experiment_budget=1,
        rationale=(
            "Deterministic DIRECT fallback from the active hypothesis after the "
            f"planner boundary returned {failure_code}."
        ),
        evidence_references=hypothesis.evidence_references,
        requires_human_approval=(
            SearchType.DIRECT in state["contract"].requires_approval_for
        ),
        proposal_source=ProposalSource.DETERMINISTIC,
        grounding_status=hypothesis.grounding_status,
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
    stage = "context_assembly"
    try:
        context = dependencies.agent_context_assembler.planner_context(
            state,
            dependencies.task_agent_context,
            dependencies.search_capabilities,
        )
        stage = "model_call"
        request = dependencies.planner_agent.plan(context)
        if isinstance(dependencies.task, CampaignRequestEnrichmentCapableTask):
            stage = "request_enrichment"
            request = dependencies.task.enrich_search_request(
                request,
                context.prior_verified_findings,
            )
    except Exception as exc:
        telemetry = consume_agent_telemetry(dependencies.planner_agent)
        code = _safe_failure_code(exc)
        fallback = (
            _deterministic_direct_fallback(
                state,
                dependencies,
                failure_code=code,
            )
            if isinstance(exc, (LiveAgentExecutionError, AgentContextAssemblyError))
            else None
        )
        if fallback is not None:
            return {
                "budget": apply_agent_telemetry(state["budget"], telemetry),
                "search_request": fallback,
                "planner_fallback_code": code,
                "planner_failure_stage": stage,
                "executed_nodes": ["plan_search"],
            }
        return {
            "status": RunStatus.FAILED,
            "budget": apply_agent_telemetry(state["budget"], telemetry),
            "search_request": None,
            "errors": [code],
            "stop_reason": code,
            "planner_failure_code": code,
            "planner_failure_stage": stage,
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
        "planner_failure_code": None,
        "planner_failure_stage": None,
        "planner_fallback_code": None,
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
