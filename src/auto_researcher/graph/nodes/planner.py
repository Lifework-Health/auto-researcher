"""Planner agent invocation with structural output reconciliation."""

import hashlib

from auto_researcher.agents.context import AgentContextAssemblyError
from auto_researcher.agents.live.base import LiveAgentExecutionError
from auto_researcher.agents.models import ResearchDirective
from auto_researcher.agents.telemetry import (
    apply_agent_telemetry,
    consume_agent_telemetry,
)
from auto_researcher.contracts.enums import (
    EventType,
    ProposalSource,
    RunStatus,
    SearchType,
)
from auto_researcher.contracts.models import SearchRequest
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies
from auto_researcher.search.openevolve.live_boundary import (
    assert_no_prohibited_dynamic_content,
)
from auto_researcher.tasks.protocols import (
    CampaignDurationCapableTask,
    CampaignDeadlinePortfolioCapableTask,
    CampaignPortfolioCapableTask,
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


def _duplicates_verified_direct_configuration(
    state: ResearchState,
    dependencies: RuntimeDependencies,
    request: SearchRequest,
) -> bool:
    if request.search_type != SearchType.DIRECT:
        return False
    try:
        proposed = dependencies.task.normalise_configuration(
            dict(request.search_space)
        )
    except (TypeError, ValueError):
        return False
    for event in dependencies.provenance_store.list_events(state["run_id"]):
        if event.event_type != EventType.EVIDENCE_VERIFIED:
            continue
        outputs = set(event.output_references)
        if not {
            "evidence:SUPPORTED",
            "verified:true",
            "constraints:true",
        }.issubset(outputs):
            continue
        raw = event.safe_payload.get("configuration")
        if not isinstance(raw, dict):
            continue
        try:
            existing = dependencies.task.normalise_configuration(dict(raw))
        except (TypeError, ValueError):
            continue
        if existing == proposed:
            return True
    return False


def _apply_research_directive(
    request: SearchRequest,
    state: ResearchState,
) -> SearchRequest:
    """Bind the active directive to planning and metadata-only mutation context."""

    directive = state.get("active_research_directive")
    if directive is None:
        return request
    if not isinstance(directive, ResearchDirective):
        try:
            directive = ResearchDirective.model_validate(directive)
        except (TypeError, ValueError) as exc:
            raise ValueError("research_director_openevolve_context_invalid") from exc
    reference = f"research-directive:{directive.directive_id}"
    evidence_references = tuple(dict.fromkeys((*request.evidence_references, reference)))
    if request.search_type != SearchType.OPENEVOLVE:
        return SearchRequest.model_validate(
            {
                **request.model_dump(mode="python"),
                "evidence_references": evidence_references,
            }
        )
    search_space = dict(request.search_space)
    raw_context = search_space.get("campaign_context", {})
    if not isinstance(raw_context, dict):
        raise ValueError("research_director_openevolve_context_invalid")
    safe_evidence_references: list[str] = []
    for evidence_reference in directive.evidence_references:
        try:
            assert_no_prohibited_dynamic_content(evidence_reference)
        except ValueError:
            continue
        safe_evidence_references.append(evidence_reference)
    projection_candidates = {
        "directive_id": directive.directive_id,
        "trigger": directive.trigger,
        "mechanism_hypothesis": directive.mechanism_hypothesis,
        "selected_operators": [item.value for item in directive.selected_operators],
        "targeted_dimensions": list(directive.targeted_dimensions),
        "expected_observation": directive.expected_observation,
        "falsification_condition": directive.falsification_condition,
        "evidence_references": safe_evidence_references,
        "confidence": directive.confidence,
    }
    projection: dict[str, object] = {}
    for name, value in projection_candidates.items():
        try:
            assert_no_prohibited_dynamic_content(value)
        except ValueError:
            # The complete directive remains durably available to the
            # controller and planner.  OpenEvolve receives only individual
            # fields that satisfy its stricter metadata-only boundary.
            continue
        projection[name] = value
    if projection.get("directive_id") != directive.directive_id:
        raise ValueError("research_director_openevolve_context_invalid")
    try:
        assert_no_prohibited_dynamic_content(projection)
    except ValueError as exc:
        raise ValueError("research_director_openevolve_context_invalid") from exc
    search_space["campaign_context"] = {
        **raw_context,
        "research_directive": projection,
    }
    return SearchRequest.model_validate(
        {
            **request.model_dump(mode="python"),
            "search_space": search_space,
            "evidence_references": evidence_references,
        }
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
    fallback_code: str | None = None
    fallback_stage: str | None = None
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
            request = fallback
            fallback_code = code
            fallback_stage = stage
        else:
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
    else:
        telemetry = consume_agent_telemetry(dependencies.planner_agent)
    if isinstance(dependencies.task, CampaignPortfolioCapableTask):
        stage = "portfolio_policy"
        try:
            request = dependencies.task.apply_campaign_portfolio(
                request,
                run_id=state["run_id"],
                cycle=state["cycle"],
                events=tuple(
                    dependencies.provenance_store.list_events(state["run_id"])
                ),
                runtime_context=dependencies.runtime_context,
            )
        except Exception as exc:
            code = _safe_failure_code(exc)
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
        if request is None:
            return {
                "status": RunStatus.COMPLETED,
                "budget": apply_agent_telemetry(state["budget"], telemetry),
                "search_request": None,
                "errors": [],
                "stop_reason": "campaign_portfolio_complete",
                "executed_nodes": ["plan_search"],
            }
    try:
        request = _apply_research_directive(request, state)
    except ValueError as exc:
        code = str(exc)
        return {
            "status": RunStatus.FAILED,
            "budget": apply_agent_telemetry(state["budget"], telemetry),
            "search_request": None,
            "errors": [code],
            "stop_reason": code,
            "planner_failure_code": code,
            "planner_failure_stage": "research_directive_projection",
            "executed_nodes": ["plan_search"],
        }
    if fallback_code is not None and _duplicates_verified_direct_configuration(
        state, dependencies, request
    ):
        code = "planner_fallback_duplicate_configuration"
        return {
            "status": RunStatus.FAILED,
            "budget": apply_agent_telemetry(state["budget"], telemetry),
            "search_request": None,
            "errors": [code],
            "stop_reason": code,
            "planner_failure_code": code,
            "planner_failure_stage": "fallback_deduplication",
            "planner_fallback_code": fallback_code,
            "executed_nodes": ["plan_search"],
        }
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
    if errors and fallback_code is None:
        fallback = _deterministic_direct_fallback(
            state,
            dependencies,
            failure_code="planner_request_invalid",
        )
        if fallback is not None:
            request = fallback
            errors = []
            fallback_code = "planner_request_invalid"
            fallback_stage = "request_validation"
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
                if isinstance(
                    dependencies.task, CampaignDeadlinePortfolioCapableTask
                ):
                    completion = dependencies.task.apply_campaign_deadline_policy(
                        request,
                        run_id=state["run_id"],
                        cycle=state["cycle"],
                        events=tuple(
                            dependencies.provenance_store.list_events(state["run_id"])
                        ),
                        remaining_seconds=remaining_time,
                        runtime_context=dependencies.runtime_context,
                    )
                    if completion is not None:
                        completion_estimate = (
                            dependencies.task.estimate_search_duration_seconds(
                                completion, dependencies.runtime_context
                            )
                        )
                        reporting_reserve = dependencies.runtime_context.task_options.get(
                            "campaign_reporting_reserve_seconds", reserve
                        )
                        if (
                            isinstance(reporting_reserve, bool)
                            or not isinstance(reporting_reserve, (int, float))
                            or reporting_reserve < 0
                        ):
                            raise ValueError("campaign_reporting_reserve_invalid")
                        if completion_estimate + float(reporting_reserve) <= remaining_time:
                            request = completion
                            deadline_stop_reason = None
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    update = {
        "search_request": request,
        "budget": apply_agent_telemetry(state["budget"], telemetry),
        "errors": errors,
        "planner_failure_code": None,
        "planner_failure_stage": fallback_stage,
        "planner_fallback_code": fallback_code,
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
