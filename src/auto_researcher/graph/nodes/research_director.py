"""Checkpointed Research Director decision boundary."""

from auto_researcher.agents.context import AgentContextAssemblyError
from auto_researcher.agents.live.base import LiveAgentExecutionError
from auto_researcher.agents.models import ResearchDirective
from auto_researcher.agents.telemetry import (
    apply_agent_telemetry,
    consume_agent_telemetry,
)
from auto_researcher.agents.research_director_policy import (
    next_research_director_trigger,
)
from auto_researcher.contracts.enums import RunStatus
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def research_director_decide(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    agent = dependencies.research_director_agent
    active = state.get("active_research_directive")
    if agent is None:
        return {"executed_nodes": ["research_director_decide"]}
    if active is not None and not isinstance(active, ResearchDirective):
        try:
            active = ResearchDirective.model_validate(active)
        except (TypeError, ValueError):
            return {
                "status": RunStatus.FAILED,
                "errors": ["research_director_checkpoint_invalid"],
                "stop_reason": "research_director_checkpoint_invalid",
                "research_director_failure_code": (
                    "research_director_checkpoint_invalid"
                ),
                "research_director_failure_stage": "checkpoint_restore",
                "executed_nodes": ["research_director_decide"],
            }
    history = tuple(state.get("research_director_trigger_history", ()))
    trigger = next_research_director_trigger(
        dependencies.provenance_store.list_events(state["run_id"]),
        history,
        explicit_trigger=state.get("research_director_trigger"),
    )
    if trigger is None:
        return {"executed_nodes": ["research_director_decide"]}
    if active is not None and active.trigger == trigger:
        return {
            "research_director_trigger_history": (*history, trigger),
            "research_director_trigger": None,
            "executed_nodes": ["research_director_decide"],
        }

    raw_reserve = dependencies.runtime_context.task_options.get(
        "campaign_finalisation_reserve_seconds",
        state["contract"].constraints.get("campaign_finalisation_reserve_seconds", 0),
    )
    if (
        isinstance(raw_reserve, bool)
        or not isinstance(raw_reserve, (int, float))
        or raw_reserve < 0
    ):
        return {
            "status": RunStatus.FAILED,
            "errors": ["research_director_finalisation_reserve_invalid"],
            "stop_reason": "research_director_finalisation_reserve_invalid",
            "research_director_failure_code": (
                "research_director_finalisation_reserve_invalid"
            ),
            "research_director_failure_stage": "reserve_validation",
            "executed_nodes": ["research_director_decide"],
        }
    reserve = float(raw_reserve)
    remaining = state["budget"].remaining_seconds(dependencies.clock())
    if remaining is not None and remaining <= reserve:
        return {"executed_nodes": ["research_director_decide"]}

    stage = "context_assembly"
    try:
        context = dependencies.agent_context_assembler.research_director_context(
            state,
            dependencies.task_agent_context,
            dependencies.search_capabilities,
            trigger=trigger,
            finalisation_reserve_seconds=reserve,
        )
        stage = "model_call"
        directive = agent.decide(context)
    except Exception as exc:
        telemetry = consume_agent_telemetry(agent)
        code = (
            exc.code
            if isinstance(exc, (LiveAgentExecutionError, AgentContextAssemblyError))
            else "research_director_failed"
        )
        if active is not None:
            return {
                "budget": apply_agent_telemetry(state["budget"], telemetry),
                "research_director_failure_code": code,
                "research_director_failure_stage": stage,
                "executed_nodes": ["research_director_decide"],
            }
        return {
            "status": RunStatus.FAILED,
            "budget": apply_agent_telemetry(state["budget"], telemetry),
            "active_research_directive": None,
            "errors": [code],
            "stop_reason": code,
            "research_director_failure_code": code,
            "research_director_failure_stage": stage,
            "executed_nodes": ["research_director_decide"],
        }
    telemetry = consume_agent_telemetry(agent)
    return {
        "active_research_directive": directive,
        "research_director_trigger_history": (*history, trigger),
        "research_director_trigger": None,
        "budget": apply_agent_telemetry(state["budget"], telemetry),
        "research_director_failure_code": None,
        "research_director_failure_stage": None,
        "executed_nodes": ["research_director_decide"],
    }
