"""Hypothesis agent invocation."""

from auto_researcher.agents.live.base import LiveAgentExecutionError
from auto_researcher.agents.telemetry import (
    apply_agent_telemetry,
    consume_agent_telemetry,
)
from auto_researcher.contracts.enums import RunStatus
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def generate_hypothesis(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    try:
        context = dependencies.agent_context_assembler.hypothesis_context(
            state,
            dependencies.task_agent_context,
        )
        hypothesis = dependencies.hypothesis_agent.generate(context)
    except Exception as exc:
        telemetry = consume_agent_telemetry(dependencies.hypothesis_agent)
        code = (
            exc.code
            if isinstance(exc, LiveAgentExecutionError)
            else "hypothesis_agent_failed"
        )
        return {
            "status": RunStatus.FAILED,
            "budget": apply_agent_telemetry(state["budget"], telemetry),
            "active_hypothesis": None,
            "errors": [code],
            "stop_reason": "hypothesis_agent_failed",
            "executed_nodes": ["generate_hypothesis"],
        }
    telemetry = consume_agent_telemetry(dependencies.hypothesis_agent)
    return {
        "active_hypothesis": hypothesis,
        "budget": apply_agent_telemetry(state["budget"], telemetry),
        "executed_nodes": ["generate_hypothesis"],
    }
