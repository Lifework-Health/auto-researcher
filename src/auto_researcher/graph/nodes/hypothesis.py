"""Hypothesis agent invocation."""

from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def generate_hypothesis(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    hypothesis = dependencies.hypothesis_agent.generate(
        state["contract"],
        cycle=state["cycle"],
    )
    return {
        "active_hypothesis": hypothesis,
        "executed_nodes": ["generate_hypothesis"],
    }
