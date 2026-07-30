"""Mandatory automatic verifier invocation."""

from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def verify_evidence(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    experiment = state["experiment_spec"]
    evaluation = state["evaluation_result"]
    assert experiment is not None and evaluation is not None
    verification = dependencies.verifier.verify(
        experiment,
        evaluation,
        state["contract"],
        claimed_score=evaluation.primary_score,
    )
    return {
        "verification_result": verification,
        "executed_nodes": ["verify_evidence"],
    }
