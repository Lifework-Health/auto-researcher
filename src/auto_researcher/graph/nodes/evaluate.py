"""Evaluator invocation and budget accounting."""

from auto_researcher.contracts.enums import SearchType
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def evaluate_experiment(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    experiment = state["experiment_spec"]
    assert experiment is not None
    result = dependencies.evaluator.evaluate(experiment, state["contract"])
    cost = float(getattr(dependencies.evaluator, "cost_per_experiment", 0.0))
    budget = state["budget"].record_experiment(cost)
    request = state.get("search_request")
    is_optuna = request is not None and request.search_type == SearchType.OPTUNA
    errors = (
        []
        if result.success or is_optuna
        else [result.error or "evaluation_failed"]
    )
    return {
        "evaluation_result": result,
        "budget": budget,
        "errors": errors,
        "executed_nodes": ["evaluate_experiment"],
    }
