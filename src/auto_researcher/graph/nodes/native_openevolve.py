"""Standard graph node for the embedded native OpenEvolve controller."""

from auto_researcher.contracts.enums import RunStatus
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies
from auto_researcher.search.openevolve.native_engine import NativeEvolutionResult


def run_native_openevolve(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    runtime = dependencies.native_openevolve_runtime
    request = state.get("search_request")
    if runtime is None or request is None:
        raise RuntimeError("native_openevolve_runtime_unavailable")
    result = runtime.run_search(request)
    previous = state.get("openevolve_native_result")
    if isinstance(previous, dict):
        previous = NativeEvolutionResult.model_validate(previous)
    previous_evaluations = 0 if previous is None else previous.expensive_evaluations
    budget = state["budget"]
    for _ in range(max(0, result.expensive_evaluations - previous_evaluations)):
        budget = budget.record_experiment(
            float(getattr(dependencies.evaluator, "cost_per_experiment", 0.0))
        )
    update = {
        "openevolve_native_result": result,
        "openevolve_native_complete": result.finished,
        "budget": budget,
        "executed_nodes": ["run_native_openevolve"],
    }
    if result.finished:
        update.update(
            status=RunStatus.COMPLETED,
            stop_reason="native_openevolve_completed",
        )
    return update
