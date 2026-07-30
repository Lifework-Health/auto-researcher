"""DIRECT backend graph node."""

from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def direct_search(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    request = state["search_request"]
    assert request is not None
    experiment = dependencies.direct_search_backend.create_experiment(
        request,
        state["contract"],
        run_id=state["run_id"],
    )
    return {
        "experiment_spec": experiment,
        "executed_nodes": ["direct_search"],
    }
