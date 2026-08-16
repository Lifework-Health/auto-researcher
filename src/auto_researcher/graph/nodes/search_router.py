"""Installed-backend and contract-allowlist enforcement."""

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import SearchBackendResult
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def search_router(
    state: ResearchState,
    dependencies: RuntimeDependencies | None = None,
) -> dict:
    request = state["search_request"]
    assert request is not None
    if request.search_type not in state["contract"].allowed_search_types:
        result = SearchBackendResult(
            requested_type=request.search_type,
            available=False,
            code="SEARCH_TYPE_NOT_ALLOWED",
            message=f"{request.search_type.value} is not allowed by the research contract",
        )
    elif dependencies is None and request.search_type == SearchType.DIRECT:
        result = SearchBackendResult(
            requested_type=request.search_type,
            available=True,
            code="BACKEND_AVAILABLE",
            message="DIRECT backend selected",
        )
    elif dependencies is None:
        result = SearchBackendResult(
            requested_type=request.search_type,
            available=False,
            code="BACKEND_UNAVAILABLE",
            message=f"{request.search_type.value} is unavailable",
        )
    else:
        capability = dependencies.search_backend_registry.capability(
            request.search_type
        )
        result = SearchBackendResult(
            requested_type=request.search_type,
            available=capability.available,
            code=capability.code,
            message=capability.message,
        )
    update = {
        "search_backend_result": result,
        "last_executed_search_type": request.search_type,
        "executed_nodes": ["search_router"],
    }
    if request.search_type == SearchType.OPENEVOLVE and dependencies is not None:
        update["openevolve_native_complete"] = (
            False if dependencies.native_openevolve_runtime is not None else None
        )
    return update
