"""Installed-backend and contract-allowlist enforcement."""

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import SearchBackendResult
from auto_researcher.graph.state import ResearchState


def search_router(state: ResearchState) -> dict:
    request = state["search_request"]
    assert request is not None
    if request.search_type not in state["contract"].allowed_search_types:
        result = SearchBackendResult(
            requested_type=request.search_type,
            available=False,
            code="SEARCH_TYPE_NOT_ALLOWED",
            message=f"{request.search_type.value} is not allowed by the research contract",
        )
    elif request.search_type == SearchType.DIRECT:
        result = SearchBackendResult(
            requested_type=request.search_type,
            available=True,
            code="BACKEND_AVAILABLE",
            message="DIRECT backend selected",
        )
    else:
        result = SearchBackendResult(
            requested_type=request.search_type,
            available=False,
            code="BACKEND_UNAVAILABLE",
            message=f"{request.search_type.value} is declared but not installed in PR 2",
        )
    return {
        "search_backend_result": result,
        "executed_nodes": ["search_router"],
    }
