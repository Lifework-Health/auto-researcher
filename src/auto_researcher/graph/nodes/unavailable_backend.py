"""Clean termination for a requested backend that is absent or disallowed."""

from auto_researcher.contracts.enums import RunStatus
from auto_researcher.graph.state import ResearchState


def unavailable_backend(state: ResearchState) -> dict:
    result = state["search_backend_result"]
    assert result is not None
    return {
        "status": RunStatus.STOPPED,
        "stop_reason": f"{result.code.lower()}:{result.requested_type.value}",
        "executed_nodes": ["unavailable_backend"],
    }
