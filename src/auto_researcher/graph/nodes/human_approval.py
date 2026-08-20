"""Replay-safe LangGraph interrupt node for human approval."""

from __future__ import annotations

from langgraph.types import interrupt

from auto_researcher.contracts.enums import RunStatus
from auto_researcher.contracts.models import ApprovalRequest
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def approval_router(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    if state["status"] != RunStatus.RUNNING:
        return {
            "pending_human_request": None,
            "executed_nodes": ["approval_router"],
        }
    recovered = set(state.get("recovered_error_codes", ()))
    if any(code not in recovered for code in state["errors"]):
        return {
            "status": RunStatus.FAILED,
            "stop_reason": "fatal_error",
            "pending_human_request": None,
            "executed_nodes": ["approval_router"],
        }
    request = state["search_request"]
    assert request is not None
    requires = (
        request.requires_human_approval
        or request.search_type in state["contract"].requires_approval_for
    )
    pending = None
    status = state["status"]
    if requires:
        pending = ApprovalRequest(
            request_id=dependencies.id_generator("approval"),
            run_id=state["run_id"],
            cycle=state["cycle"],
            search_request_id=request.request_id,
            search_type=request.search_type,
            target=request.target,
            rationale=request.rationale,
        )
        status = RunStatus.WAITING_FOR_APPROVAL
    return {
        "pending_human_request": pending,
        "status": status,
        "executed_nodes": ["approval_router"],
    }


def human_approval(state: ResearchState) -> dict:
    pending = state["pending_human_request"]
    assert pending is not None
    # interrupt is deliberately the first effect: this node is replay-safe on resume.
    response = interrupt(pending.model_dump(mode="json"))
    approved = response if isinstance(response, bool) else bool(response.get("approved", False))
    update: dict = {
        "human_approval_granted": approved,
        "pending_human_request": None,
        "executed_nodes": ["human_approval"],
    }
    if approved:
        update["status"] = RunStatus.RUNNING
    else:
        update["status"] = RunStatus.STOPPED
        update["stop_reason"] = "human_rejected"
    return update
