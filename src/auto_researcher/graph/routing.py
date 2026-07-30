"""Pure routing functions: no agent may alter these control decisions."""

from __future__ import annotations

from typing import Literal

from auto_researcher.contracts.enums import RunStatus, SearchType
from auto_researcher.graph.state import ResearchState


def route_after_initialise(
    state: ResearchState,
) -> Literal["supervisor_prepare", "__end__"]:
    return "supervisor_prepare" if state["status"] == RunStatus.RUNNING else "__end__"


def route_after_prepare(
    state: ResearchState,
) -> Literal["generate_hypothesis", "record_provenance"]:
    return "record_provenance" if state["status"] != RunStatus.RUNNING else "generate_hypothesis"


def route_approval(
    state: ResearchState,
) -> Literal["human_approval", "search_router", "record_provenance"]:
    if state["status"] not in {
        RunStatus.RUNNING,
        RunStatus.WAITING_FOR_APPROVAL,
    }:
        return "record_provenance"
    request = state["search_request"]
    contract = state["contract"]
    assert request is not None
    requires = (
        request.requires_human_approval
        or request.search_type in contract.requires_approval_for
    )
    return "human_approval" if requires else "search_router"


def route_after_human(
    state: ResearchState,
) -> Literal["search_router", "record_provenance"]:
    return "search_router" if state.get("human_approval_granted") else "record_provenance"


def route_search_backend(
    state: ResearchState,
) -> Literal["direct_search", "optuna_prepare_study", "unavailable_backend"]:
    backend = state["search_backend_result"]
    if backend and backend.available and backend.requested_type == SearchType.DIRECT:
        return "direct_search"
    if backend and backend.available and backend.requested_type == SearchType.OPTUNA:
        return "optuna_prepare_study"
    return "unavailable_backend"


def route_after_optuna_prepare(
    state: ResearchState,
) -> Literal[
    "optuna_ask_trial",
    "optuna_create_experiment",
    "optuna_finalise_study",
]:
    summary = state["optuna_study_state"]
    assert summary is not None
    if summary.finished:
        return "optuna_finalise_study"
    if summary.current_trial is not None:
        return "optuna_create_experiment"
    return "optuna_ask_trial"


def route_after_verification(
    state: ResearchState,
) -> Literal["record_provenance", "optuna_tell_trial"]:
    request = state.get("search_request")
    if request is not None and request.search_type == SearchType.OPTUNA:
        return "optuna_tell_trial"
    return "record_provenance"


def route_after_optuna_decision(
    state: ResearchState,
) -> Literal["optuna_ask_trial", "optuna_finalise_study"]:
    summary = state["optuna_study_state"]
    assert summary is not None
    return "optuna_finalise_study" if summary.finished else "optuna_ask_trial"


def route_after_decision(
    state: ResearchState,
) -> Literal["generate_hypothesis", "__end__"]:
    return "generate_hypothesis" if state["status"] == RunStatus.RUNNING else "__end__"
