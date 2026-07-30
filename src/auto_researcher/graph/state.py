"""Compact checkpointed state; large scientific artefacts stay outside the graph."""

from __future__ import annotations

import operator
from typing import Annotated, NotRequired, TypedDict

from auto_researcher.contracts.enums import RunStatus
from auto_researcher.contracts.models import (
    ApprovalRequest,
    BudgetState,
    EvaluationResult,
    ExperimentSpec,
    Hypothesis,
    ResearchContract,
    SearchBackendResult,
    SearchRequest,
    VerificationResult,
)


class ResearchState(TypedDict):
    run_id: str
    thread_id: str
    contract: ResearchContract
    status: RunStatus
    cycle: int
    budget: BudgetState
    active_hypothesis: NotRequired[Hypothesis | None]
    search_request: NotRequired[SearchRequest | None]
    search_backend_result: NotRequired[SearchBackendResult | None]
    experiment_spec: NotRequired[ExperimentSpec | None]
    evaluation_result: NotRequired[EvaluationResult | None]
    verification_result: NotRequired[VerificationResult | None]
    decision_event_ids: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    executed_nodes: Annotated[list[str], operator.add]
    pending_human_request: NotRequired[ApprovalRequest | None]
    human_approval_granted: NotRequired[bool | None]
    stop_reason: NotRequired[str | None]
