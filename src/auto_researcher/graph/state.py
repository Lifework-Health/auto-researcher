"""Compact checkpointed state; large scientific artefacts stay outside the graph."""

from __future__ import annotations

import operator
from typing import Annotated, NotRequired, TypedDict

from auto_researcher.contracts.enums import (
    KnowledgeRetrievalStatus,
    RunStatus,
    SearchType,
)
from auto_researcher.contracts.models import (
    ApprovalRequest,
    BudgetState,
    EvaluationResult,
    ExperimentSpec,
    Hypothesis,
    ResearchContract,
    RunExecutionIdentity,
    SearchBackendResult,
    SearchRequest,
    VerificationResult,
)
from auto_researcher.search.optuna.models import (
    OptunaStudyResult,
    OptunaStudySpec,
    OptunaStudyState,
    OptunaTrialOutcome,
)
from auto_researcher.knowledge.models import KnowledgeBundleReference
from auto_researcher.search.openevolve.models import (
    CandidatePreparationResult,
    CandidateValidationResult,
    MutationReservation,
    OpenEvolveCandidate,
    OpenEvolveCandidateCollection,
    OpenEvolvePopulationState,
    OpenEvolveSearchContract,
    OpenEvolveSearchResult,
)
from auto_researcher.search.openevolve.upstream_models import (
    UpstreamOpenEvolveAdapterState,
)
from auto_researcher.search.openevolve.native_engine import NativeEvolutionResult


class ResearchState(TypedDict):
    run_id: str
    thread_id: str
    contract: ResearchContract
    execution_identity: NotRequired[RunExecutionIdentity]
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
    planner_failure_code: NotRequired[str | None]
    planner_failure_stage: NotRequired[str | None]
    planner_fallback_code: NotRequired[str | None]
    hypothesis_failure_code: NotRequired[str | None]
    hypothesis_failure_stage: NotRequired[str | None]
    hypothesis_fallback_code: NotRequired[str | None]
    recovered_error_codes: NotRequired[list[str]]
    last_executed_search_type: NotRequired[SearchType | None]
    optuna_study_spec: NotRequired[OptunaStudySpec | None]
    optuna_study_state: NotRequired[OptunaStudyState | None]
    optuna_study_result: NotRequired[OptunaStudyResult | None]
    optuna_trial_outcome: NotRequired[OptunaTrialOutcome | None]
    optuna_trial_pruned: NotRequired[bool]
    optuna_trial_operational_terminal: NotRequired[bool]
    optuna_evaluation_reused: NotRequired[bool]
    diagnostic_experiment_spec: NotRequired[ExperimentSpec | None]
    diagnostic_evaluation_result: NotRequired[EvaluationResult | None]
    diagnostic_verification_result: NotRequired[VerificationResult | None]
    knowledge_retrieval_status: NotRequired[KnowledgeRetrievalStatus]
    knowledge_bundle_reference: NotRequired[KnowledgeBundleReference | None]
    knowledge_errors: Annotated[list[str], operator.add]
    knowledge_warnings: Annotated[list[str], operator.add]
    openevolve_search_contract: NotRequired[OpenEvolveSearchContract | None]
    openevolve_population_state: NotRequired[OpenEvolvePopulationState | None]
    openevolve_candidates: NotRequired[OpenEvolveCandidateCollection]
    openevolve_current_candidate: NotRequired[OpenEvolveCandidate | None]
    openevolve_mutation_reservation: NotRequired[MutationReservation | None]
    openevolve_validation_result: NotRequired[CandidateValidationResult | None]
    openevolve_preparation_result: NotRequired[CandidatePreparationResult | None]
    openevolve_search_result: NotRequired[OpenEvolveSearchResult | None]
    upstream_openevolve_adapter_state: NotRequired[
        UpstreamOpenEvolveAdapterState | None
    ]
    openevolve_native_result: NotRequired[NativeEvolutionResult | None]
    openevolve_native_complete: NotRequired[bool]
