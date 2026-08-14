"""Factories for executable state persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

ALLOWED_CHECKPOINT_TYPES = [
    ("auto_researcher.contracts.enums", "EvidenceStatus"),
    ("auto_researcher.contracts.enums", "GroundingStatus"),
    ("auto_researcher.contracts.enums", "EventType"),
    ("auto_researcher.contracts.enums", "HypothesisStatus"),
    ("auto_researcher.contracts.enums", "KnowledgeGroundingMode"),
    ("auto_researcher.contracts.enums", "KnowledgeRetrievalStatus"),
    ("auto_researcher.contracts.enums", "ProvenanceKind"),
    ("auto_researcher.contracts.enums", "ProposalSource"),
    ("auto_researcher.contracts.enums", "ReadSafetyMode"),
    ("auto_researcher.contracts.enums", "RunStatus"),
    ("auto_researcher.contracts.enums", "SearchType"),
    ("auto_researcher.contracts.models", "ApprovalRequest"),
    ("auto_researcher.contracts.models", "BudgetState"),
    ("auto_researcher.contracts.models", "DecisionEvent"),
    ("auto_researcher.contracts.models", "EvaluationResult"),
    ("auto_researcher.contracts.models", "ExperimentSpec"),
    ("auto_researcher.contracts.models", "Hypothesis"),
    ("auto_researcher.contracts.models", "ResearchContract"),
    ("auto_researcher.contracts.models", "RunExecutionIdentity"),
    ("auto_researcher.contracts.models", "SearchBackendResult"),
    ("auto_researcher.contracts.models", "SearchRequest"),
    ("auto_researcher.contracts.models", "VerificationResult"),
    ("auto_researcher.knowledge.models", "KnowledgeBundleReference"),
    ("auto_researcher.search.optuna.models", "OptimisationDirection"),
    ("auto_researcher.search.optuna.models", "OptunaTrialStatus"),
    ("auto_researcher.search.optuna.models", "FloatParameterSpec"),
    ("auto_researcher.search.optuna.models", "IntParameterSpec"),
    ("auto_researcher.search.optuna.models", "CategoricalParameterSpec"),
    ("auto_researcher.search.optuna.models", "OptunaConditionSpec"),
    ("auto_researcher.search.optuna.models", "OptunaObjectiveSpec"),
    ("auto_researcher.search.optuna.models", "OptunaConstraintSpec"),
    ("auto_researcher.search.optuna.models", "OptunaSamplerSpec"),
    ("auto_researcher.search.optuna.models", "OptunaPrunerSpec"),
    ("auto_researcher.search.optuna.models", "OptunaDiagnosticsSpec"),
    ("auto_researcher.search.optuna.models", "OptunaStudyDiagnostics"),
    ("auto_researcher.search.optuna.models", "OptunaStudySpec"),
    ("auto_researcher.search.optuna.models", "OptunaTrialReference"),
    ("auto_researcher.search.optuna.models", "OptunaTrialOutcome"),
    ("auto_researcher.search.optuna.models", "OptunaStudyState"),
    ("auto_researcher.search.optuna.models", "OptunaStudyResult"),
    ("auto_researcher.search.openevolve.models", "ObjectiveDirection"),
    ("auto_researcher.search.openevolve.models", "CandidateStatus"),
    ("auto_researcher.search.openevolve.models", "CandidateExecutionStatus"),
    ("auto_researcher.search.openevolve.models", "CandidateValidationStatus"),
    ("auto_researcher.search.openevolve.models", "MutationOperatorPolicy"),
    ("auto_researcher.search.openevolve.models", "SelectionPolicy"),
    ("auto_researcher.search.openevolve.models", "ReplacementPolicy"),
    ("auto_researcher.search.openevolve.models", "SandboxPolicy"),
    ("auto_researcher.search.openevolve.models", "EvolvableComponentSpec"),
    ("auto_researcher.search.openevolve.models", "OpenEvolveSearchContract"),
    ("auto_researcher.search.openevolve.models", "MutationReservation"),
    ("auto_researcher.search.openevolve.models", "CandidateValidationResult"),
    ("auto_researcher.search.openevolve.models", "CandidatePreparationResult"),
    ("auto_researcher.search.openevolve.models", "OpenEvolveCandidate"),
    ("auto_researcher.search.openevolve.models", "OpenEvolveCandidateCollection"),
    ("auto_researcher.search.openevolve.models", "CandidateOutcome"),
    ("auto_researcher.search.openevolve.models", "LineageRecord"),
    ("auto_researcher.search.openevolve.models", "OpenEvolveBudgetState"),
    ("auto_researcher.search.openevolve.models", "OpenEvolvePopulationState"),
    ("auto_researcher.search.openevolve.models", "OpenEvolveSearchResult"),
    (
        "auto_researcher.search.openevolve.upstream_models",
        "UpstreamOpenEvolveAdapterState",
    ),
    (
        "auto_researcher.search.openevolve.native_engine",
        "SafeEvolutionFeedback",
    ),
    (
        "auto_researcher.search.openevolve.native_engine",
        "NativeEvolutionDecision",
    ),
    (
        "auto_researcher.search.openevolve.native_engine",
        "NativeEvolutionResult",
    ),
]


def checkpoint_serializer() -> JsonPlusSerializer:
    """Allow only the domain types deliberately persisted in graph state."""
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_CHECKPOINT_TYPES)


def memory_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=checkpoint_serializer())


def sqlite_checkpointer(path: str | Path) -> tuple[SqliteSaver, sqlite3.Connection]:
    connection = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(connection, serde=checkpoint_serializer()), connection
