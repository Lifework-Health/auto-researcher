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
    ("auto_researcher.search.optuna.models", "OptunaStudySpec"),
    ("auto_researcher.search.optuna.models", "OptunaTrialReference"),
    ("auto_researcher.search.optuna.models", "OptunaTrialOutcome"),
    ("auto_researcher.search.optuna.models", "OptunaStudyState"),
    ("auto_researcher.search.optuna.models", "OptunaStudyResult"),
]


def checkpoint_serializer() -> JsonPlusSerializer:
    """Allow only the domain types deliberately persisted in graph state."""
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_CHECKPOINT_TYPES)


def memory_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=checkpoint_serializer())


def sqlite_checkpointer(path: str | Path) -> tuple[SqliteSaver, sqlite3.Connection]:
    connection = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(connection, serde=checkpoint_serializer()), connection
