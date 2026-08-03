"""Versioned, JSON-serialisable Pydantic domain contracts."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from auto_researcher.contracts.enums import (
    EvidenceStatus,
    EventType,
    GroundingStatus,
    HypothesisStatus,
    KnowledgeGroundingMode,
    ProposalSource,
    ProvenanceKind,
    SearchType,
)


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class ImmutableDomainModel(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class FrozenDict(dict):
    """JSON-compatible mapping that rejects mutations, including nested contract edits."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("research contract values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class FrozenList(list):
    """JSON-compatible list that rejects in-place changes."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("research contract values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


def _freeze_json(value):
    if isinstance(value, dict):
        return FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(_freeze_json(item) for item in value)
    return value


FrozenJsonDict = Annotated[dict[str, JsonValue], AfterValidator(_freeze_json)]


class KnowledgeGroundingRequirement(ImmutableDomainModel):
    mode: KnowledgeGroundingMode = KnowledgeGroundingMode.DISABLED
    permitted_providers: frozenset[str] = Field(default_factory=frozenset)
    permitted_trust_tiers: frozenset[str] = Field(
        default_factory=lambda: frozenset({"CURATED", "CORPUS"})
    )
    minimum_assertion_confidence: float = Field(default=0.6, ge=0, le=1)
    maximum_knowledge_references: int = Field(default=20, ge=0, le=100)
    maximum_query_records: int = Field(default=100, ge=1, le=10_000)
    maximum_graph_hops: int = Field(default=3, ge=0, le=6)
    maximum_retrieval_duration: float = Field(default=20.0, gt=0, le=300)
    knowledge_schema_version: str = Field(default="none", min_length=1)
    knowledge_content_version: str = Field(default="none", min_length=1)

    @model_validator(mode="after")
    def enabled_modes_require_providers(self) -> "KnowledgeGroundingRequirement":
        allowed_tiers = {"CURATED", "CORPUS", "LIVE", "UNVERIFIED"}
        if not self.permitted_trust_tiers.issubset(allowed_tiers):
            raise ValueError("unknown knowledge trust tier")
        if any(not provider.strip() for provider in self.permitted_providers):
            raise ValueError("knowledge provider IDs cannot be empty")
        if (
            self.mode != KnowledgeGroundingMode.DISABLED
            and not self.permitted_providers
        ):
            raise ValueError(
                "enabled knowledge grounding requires a permitted provider"
            )
        if self.mode != KnowledgeGroundingMode.DISABLED and (
            self.knowledge_schema_version == "none"
            or self.knowledge_content_version == "none"
        ):
            raise ValueError("enabled grounding requires schema and content versions")
        if (
            self.mode != KnowledgeGroundingMode.DISABLED
            and self.maximum_knowledge_references == 0
        ):
            raise ValueError("enabled grounding requires at least one reference")
        return self


class ResearchContract(ImmutableDomainModel):
    contract_id: str = Field(min_length=1)
    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    objective_version: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    task_constraints_version: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    constraints: FrozenJsonDict = Field(default_factory=dict)
    allowed_search_types: frozenset[SearchType] = Field(min_length=1)
    evaluator_id: str = Field(min_length=1)
    verifier_id: str = Field(min_length=1)
    maximum_cycles: int = Field(ge=0)
    maximum_experiments: int = Field(ge=0)
    maximum_cost: float = Field(ge=0)
    requires_approval_for: frozenset[SearchType] = Field(default_factory=frozenset)
    grounding: KnowledgeGroundingRequirement = Field(
        default_factory=KnowledgeGroundingRequirement
    )
    provenance: ProvenanceKind

    @model_validator(mode="after")
    def approval_types_must_be_allowed(self) -> "ResearchContract":
        if not self.requires_approval_for.issubset(self.allowed_search_types):
            raise ValueError(
                "requires_approval_for must be a subset of allowed_search_types"
            )
        return self


class RunExecutionIdentity(ImmutableDomainModel):
    """Stable identity bound to one checkpointed LangGraph thread."""

    execution_protocol: str = "run-execution-v2"
    graph_schema_version: str
    thread_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class Hypothesis(ImmutableDomainModel):
    hypothesis_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    predicted_subspace: dict[str, JsonValue]
    expected_observation: str = Field(min_length=1)
    falsification_condition: str = Field(min_length=1)
    evidence_references: tuple[str, ...] = ()
    prior_weight: float = Field(ge=0, le=1)
    status: HypothesisStatus = HypothesisStatus.OPEN
    provenance: ProvenanceKind
    proposal_source: ProposalSource = ProposalSource.DETERMINISTIC
    grounding_status: GroundingStatus = GroundingStatus.CONTRACT_GROUNDED
    agent_call_id: str | None = None
    prompt_version: str | None = None


class SearchRequest(ImmutableDomainModel):
    request_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    search_type: SearchType
    target: str = Field(min_length=1)
    search_space: FrozenJsonDict
    experiment_budget: int = Field(ge=1)
    rationale: str = Field(min_length=1)
    evidence_references: tuple[str, ...] = ()
    requires_human_approval: bool = False
    proposal_source: ProposalSource = ProposalSource.DETERMINISTIC
    grounding_status: GroundingStatus = GroundingStatus.CONTRACT_GROUNDED
    agent_call_id: str | None = None
    prompt_version: str | None = None


class ExperimentSpec(ImmutableDomainModel):
    experiment_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    search_request_id: str = Field(min_length=1)
    configuration: dict[str, JsonValue]
    evaluator_id: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    provenance: ProvenanceKind


class EvaluationResult(ImmutableDomainModel):
    experiment_id: str = Field(min_length=1)
    success: bool
    primary_score: float | None
    metrics: dict[str, JsonValue]
    constraint_results: dict[str, bool]
    artefact_references: tuple[str, ...] = ()
    evaluator_version: str = Field(min_length=1)
    provenance: ProvenanceKind
    error: str | None = None

    @field_validator("constraint_results", mode="before")
    @classmethod
    def constraints_must_be_explicit_booleans(cls, value):
        if not isinstance(value, dict) or any(
            type(item) is not bool for item in value.values()
        ):
            raise ValueError("constraint_results values must be explicit booleans")
        return value

    @model_validator(mode="after")
    def success_requires_score(self) -> "EvaluationResult":
        if self.success and self.primary_score is None:
            raise ValueError("a successful evaluation requires primary_score")
        if self.primary_score is not None and not math.isfinite(self.primary_score):
            raise ValueError("primary_score must be finite")
        if not self.success and self.error is None:
            raise ValueError("a failed evaluation requires error")
        try:
            json.dumps(self.metrics, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation metrics must be strict finite JSON") from exc
        return self


class VerificationResult(ImmutableDomainModel):
    experiment_id: str = Field(min_length=1)
    verified: bool
    claimed_score: float | None
    measured_score: float | None
    constraint_compliant: bool
    evidence_status: EvidenceStatus
    reasons: tuple[str, ...]
    provenance: ProvenanceKind

    @model_validator(mode="after")
    def synthetic_evidence_cannot_support(self) -> "VerificationResult":
        if (
            self.provenance in {ProvenanceKind.MOCK, ProvenanceKind.SIMULATED}
            and self.evidence_status == EvidenceStatus.SUPPORTED
        ):
            raise ValueError("MOCK and SIMULATED evidence cannot be SUPPORTED")
        return self


class DecisionEvent(ImmutableDomainModel):
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    cycle: int = Field(ge=0)
    event_type: EventType
    actor: str = Field(min_length=1)
    input_references: tuple[str, ...] = ()
    output_references: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)
    timestamp: datetime
    code_version: str = Field(min_length=1)
    provenance: ProvenanceKind


class BudgetState(ImmutableDomainModel):
    maximum_cycles: int = Field(ge=0)
    maximum_experiments: int = Field(ge=0)
    maximum_cost: float = Field(ge=0)
    cycles_used: int = Field(default=0, ge=0)
    experiments_used: int = Field(default=0, ge=0)
    cost_used: float = Field(default=0, ge=0)
    model_calls_used: int = Field(default=0, ge=0)
    model_input_tokens_used: int = Field(default=0, ge=0)
    model_output_tokens_used: int = Field(default=0, ge=0)
    model_cache_tokens_used: int = Field(default=0, ge=0)
    model_cache_creation_tokens_used: int = Field(default=0, ge=0)
    model_cache_read_tokens_used: int = Field(default=0, ge=0)
    model_cost_used: float = Field(default=0, ge=0)
    evaluator_cost_used: float = Field(default=0, ge=0)
    exhausted: bool = False
    exhaustion_reason: str | None = None

    def before_cycle(self) -> "BudgetState":
        reason: str | None = None
        if self.cycles_used >= self.maximum_cycles:
            reason = "maximum_cycles_reached"
        elif self.experiments_used >= self.maximum_experiments:
            reason = "maximum_experiments_reached"
        elif self.cost_used >= self.maximum_cost:
            reason = "maximum_cost_reached"
        if reason:
            return self.model_copy(
                update={"exhausted": True, "exhaustion_reason": reason}
            )
        return self.model_copy(update={"cycles_used": self.cycles_used + 1})

    def record_experiment(self, cost: float = 0.0) -> "BudgetState":
        experiments = self.experiments_used + 1
        total_cost = self.cost_used + cost
        evaluator_cost = self.evaluator_cost_used + cost
        reason: str | None = None
        if experiments >= self.maximum_experiments:
            reason = "maximum_experiments_reached"
        elif total_cost >= self.maximum_cost:
            reason = "maximum_cost_reached"
        return self.model_copy(
            update={
                "experiments_used": experiments,
                "cost_used": total_cost,
                "evaluator_cost_used": evaluator_cost,
                "exhausted": reason is not None,
                "exhaustion_reason": reason,
            }
        )

    def record_model_usage(
        self,
        *,
        calls: int = 1,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        cost: float,
    ) -> "BudgetState":
        total_cost = self.cost_used + cost
        reason = (
            "maximum_cost_reached"
            if total_cost >= self.maximum_cost
            else self.exhaustion_reason
        )
        return self.model_copy(
            update={
                "model_calls_used": self.model_calls_used + calls,
                "model_input_tokens_used": self.model_input_tokens_used + input_tokens,
                "model_output_tokens_used": self.model_output_tokens_used
                + output_tokens,
                "model_cache_tokens_used": (
                    self.model_cache_tokens_used
                    + cache_creation_tokens
                    + cache_read_tokens
                ),
                "model_cache_creation_tokens_used": (
                    self.model_cache_creation_tokens_used + cache_creation_tokens
                ),
                "model_cache_read_tokens_used": (
                    self.model_cache_read_tokens_used + cache_read_tokens
                ),
                "model_cost_used": self.model_cost_used + cost,
                "cost_used": total_cost,
                "exhausted": reason is not None,
                "exhaustion_reason": reason,
            }
        )


class ApprovalRequest(ImmutableDomainModel):
    request_id: str
    run_id: str
    cycle: int = Field(ge=1)
    search_request_id: str
    search_type: SearchType
    target: str
    rationale: str


class SearchBackendResult(ImmutableDomainModel):
    requested_type: SearchType
    available: bool
    code: str
    message: str
