"""Versioned, JSON-serialisable Pydantic domain contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, model_validator

from auto_researcher.contracts.enums import (
    EvidenceStatus,
    EventType,
    HypothesisStatus,
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


class ResearchContract(ImmutableDomainModel):
    contract_id: str = Field(min_length=1)
    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    objective_version: str = Field(min_length=1)
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
    provenance: ProvenanceKind

    @model_validator(mode="after")
    def approval_types_must_be_allowed(self) -> "ResearchContract":
        if not self.requires_approval_for.issubset(self.allowed_search_types):
            raise ValueError("requires_approval_for must be a subset of allowed_search_types")
        return self


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


class SearchRequest(ImmutableDomainModel):
    request_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    search_type: SearchType
    target: str = Field(min_length=1)
    search_space: dict[str, JsonValue]
    experiment_budget: int = Field(ge=1)
    rationale: str = Field(min_length=1)
    requires_human_approval: bool = False


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

    @model_validator(mode="after")
    def success_requires_score(self) -> "EvaluationResult":
        if self.success and self.primary_score is None:
            raise ValueError("a successful evaluation requires primary_score")
        if not self.success and self.error is None:
            raise ValueError("a failed evaluation requires error")
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
            return self.model_copy(update={"exhausted": True, "exhaustion_reason": reason})
        return self.model_copy(update={"cycles_used": self.cycles_used + 1})

    def record_experiment(self, cost: float = 0.0) -> "BudgetState":
        experiments = self.experiments_used + 1
        total_cost = self.cost_used + cost
        reason: str | None = None
        if experiments >= self.maximum_experiments:
            reason = "maximum_experiments_reached"
        elif total_cost >= self.maximum_cost:
            reason = "maximum_cost_reached"
        return self.model_copy(
            update={
                "experiments_used": experiments,
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
