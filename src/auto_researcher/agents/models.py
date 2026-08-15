"""Immutable contracts for bounded model calls and scientific proposals."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

from auto_researcher.contracts.enums import (
    AgentCallStatus,
    AgentRole,
    EvidenceStatus,
    GroundingStatus,
    ProviderErrorCode,
    SearchType,
)
from auto_researcher.contracts.models import FrozenJsonDict, Hypothesis
from auto_researcher.knowledge.models import KnowledgeContextReference


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class ModelPricing(AgentModel):
    version: str = Field(min_length=1)
    input_cost_per_million_tokens: float = Field(gt=0)
    output_cost_per_million_tokens: float = Field(gt=0)
    cache_write_cost_per_million_tokens: float | None = Field(default=None, ge=0)
    cache_read_cost_per_million_tokens: float | None = Field(default=None, ge=0)
    currency: str = Field(min_length=1)

    def estimate(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> float:
        input_rate = self.input_cost_per_million_tokens
        cache_write_rate = self.cache_write_cost_per_million_tokens
        cache_read_rate = self.cache_read_cost_per_million_tokens
        uncached_input_tokens = max(
            0,
            input_tokens - cache_creation_input_tokens - cache_read_input_tokens,
        )
        return (
            uncached_input_tokens * input_rate
            + output_tokens * self.output_cost_per_million_tokens
            + cache_creation_input_tokens
            * (cache_write_rate if cache_write_rate is not None else input_rate)
            + cache_read_input_tokens
            * (cache_read_rate if cache_read_rate is not None else input_rate)
        ) / 1_000_000


class ModelCallConfig(AgentModel):
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    temperature: float = Field(ge=0)
    maximum_output_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    maximum_attempts: int = Field(ge=1)
    maximum_cost_per_call: float = Field(gt=0)
    pricing: ModelPricing
    prompt_version: str = Field(min_length=1)
    structured_output_strategy: Literal["pydantic"] = "pydantic"

    @model_validator(mode="after")
    def reject_floating_model_aliases(self) -> "ModelCallConfig":
        if self.model_id.lower().endswith(("-latest", ":latest")):
            raise ValueError(
                "live model_id must be an explicit version, not a latest alias"
            )
        return self


class StructuredModelResponse(AgentModel):
    call_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    structured_output: FrozenJsonDict
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    estimated_cost: float = Field(ge=0)
    latency_ms: int = Field(ge=0)
    attempts: int = Field(default=1, ge=1)
    finish_reason: str | None = None
    provider_request_id: str | None = None
    prompt_version: str = Field(min_length=1)
    context_hash: str = Field(min_length=1)
    response_hash: str = Field(min_length=1)


class HypothesisProposal(AgentModel):
    statement: str = Field(
        min_length=1,
        description="One prospective, falsifiable claim; never claim existing support.",
    )
    rationale: str = Field(
        min_length=1,
        description="Why the bounded experiment is worth running.",
    )
    predicted_subspace: FrozenJsonDict = Field(
        description=(
            "A non-empty JSON object whose keys are copied exactly from the task's "
            "direct_configuration_schema or optuna_space_summary."
        )
    )
    expected_observation: str = Field(
        min_length=1,
        description=(
            "A measurable prospective observation that literally includes the exact "
            "contract primary_metric string."
        ),
    )
    falsification_condition: str = Field(
        min_length=1,
        description="A distinct observation that would falsify the claim.",
    )
    evidence_references: tuple[str, ...] = Field(
        default=(),
        description=(
            "Only IDs copied exactly from permitted_evidence_reference_ids; use an "
            "empty array when none are needed."
        ),
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="A numeric prior confidence between 0 and 1 inclusive.",
    )


class PlannerProposal(AgentModel):
    search_type: SearchType
    target: str = Field(min_length=1)
    proposed_search_space: FrozenJsonDict = Field(
        description=(
            "A JSON object containing only task-registered parameter names and "
            "contract-compatible values. Do not widen registered ranges."
        )
    )
    requested_experiment_budget: int = Field(
        ge=1,
        description=(
            "A positive integer no greater than remaining_experiment_budget; DIRECT "
            "requires exactly 1."
        ),
    )
    rationale: str = Field(min_length=1)
    evidence_references: tuple[str, ...] = Field(
        default=(),
        description=(
            "Only IDs copied exactly from permitted_evidence_reference_ids; use an "
            "empty array when none are needed."
        ),
    )
    recommends_human_approval: bool = False

    @model_validator(mode="after")
    def only_supported_search_types(self) -> "PlannerProposal":
        if self.search_type not in {
            SearchType.DIRECT,
            SearchType.OPTUNA,
            SearchType.OPENEVOLVE,
        }:
            raise ValueError("planner proposal search type is unsupported")
        return self


class AgentBudgetPolicy(AgentModel):
    maximum_hypothesis_calls_per_cycle: int = Field(default=1, ge=1)
    maximum_planner_calls_per_cycle: int = Field(default=1, ge=1)
    maximum_attempts_per_agent_call: int = Field(default=2, ge=1)
    maximum_input_context_size: int = Field(default=24_000, ge=1)
    maximum_output_tokens: int = Field(default=2_048, ge=1)
    maximum_cost_per_call: float = Field(default=1.0, gt=0)
    maximum_total_model_calls: int = Field(default=20, ge=1)


class ContractAgentSummary(AgentModel):
    contract_id: str
    task_id: str
    task_version: str
    objective_version: str
    question: str
    objective: str
    primary_metric: str
    constraints: FrozenJsonDict
    allowed_search_types: tuple[SearchType, ...]
    maximum_experiments: int = Field(ge=0)
    requires_approval_for: tuple[SearchType, ...] = ()


class TaskAgentContext(AgentModel):
    task_id: str
    task_version: str
    display_name: str
    domain: str
    task_description: str
    safe_scientific_vocabulary: tuple[str, ...]
    primary_metric_description: str
    scientific_constraint_summary: tuple[str, ...]
    dataset_summary: FrozenJsonDict
    available_search_types: tuple[SearchType, ...]
    direct_configuration_schema: FrozenJsonDict
    optuna_space_summary: FrozenJsonDict
    openevolve_space_summary: FrozenJsonDict = Field(default_factory=dict)
    fixed_scientific_context: FrozenJsonDict = Field(default_factory=dict)
    task_limitations: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()


class PriorResearchSummary(AgentModel):
    hypothesis_reference: str
    experiment_reference: str
    search_type: SearchType
    primary_score: float | None
    evidence_status: EvidenceStatus
    constraint_compliant: bool
    concise_verified_finding: str
    safe_artefact_references: tuple[str, ...] = ()
    safe_configuration: FrozenJsonDict = Field(default_factory=dict)
    aggregate_metrics: FrozenJsonDict = Field(default_factory=dict)


class HypothesisAgentContext(AgentModel):
    run_id: str
    contract: ContractAgentSummary
    task: TaskAgentContext
    cycle: int = Field(ge=1)
    remaining_experiment_budget: int = Field(ge=0)
    remaining_cost_budget: float = Field(ge=0)
    model_calls_used: int = Field(ge=0)
    previous_hypotheses: tuple[str, ...] = ()
    prior_verified_findings: tuple[PriorResearchSummary, ...] = ()
    permitted_evidence_reference_ids: tuple[str, ...] = ()
    grounding_availability: tuple[GroundingStatus, ...] = ()
    knowledge_references: tuple[KnowledgeContextReference, ...] = ()
    knowledge_bundle_id: str | None = None
    knowledge_bundle_hash: str | None = None
    context_hash: str = Field(min_length=1)


class PlannerAgentContext(AgentModel):
    run_id: str
    contract: ContractAgentSummary
    task: TaskAgentContext
    hypothesis: Hypothesis
    cycle: int = Field(ge=1)
    installed_search_capabilities: tuple[SearchType, ...]
    remaining_experiment_budget: int = Field(ge=0)
    remaining_cost_budget: float = Field(ge=0)
    remaining_time_seconds: float | None = Field(default=None, ge=0)
    campaign_deadline_at: AwareDatetime | None = None
    model_calls_used: int = Field(ge=0)
    approval_requirements: tuple[SearchType, ...]
    prior_verified_findings: tuple[PriorResearchSummary, ...] = ()
    permitted_evidence_reference_ids: tuple[str, ...] = ()
    knowledge_references: tuple[KnowledgeContextReference, ...] = ()
    knowledge_bundle_id: str | None = None
    knowledge_bundle_hash: str | None = None
    permitted_direct_configuration_schema: FrozenJsonDict
    permitted_optuna_maximum_space: FrozenJsonDict
    optuna_narrowing_rules: tuple[str, ...] = ()
    context_hash: str = Field(min_length=1)


class AgentCallRecord(AgentModel):
    record_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    cycle: int = Field(ge=1)
    role: AgentRole
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    prompt_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    context_hash: str = Field(min_length=1)
    response_schema_version: str = Field(min_length=1)
    status: AgentCallStatus
    structured_output: FrozenJsonDict | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0, ge=0)
    pricing: ModelPricing
    pricing_version: str = Field(min_length=1)
    pricing_currency: str = Field(min_length=1)
    latency_ms: int = Field(default=0, ge=0)
    provider_request_id: str | None = None
    attempt_count: int = Field(default=0, ge=0)
    created_at: AwareDatetime
    error_code: ProviderErrorCode | None = None
    response_hash: str | None = None
    retry_of_call_id: str | None = None
    provider_request_started: bool = False
    # Optional agent-call-store-v2 fields. They remain optional so v1 hypothesis and
    # planner records deserialize without reinterpretation.
    semantic_key: str | None = None
    approval_id: str | None = None
    approval_hash: str | None = None
    budget_identity: str | None = None
    input_payload_hash: str | None = None
    maximum_reserved_cost: float = Field(default=0, ge=0)
    reservation_timestamp: AwareDatetime | None = None
    dispatch_timestamp: AwareDatetime | None = None
    completion_timestamp: AwareDatetime | None = None
    completion_identity: str | None = None
    finish_reason: str | None = None
    replayed: bool = False

    @model_validator(mode="after")
    def status_payload_is_consistent(self) -> "AgentCallRecord":
        if self.pricing.version != self.pricing_version:
            raise ValueError("pricing version snapshot mismatch")
        if self.pricing.currency != self.pricing_currency:
            raise ValueError("pricing currency snapshot mismatch")
        if self.status == AgentCallStatus.COMPLETED and (
            self.structured_output is None or self.response_hash is None
        ):
            raise ValueError(
                "completed calls require structured output and response hash"
            )
        if (
            self.status
            in {
                AgentCallStatus.FAILED,
                AgentCallStatus.FAILED_BEFORE_DISPATCH,
                AgentCallStatus.FAILED_CONFIRMED,
                AgentCallStatus.OUTCOME_UNKNOWN,
                AgentCallStatus.REJECTED,
            }
            and self.error_code is None
        ):
            raise ValueError("failed calls require a safe error code")
        if (
            self.status == AgentCallStatus.RESERVED
            and self.structured_output is not None
        ):
            raise ValueError("reserved calls cannot contain structured output")
        if self.role == AgentRole.OPENEVOLVE_MUTATION:
            required = {
                "semantic_key": self.semantic_key,
                "approval_id": self.approval_id,
                "approval_hash": self.approval_hash,
                "budget_identity": self.budget_identity,
                "input_payload_hash": self.input_payload_hash,
            }
            if any(value is None for value in required.values()):
                raise ValueError("OpenEvolve mutation calls require v2 identity fields")
            if self.status == AgentCallStatus.COMPLETED and (
                self.completion_timestamp is None or self.completion_identity is None
            ):
                raise ValueError("completed mutation calls require completion identity")
        return self


class AgentCallTelemetry(AgentModel):
    call_id: str
    role: AgentRole
    provider: str
    model_id: str
    prompt_version: str
    context_hash: str
    grounding_status: GroundingStatus | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0, ge=0)
    provider_attempts: int = Field(default=0, ge=0)
    replayed: bool = False
    failed: bool = False
    cost_limit_exceeded: bool = False
    error_code: ProviderErrorCode | None = None


def json_schema_version(model: type[BaseModel]) -> str:
    """Stable schema fingerprint stored with every model call."""
    import hashlib
    import json

    encoded = json.dumps(
        model.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def json_safe_model(value: BaseModel | dict[str, Any]) -> dict[str, JsonValue]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)
