"""Versioned, finite and checkpoint-safe OpenEvolve contracts."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from auto_researcher.contracts.enums import EvidenceStatus, SearchType
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    FrozenJsonDict,
    VerificationResult,
)

OPENEVOLVE_SEARCH_VERSION: Final = "openevolve-search-v1"
OPENEVOLVE_CANDIDATE_VERSION: Final = "openevolve-candidate-v1"
OPENEVOLVE_POPULATION_VERSION: Final = "openevolve-population-v1"
OPENEVOLVE_LINEAGE_VERSION: Final = "openevolve-lineage-v1"
OPENEVOLVE_SANDBOX_VERSION: Final = "openevolve-sandbox-v1"
OPENEVOLVE_PROVENANCE_VERSION: Final = "openevolve-provenance-v1"
OPENEVOLVE_SELECTION_POLICY_VERSION: Final = "constraint-verification-objective-v2"


class OpenEvolveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"


class CandidateStatus(StrEnum):
    PROPOSED = "PROPOSED"
    REJECTED = "REJECTED"
    VALIDATED = "VALIDATED"
    PREPARED = "PREPARED"
    EVALUATED = "EVALUATED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


class CandidateExecutionStatus(StrEnum):
    NOT_RUN = "NOT_RUN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    RESOURCE_LIMITED = "RESOURCE_LIMITED"


class CandidateValidationStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"


class MutationOperatorPolicy(OpenEvolveModel):
    policy_id: str = Field(min_length=1)
    allowed_operator_ids: tuple[str, ...] = Field(min_length=1)
    maximum_patch_bytes: int = Field(gt=0, le=1_000_000)
    maximum_resulting_source_bytes: int = Field(gt=0, le=1_000_000)
    structured_full_file_replacement_only: bool = True
    allow_crossover: bool = False


class SelectionPolicy(OpenEvolveModel):
    policy_id: str = OPENEVOLVE_SELECTION_POLICY_VERSION
    direction: ObjectiveDirection
    objective_metric: str = Field(min_length=1)
    deterministic_tie_break: Literal["candidate_id"] = "candidate_id"


class ReplacementPolicy(OpenEvolveModel):
    policy_id: str = "bounded-elitist-replacement-v1"
    preserve_archive: bool = True


class SandboxPolicy(OpenEvolveModel):
    policy_id: str = OPENEVOLVE_SANDBOX_VERSION
    cpu_time_seconds: int = Field(gt=0, le=60)
    wall_time_seconds: float = Field(gt=0, le=120)
    memory_bytes: int = Field(gt=0, le=2_147_483_648)
    process_limit: int = Field(gt=0, le=16)
    output_bytes: int = Field(gt=0, le=10_000_000)
    log_bytes: int = Field(gt=0, le=1_000_000)
    file_count_limit: int = Field(gt=0, le=100)
    workspace_bytes: int = Field(default=1_048_576, gt=0, le=100_000_000)
    file_size_bytes: int = Field(default=64_000, gt=0, le=10_000_000)
    dependency_allowlist: tuple[str, ...] = ()
    network_access: Literal[False] = False
    shell_access: Literal[False] = False
    package_installation: Literal[False] = False
    inherit_environment: Literal[False] = False
    locale: Literal["C"] = "C"
    timezone: Literal["UTC"] = "UTC"


class EvolvableComponentSpec(OpenEvolveModel):
    component_id: str = Field(min_length=1)
    component_version: str = Field(min_length=1)
    representation_type: Literal["python_source_v1"] = "python_source_v1"
    source_language: Literal["python"] = "python"
    mutable_file: str = Field(pattern=r"^[A-Za-z0-9_.-]+\.py$")
    allowed_files: tuple[str, ...] = Field(min_length=1, max_length=8)
    entry_point: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    immutable_interface_contract: str = Field(min_length=1)
    allowed_imports: tuple[str, ...] = ()
    allowed_dependencies: tuple[str, ...] = ()
    parameter_schema: FrozenJsonDict
    output_schema: FrozenJsonDict
    seed_source: str = Field(min_length=1)
    deterministic_mutation_sources: tuple[str, ...] = ()
    maximum_source_bytes: int = Field(gt=0, le=1_000_000)
    task_mutation_context: FrozenJsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def mutable_surface_is_explicit_and_narrow(self) -> "EvolvableComponentSpec":
        if self.allowed_files != (self.mutable_file,):
            raise ValueError("PR 6 permits exactly one explicitly mutable Python file")
        if any(
            "/" in item or "\\" in item or item in {".", ".."}
            for item in self.allowed_files
        ):
            raise ValueError("allowed files must be safe relative file names")
        if len(self.seed_source.encode("utf-8")) > self.maximum_source_bytes:
            raise ValueError("seed source exceeds the component source cap")
        if any(
            len(item.encode("utf-8")) > self.maximum_source_bytes
            for item in self.deterministic_mutation_sources
        ):
            raise ValueError("mutation source exceeds the component source cap")
        if set(self.allowed_imports) != set(self.allowed_dependencies):
            raise ValueError(
                "allowed imports and dependencies must have identical scope"
            )
        return self


class OpenEvolveSearchContract(OpenEvolveModel):
    protocol_version: Literal["openevolve-search-v1"] = OPENEVOLVE_SEARCH_VERSION
    search_request_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    search_type: Literal[SearchType.OPENEVOLVE] = SearchType.OPENEVOLVE
    evolvable_component_id: str = Field(min_length=1)
    evolvable_component_version: str = Field(min_length=1)
    seed_candidate_id: str = Field(min_length=1)
    population_size: int = Field(gt=0, le=1_000)
    maximum_generations: int = Field(gt=0, le=100_000)
    maximum_candidate_evaluations: int = Field(gt=0, le=100_000)
    maximum_wall_time_seconds: float = Field(gt=0, le=604_800)
    maximum_model_calls: int = Field(ge=0, le=100_000)
    maximum_failed_candidates: int = Field(gt=0, le=100_000)
    maximum_consecutive_failures: int = Field(gt=0, le=100_000)
    maximum_artefact_bytes: int = Field(gt=0, le=10_000_000_000)
    mutation_operator_policy: MutationOperatorPolicy
    selection_policy: SelectionPolicy
    replacement_policy: ReplacementPolicy
    sandbox_policy: SandboxPolicy
    evaluator_identity: str = Field(min_length=1)
    verifier_identity: str = Field(min_length=1)
    random_seed: int
    resume_policy: Literal["reuse_completed_side_effects_v1"] = (
        "reuse_completed_side_effects_v1"
    )
    stopping_policy: Literal["deterministic_bounded_v1"] = "deterministic_bounded_v1"
    objective_threshold: float | None = None

    @field_validator("maximum_wall_time_seconds", "objective_threshold")
    @classmethod
    def finite_numbers(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("OpenEvolve numeric limits and thresholds must be finite")
        return value

    @model_validator(mode="after")
    def limits_are_coherent(self) -> "OpenEvolveSearchContract":
        if self.population_size > self.maximum_candidate_evaluations:
            raise ValueError(
                "population size cannot exceed candidate evaluation budget"
            )
        if self.maximum_consecutive_failures > self.maximum_failed_candidates:
            raise ValueError(
                "consecutive failure limit cannot exceed total failure limit"
            )
        return self


class MutationReservation(OpenEvolveModel):
    reservation_id: str = Field(min_length=1)
    search_request_id: str = Field(min_length=1)
    parent_candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=1)
    generation: int = Field(ge=1)
    birth_index: int = Field(ge=1)
    mutation_operator: str = Field(min_length=1)
    input_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    mutation_request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_context: FrozenJsonDict = Field(default_factory=dict)
    parent_feedback: FrozenJsonDict = Field(default_factory=dict)


class CandidateValidationResult(OpenEvolveModel):
    protocol_version: Literal["candidate-validation-v1"] = "candidate-validation-v1"
    candidate_id: str = Field(min_length=1)
    status: CandidateValidationStatus
    safe_error_code: str | None = None
    reason_codes: tuple[str, ...] = ()
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    interface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class CandidatePreparationResult(OpenEvolveModel):
    protocol_version: Literal[
        "candidate-preparation-v1", "candidate-preparation-v2"
    ] = "candidate-preparation-v1"
    candidate_id: str = Field(min_length=1)
    validation_status: CandidateValidationStatus
    execution_status: CandidateExecutionStatus
    safe_error_code: str | None = None
    timeout: bool = False
    resource_limited: bool = False
    output_references: tuple[str, ...] = ()
    output_hashes: tuple[str, ...] = ()
    generated_configuration: FrozenJsonDict = Field(default_factory=dict)
    generated_experiment_id: str | None = None
    safe_log_excerpt: str = ""
    log_truncated: bool = False
    runtime_seconds: float = Field(default=0.0, ge=0)
    cleanup_complete: bool
    executor_id: str | None = None
    executor_policy_identity: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    execution_request_identity: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    workspace_policy_identity: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    worker_protocol_version: str | None = None
    supervisor_identity: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    image_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    declared_file_count_limit: int | None = Field(default=None, gt=0)
    derived_inode_limit: int | None = Field(default=None, gt=1)
    observed_workspace_entry_count: int | None = Field(default=None, ge=0)
    observed_workspace_bytes: int | None = Field(default=None, ge=0)
    observed_max_file_bytes: int | None = Field(default=None, ge=0)
    workspace_bytes_limit: int | None = Field(default=None, gt=0)
    file_size_bytes_limit: int | None = Field(default=None, gt=0)
    observed_output_bytes: int | None = Field(default=None, ge=0)
    resource_limit_reason: str | None = None

    @field_validator("runtime_seconds")
    @classmethod
    def runtime_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("candidate runtime must be finite")
        return value


class OpenEvolveCandidate(OpenEvolveModel):
    protocol_version: Literal["openevolve-candidate-v1"] = OPENEVOLVE_CANDIDATE_VERSION
    candidate_id: str = Field(min_length=1)
    candidate_version: str = OPENEVOLVE_CANDIDATE_VERSION
    search_request_id: str = Field(min_length=1)
    parent_candidate_ids: tuple[str, ...] = Field(max_length=1)
    generation: int = Field(ge=0)
    birth_index: int = Field(ge=0)
    mutation_operator: str = Field(min_length=1)
    mutation_description: str = Field(min_length=1, max_length=2_000)
    mutable_file: str = Field(min_length=1)
    source_payload: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_source_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    component_interface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sandbox_policy_id: str = Field(min_length=1)
    status: CandidateStatus
    validation_result: CandidateValidationResult | None = None
    preparation_result: CandidatePreparationResult | None = None
    evaluation_identity: str | None = None
    model_call_id: str | None = None
    creation_provenance: Literal[
        "SEED", "DETERMINISTIC_FIXTURE", "FAKE_MODEL", "LIVE_MODEL"
    ]


class OpenEvolveCandidateCollection(OpenEvolveModel):
    """Typed wrapper preserving tuple identity through checkpoint reconstruction."""

    candidates: tuple[OpenEvolveCandidate, ...] = ()


class CandidateOutcome(OpenEvolveModel):
    candidate_id: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: CandidateStatus
    objective_value: float | None = None
    constraint_compliant: bool = False
    verified: bool = False
    evidence_status: EvidenceStatus | None = None
    experiment: ExperimentSpec | None = None
    evaluation: EvaluationResult | None = None
    verification: VerificationResult | None = None
    selection_outcome: str = Field(min_length=1)
    rejection_reason: str | None = None
    replacement_outcome: str = Field(min_length=1)

    @field_validator("objective_value")
    @classmethod
    def objective_is_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("candidate objective must be finite")
        return value


class LineageRecord(OpenEvolveModel):
    protocol_version: Literal["openevolve-lineage-v1"] = OPENEVOLVE_LINEAGE_VERSION
    candidate_id: str = Field(min_length=1)
    parent_candidate_ids: tuple[str, ...] = Field(max_length=1)
    generation: int = Field(ge=0)
    mutation_operator: str = Field(min_length=1)
    model_call_id: str | None = None
    source_hash_before: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_hash_after: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_code: str = Field(min_length=1)
    evaluation_identity: str | None = None
    selection_outcome: str = Field(min_length=1)
    rejection_reason: str | None = None
    replacement_outcome: str = Field(min_length=1)


class OpenEvolveBudgetState(OpenEvolveModel):
    generations_used: int = Field(default=0, ge=0)
    candidate_proposals: int = Field(default=0, ge=0)
    successful_preparations: int = Field(default=0, ge=0)
    failed_preparations: int = Field(default=0, ge=0)
    candidate_evaluations: int = Field(default=0, ge=0)
    verifier_calls: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    wall_time_elapsed: float = Field(default=0.0, ge=0)
    candidate_runtime: float = Field(default=0.0, ge=0)
    artefact_bytes: int = Field(default=0, ge=0)
    failed_candidates: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)


class OpenEvolvePopulationState(OpenEvolveModel):
    protocol_version: Literal["openevolve-population-v1"] = (
        OPENEVOLVE_POPULATION_VERSION
    )
    search_request_id: str = Field(min_length=1)
    search_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: int = Field(default=0, ge=0)
    active_population_candidate_ids: tuple[str, ...] = ()
    archive_candidate_ids: tuple[str, ...] = ()
    evaluated_candidate_ids: tuple[str, ...] = ()
    failed_candidate_ids: tuple[str, ...] = ()
    selected_parent_ids: tuple[str, ...] = ()
    best_known_candidate_ids: tuple[str, ...] = ()
    source_hashes: tuple[str, ...] = ()
    outcomes: tuple[CandidateOutcome, ...] = ()
    lineage: tuple[LineageRecord, ...] = ()
    diversity_metadata: FrozenJsonDict = Field(default_factory=dict)
    budget: OpenEvolveBudgetState = Field(default_factory=OpenEvolveBudgetState)
    random_seed_state: int
    stopping_status: Literal["RUNNING", "STOPPED"] = "RUNNING"
    stop_reason: str | None = None
    current_candidate_id: str | None = None
    current_reservation_id: str | None = None


class OpenEvolveSearchResult(OpenEvolveModel):
    protocol_version: Literal["openevolve-search-result-v1"] = (
        "openevolve-search-result-v1"
    )
    search_request_id: str = Field(min_length=1)
    search_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates_proposed: int = Field(ge=0)
    candidates_evaluated: int = Field(ge=0)
    candidates_failed: int = Field(ge=0)
    generations_completed: int = Field(ge=0)
    best_candidate_ids: tuple[str, ...]
    feasible_candidate_found: bool
    stop_reason: str = Field(min_length=1)
    artefact_references: tuple[str, ...] = ()
