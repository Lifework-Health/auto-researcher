"""Task-agnostic, epistemically typed Research State v1 contracts."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from auto_researcher.knowledge.identity import content_hash, stable_identifier
from auto_researcher.research_intelligence.models import (
    EVIDENCE_CARD_VERSION,
    EXTERNAL_EVIDENCE_BOUNDARY,
    EvidenceCard,
)

RESEARCH_STATE_VERSION = "research-state-v1"
INTERNAL_EXPERIMENTAL_BOUNDARY = "INTERNAL_EXPERIMENTAL_OBSERVATION"
INTERNAL_DIAGNOSTIC_BOUNDARY = "INTERNAL_DIAGNOSTIC_OBSERVATION"
PLANNER_INFERENCE_BOUNDARY = "PLANNER_INFERENCE"
PLANNER_DECISION_BOUNDARY = "PLANNER_DECISION"


class ResearchStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class HypothesisStatus(StrEnum):
    PROPOSED = "PROPOSED"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    UNRESOLVED = "UNRESOLVED"
    SUPERSEDED = "SUPERSEDED"


class UncertaintyStatus(StrEnum):
    OPEN = "OPEN"
    REDUCED = "REDUCED"
    RESOLVED = "RESOLVED"
    SUPERSEDED = "SUPERSEDED"


class ExperimentStatus(StrEnum):
    PROPOSED = "PROPOSED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    SUPERSEDED = "SUPERSEDED"


class WorkStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    SUPERSEDED = "SUPERSEDED"


class ObservationRole(StrEnum):
    PRIMARY = "PRIMARY"
    ROBUSTNESS = "ROBUSTNESS"
    REPLICATION = "REPLICATION"
    ABLATION = "ABLATION"
    CONFIRMATION = "CONFIRMATION"
    CLASS_SPECIFIC = "CLASS_SPECIFIC"
    TRAJECTORY = "TRAJECTORY"


class ExperimentModality(StrEnum):
    HPO = "HPO"
    OPENEVOLVE = "OPENEVOLVE"
    ABLATION = "ABLATION"
    REPLICATION = "REPLICATION"
    ROBUSTNESS = "ROBUSTNESS"
    DIAGNOSTIC = "DIAGNOSTIC"
    CONFIRMATION = "CONFIRMATION"
    DIRECT = "DIRECT"


class InformationValueClass(StrEnum):
    EXPLORATORY = "EXPLORATORY"
    DISCRIMINATING = "DISCRIMINATING"
    CONFIRMATORY = "CONFIRMATORY"
    ROBUSTNESS = "ROBUSTNESS"
    REPLICATION = "REPLICATION"
    DIAGNOSTIC = "DIAGNOSTIC"


class ConfidenceLevel(StrEnum):
    UNASSESSED = "UNASSESSED"
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class HypothesisOrigin(StrEnum):
    EXTERNAL_EVIDENCE = "EXTERNAL_EVIDENCE"
    INTERNAL_OBSERVATION = "INTERNAL_OBSERVATION"
    DIAGNOSTIC_OBSERVATION = "DIAGNOSTIC_OBSERVATION"
    PLANNER_INFERENCE = "PLANNER_INFERENCE"
    HUMAN = "HUMAN"
    MIXED = "MIXED"


class RecordType(StrEnum):
    EXTERNAL_EVIDENCE = "EXTERNAL_EVIDENCE"
    INTERNAL_OBSERVATION = "INTERNAL_OBSERVATION"
    DIAGNOSTIC_OBSERVATION = "DIAGNOSTIC_OBSERVATION"
    HYPOTHESIS = "HYPOTHESIS"
    UNCERTAINTY = "UNCERTAINTY"
    PLANNER_INFERENCE = "PLANNER_INFERENCE"
    PLANNER_DECISION = "PLANNER_DECISION"
    EXPERIMENT = "EXPERIMENT"
    WORK_ITEM = "WORK_ITEM"
    NEXT_ACTION = "NEXT_ACTION"


class ResearchObjective(ResearchStateModel):
    objective_id: str = Field(min_length=1, max_length=200)
    statement: str = Field(min_length=1, max_length=2_000)
    metric_ids: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()


class ProgrammeContext(ResearchStateModel):
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    domains: tuple[str, ...] = Field(min_length=1)
    architecture_references: tuple[str, ...] = ()
    dataset_references: tuple[str, ...] = ()
    evaluator_references: tuple[str, ...] = ()


class ResearchProgramme(ResearchStateModel):
    programme_id: str = Field(min_length=1, max_length=300)
    programme_version: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=500)
    objectives: tuple[ResearchObjective, ...] = Field(min_length=1)
    context: ProgrammeContext
    created_at: AwareDatetime
    state_schema_version: Literal["research-state-v1"] = RESEARCH_STATE_VERSION

    @model_validator(mode="after")
    def objective_identities_are_unique(self) -> "ResearchProgramme":
        identifiers = tuple(item.objective_id for item in self.objectives)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("research programme objective identities must be unique")
        return self


class EvidenceReferences(ResearchStateModel):
    """Typed references only; never an untyped evidence/prose container."""

    external_evidence_card_ids: tuple[str, ...] = ()
    internal_observation_ids: tuple[str, ...] = ()
    diagnostic_observation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def references_are_unique(self) -> "EvidenceReferences":
        groups = (
            self.external_evidence_card_ids,
            self.internal_observation_ids,
            self.diagnostic_observation_ids,
        )
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("evidence references must be unique within each boundary")
        return self

    @property
    def is_empty(self) -> bool:
        return not (
            self.external_evidence_card_ids
            or self.internal_observation_ids
            or self.diagnostic_observation_ids
        )

    def merged(self, other: "EvidenceReferences") -> "EvidenceReferences":
        return EvidenceReferences(
            external_evidence_card_ids=tuple(
                sorted(
                    set(self.external_evidence_card_ids)
                    | set(other.external_evidence_card_ids)
                )
            ),
            internal_observation_ids=tuple(
                sorted(
                    set(self.internal_observation_ids)
                    | set(other.internal_observation_ids)
                )
            ),
            diagnostic_observation_ids=tuple(
                sorted(
                    set(self.diagnostic_observation_ids)
                    | set(other.diagnostic_observation_ids)
                )
            ),
        )


class VersionedRecord(ResearchStateModel):
    programme_id: str = Field(min_length=1)
    revision: int = Field(default=1, ge=1)
    recorded_at: AwareDatetime


class ExternalEvidenceReference(VersionedRecord):
    record_type: Literal["EXTERNAL_EVIDENCE"] = RecordType.EXTERNAL_EVIDENCE.value
    evidence_boundary: Literal["EXTERNAL_RESEARCH_INTELLIGENCE"] = (
        EXTERNAL_EVIDENCE_BOUNDARY
    )
    evidence_card_id: str = Field(pattern=r"^evidence-[0-9a-f]{24}$")
    evidence_card_version: Literal["research-intelligence-evidence-card-v1"] = (
        EVIDENCE_CARD_VERSION
    )
    evidence_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_store_reference: str = Field(min_length=1, max_length=2_000)


class Fidelity(ResearchStateModel):
    level: str = Field(min_length=1, max_length=100)
    dataset_fraction: float | None = Field(default=None, gt=0, le=1)
    training_steps: int | None = Field(default=None, ge=1)
    replicate_index: int | None = Field(default=None, ge=0)


class StructuredResultReference(ResearchStateModel):
    reference_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    location: str = Field(min_length=1, max_length=2_000)


class InternalExperimentalObservation(VersionedRecord):
    record_type: Literal["INTERNAL_OBSERVATION"] = RecordType.INTERNAL_OBSERVATION.value
    evidence_boundary: Literal["INTERNAL_EXPERIMENTAL_OBSERVATION"] = (
        INTERNAL_EXPERIMENTAL_BOUNDARY
    )
    observation_id: str = Field(min_length=1, max_length=300)
    experiment_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    result_id: str = Field(min_length=1)
    metric_id: str = Field(min_length=1)
    measured_value: float | int | str | bool | None = None
    structured_result_reference: StructuredResultReference | None = None
    unit: str | None = Field(default=None, min_length=1, max_length=100)
    fidelity: Fidelity
    dataset_reference: str | None = Field(default=None, min_length=1)
    split_reference: str | None = Field(default=None, min_length=1)
    evaluator_reference: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    observed_at: AwareDatetime
    provenance_reference: str = Field(min_length=1)
    observation_role: ObservationRole

    @model_validator(mode="after")
    def has_exactly_one_measured_result(self) -> "InternalExperimentalObservation":
        if (self.measured_value is None) == (self.structured_result_reference is None):
            raise ValueError(
                "internal observation requires exactly one measured value or structured result reference"
            )
        if isinstance(self.measured_value, float) and not math.isfinite(
            self.measured_value
        ):
            raise ValueError("internal observation measured value must be finite")
        return self


class DiagnosticObservation(VersionedRecord):
    record_type: Literal["DIAGNOSTIC_OBSERVATION"] = (
        RecordType.DIAGNOSTIC_OBSERVATION.value
    )
    evidence_boundary: Literal["INTERNAL_DIAGNOSTIC_OBSERVATION"] = (
        INTERNAL_DIAGNOSTIC_BOUNDARY
    )
    diagnostic_observation_id: str = Field(min_length=1, max_length=300)
    diagnostic_run_id: str = Field(min_length=1)
    diagnostic_kind: str = Field(min_length=1)
    subject_references: tuple[str, ...] = Field(min_length=1)
    result_reference: StructuredResultReference
    finding: str = Field(min_length=1, max_length=2_000)
    diagnostic_system_reference: str = Field(min_length=1)
    diagnostic_system_version: str = Field(min_length=1)
    observed_at: AwareDatetime
    provenance_reference: str = Field(min_length=1)


class ConfidenceAssessment(ResearchStateModel):
    level: ConfidenceLevel = ConfidenceLevel.UNASSESSED
    rationale: str = Field(min_length=1, max_length=1_000)
    semantics: Literal["ORDINAL_JUDGEMENT_NOT_PROBABILITY"] = (
        "ORDINAL_JUDGEMENT_NOT_PROBABILITY"
    )


class ResearchHypothesis(VersionedRecord):
    record_type: Literal["HYPOTHESIS"] = RecordType.HYPOTHESIS.value
    hypothesis_id: str = Field(min_length=1, max_length=300)
    proposition: str = Field(min_length=1, max_length=2_000)
    status: HypothesisStatus
    origin: HypothesisOrigin
    motivation: str = Field(min_length=1, max_length=2_000)
    motivating_evidence: EvidenceReferences = Field(default_factory=EvidenceReferences)
    competing_hypothesis_ids: tuple[str, ...] = ()
    supporting_evidence: EvidenceReferences = Field(default_factory=EvidenceReferences)
    refuting_evidence: EvidenceReferences = Field(default_factory=EvidenceReferences)
    confidence: ConfidenceAssessment
    proposed_discriminating_experiment: str | None = Field(
        default=None, min_length=1, max_length=2_000
    )
    proposed_discriminating_question: str | None = Field(
        default=None, min_length=1, max_length=2_000
    )

    @model_validator(mode="after")
    def status_has_corresponding_evidence(self) -> "ResearchHypothesis":
        if self.hypothesis_id in self.competing_hypothesis_ids:
            raise ValueError("a hypothesis cannot compete with itself")
        if (
            self.status == HypothesisStatus.SUPPORTED
            and self.supporting_evidence.is_empty
        ):
            raise ValueError("supported hypothesis requires supporting evidence")
        if self.status == HypothesisStatus.REFUTED and self.refuting_evidence.is_empty:
            raise ValueError("refuted hypothesis requires refuting evidence")
        return self


class ResearchUncertainty(VersionedRecord):
    record_type: Literal["UNCERTAINTY"] = RecordType.UNCERTAINTY.value
    uncertainty_id: str = Field(min_length=1, max_length=300)
    question: str = Field(min_length=1, max_length=2_000)
    status: UncertaintyStatus
    affects_hypothesis_ids: tuple[str, ...] = ()
    affects_evidence: EvidenceReferences = Field(default_factory=EvidenceReferences)
    affects_decision_ids: tuple[str, ...] = ()
    resolution: str | None = Field(default=None, min_length=1, max_length=2_000)
    resolved_by_evidence: EvidenceReferences = Field(default_factory=EvidenceReferences)
    superseded_by_uncertainty_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def terminal_state_has_explanation(self) -> "ResearchUncertainty":
        if self.status == UncertaintyStatus.RESOLVED and self.resolution is None:
            raise ValueError("resolved uncertainty requires a resolution")
        if (
            self.status == UncertaintyStatus.SUPERSEDED
            and self.superseded_by_uncertainty_id is None
        ):
            raise ValueError("superseded uncertainty requires a successor")
        return self


class PlannerInference(VersionedRecord):
    record_type: Literal["PLANNER_INFERENCE"] = RecordType.PLANNER_INFERENCE.value
    evidence_boundary: Literal["PLANNER_INFERENCE"] = PLANNER_INFERENCE_BOUNDARY
    inference_id: str = Field(min_length=1, max_length=300)
    interpretation: str = Field(min_length=1, max_length=3_000)
    derived_from_evidence: EvidenceReferences
    hypothesis_ids: tuple[str, ...] = ()
    uncertainty_ids: tuple[str, ...] = ()
    inference_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def inference_is_evidence_derived(self) -> "PlannerInference":
        if self.derived_from_evidence.is_empty:
            raise ValueError("planner inference requires typed evidence references")
        return self


class ResourceImplication(ResearchStateModel):
    resource: str = Field(min_length=1, max_length=200)
    implication: str = Field(min_length=1, max_length=1_000)
    amount: float | None = None
    unit: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def amount_and_unit_are_paired(self) -> "ResourceImplication":
        if (self.amount is None) != (self.unit is None):
            raise ValueError("resource amount and unit must be supplied together")
        if self.amount is not None and not math.isfinite(self.amount):
            raise ValueError("resource amount must be finite")
        return self


class PlannerDecision(VersionedRecord):
    record_type: Literal["PLANNER_DECISION"] = RecordType.PLANNER_DECISION.value
    evidence_boundary: Literal["PLANNER_DECISION"] = PLANNER_DECISION_BOUNDARY
    decision_id: str = Field(min_length=1, max_length=300)
    action: str = Field(min_length=1, max_length=2_000)
    rationale: str = Field(min_length=1, max_length=3_000)
    supporting_evidence: EvidenceReferences
    inference_ids: tuple[str, ...] = ()
    hypothesis_ids: tuple[str, ...] = ()
    uncertainty_ids: tuple[str, ...] = ()
    experiment_ids: tuple[str, ...] = ()
    alternatives_considered: tuple[str, ...] = ()
    resource_implications: tuple[ResourceImplication, ...] = Field(min_length=1)
    decision_version: str = Field(min_length=1)
    decided_at: AwareDatetime

    @model_validator(mode="after")
    def decision_is_supported(self) -> "PlannerDecision":
        if self.supporting_evidence.is_empty:
            raise ValueError("planner decision requires typed supporting evidence")
        return self


class ExperimentIntent(ResearchStateModel):
    research_question: str = Field(min_length=1, max_length=2_000)
    hypothesis_ids: tuple[str, ...] = ()
    expected_learning: str = Field(min_length=1, max_length=2_000)
    modality: ExperimentModality
    motivated_by_evidence: EvidenceReferences = Field(
        default_factory=EvidenceReferences
    )
    information_value_class: InformationValueClass
    continue_criterion: str | None = Field(default=None, min_length=1, max_length=1_000)
    stop_criterion: str | None = Field(default=None, min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def has_epistemic_motivation(self) -> "ExperimentIntent":
        if not self.hypothesis_ids and self.motivated_by_evidence.is_empty:
            raise ValueError(
                "experiment intent requires a hypothesis or evidence motivation"
            )
        return self


class ResearchExperiment(VersionedRecord):
    record_type: Literal["EXPERIMENT"] = RecordType.EXPERIMENT.value
    experiment_id: str = Field(min_length=1, max_length=300)
    candidate_ids: tuple[str, ...] = ()
    experiment_spec_reference: str = Field(min_length=1)
    intent: ExperimentIntent
    status: ExperimentStatus
    observation_ids: tuple[str, ...] = ()
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def lifecycle_timestamps_are_consistent(self) -> "ResearchExperiment":
        if self.status == ExperimentStatus.ACTIVE and self.started_at is None:
            raise ValueError("active experiment requires started_at")
        if self.status == ExperimentStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed experiment requires completed_at")
        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("experiment completion cannot precede start")
        return self


class ResearchWorkItem(VersionedRecord):
    record_type: Literal["WORK_ITEM"] = RecordType.WORK_ITEM.value
    work_item_id: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=2_000)
    status: WorkStatus
    reference_type: str = Field(min_length=1, max_length=100)
    reference_id: str = Field(min_length=1)


class CandidateNextAction(VersionedRecord):
    record_type: Literal["NEXT_ACTION"] = RecordType.NEXT_ACTION.value
    next_action_id: str = Field(min_length=1, max_length=300)
    action: str = Field(min_length=1, max_length=2_000)
    rationale: str = Field(min_length=1, max_length=2_000)
    status: WorkStatus = WorkStatus.CANDIDATE
    hypothesis_ids: tuple[str, ...] = ()
    uncertainty_ids: tuple[str, ...] = ()
    motivated_by_evidence: EvidenceReferences = Field(
        default_factory=EvidenceReferences
    )
    expected_information_value: InformationValueClass


ResearchStateRecord: TypeAlias = Annotated[
    ExternalEvidenceReference
    | InternalExperimentalObservation
    | DiagnosticObservation
    | ResearchHypothesis
    | ResearchUncertainty
    | PlannerInference
    | PlannerDecision
    | ResearchExperiment
    | ResearchWorkItem
    | CandidateNextAction,
    Field(discriminator="record_type"),
]


class StateRevision(ResearchStateModel):
    state_revision: int = Field(ge=1)
    state_revision_id: str = Field(pattern=r"^research-state-revision-[0-9a-f]{24}$")
    programme_id: str = Field(min_length=1)
    record_type: RecordType
    record_id: str = Field(min_length=1)
    record_revision: int = Field(ge=1)
    recorded_at: AwareDatetime

    @model_validator(mode="after")
    def identity_is_stable(self) -> "StateRevision":
        expected = stable_identifier(
            "research-state-revision",
            self.programme_id,
            str(self.state_revision),
            self.record_type.value,
            self.record_id,
            str(self.record_revision),
        )
        if self.state_revision_id != expected:
            raise ValueError("research_state_revision_identity_mismatch")
        return self


class ResearchState(ResearchStateModel):
    programme: ResearchProgramme
    state_revision: int = Field(ge=0)
    revision_history: tuple[StateRevision, ...] = ()
    external_evidence: tuple[ExternalEvidenceReference, ...] = ()
    internal_observations: tuple[InternalExperimentalObservation, ...] = ()
    diagnostic_observations: tuple[DiagnosticObservation, ...] = ()
    hypotheses: tuple[ResearchHypothesis, ...] = ()
    uncertainties: tuple[ResearchUncertainty, ...] = ()
    planner_inferences: tuple[PlannerInference, ...] = ()
    planner_decisions: tuple[PlannerDecision, ...] = ()
    experiments: tuple[ResearchExperiment, ...] = ()
    work_items: tuple[ResearchWorkItem, ...] = ()
    candidate_next_actions: tuple[CandidateNextAction, ...] = ()
    state_schema_version: Literal["research-state-v1"] = RESEARCH_STATE_VERSION


def research_programme_identity(name: str, programme_version: str) -> str:
    return stable_identifier("research-programme", name.strip(), programme_version)


def external_evidence_reference(
    programme_id: str,
    card: EvidenceCard,
    *,
    evidence_store_reference: str,
    recorded_at,
) -> ExternalEvidenceReference:
    """Create a durable identity-only reference to a PR #49 Evidence Card."""

    return ExternalEvidenceReference(
        programme_id=programme_id,
        evidence_card_id=card.evidence_id,
        evidence_card_version=card.card_version,
        evidence_content_hash=card.evidence_content_hash,
        evidence_store_reference=evidence_store_reference,
        recorded_at=recorded_at,
    )


def record_identity(record: ResearchStateRecord) -> str:
    names = {
        RecordType.EXTERNAL_EVIDENCE: "evidence_card_id",
        RecordType.INTERNAL_OBSERVATION: "observation_id",
        RecordType.DIAGNOSTIC_OBSERVATION: "diagnostic_observation_id",
        RecordType.HYPOTHESIS: "hypothesis_id",
        RecordType.UNCERTAINTY: "uncertainty_id",
        RecordType.PLANNER_INFERENCE: "inference_id",
        RecordType.PLANNER_DECISION: "decision_id",
        RecordType.EXPERIMENT: "experiment_id",
        RecordType.WORK_ITEM: "work_item_id",
        RecordType.NEXT_ACTION: "next_action_id",
    }
    return str(getattr(record, names[RecordType(record.record_type)]))


def record_content_hash(record: ResearchStateRecord) -> str:
    return content_hash(
        {
            "domain": "research-state-record",
            "version": RESEARCH_STATE_VERSION,
            "record": record,
        }
    )


def state_revision_for(
    programme_id: str,
    state_revision: int,
    record: ResearchStateRecord,
) -> StateRevision:
    record_type = RecordType(record.record_type)
    record_id = record_identity(record)
    return StateRevision(
        state_revision=state_revision,
        state_revision_id=stable_identifier(
            "research-state-revision",
            programme_id,
            str(state_revision),
            record_type.value,
            record_id,
            str(record.revision),
        ),
        programme_id=programme_id,
        record_type=record_type,
        record_id=record_id,
        record_revision=record.revision,
        recorded_at=record.recorded_at,
    )
