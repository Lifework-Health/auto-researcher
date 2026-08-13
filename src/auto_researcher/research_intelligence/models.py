"""Typed external-evidence contracts for Research Intelligence."""

from __future__ import annotations

import math
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from auto_researcher.knowledge.identity import content_hash, stable_identifier

SOURCE_RECORD_VERSION = "research-intelligence-source-v1"
EVIDENCE_CARD_VERSION = "research-intelligence-evidence-card-v1"
SYNTHESIS_VERSION = "deterministic-evidence-synthesis-v1"
BRIEF_VERSION = "research-intelligence-brief-v1"
EXTERNAL_EVIDENCE_BOUNDARY = "EXTERNAL_RESEARCH_INTELLIGENCE"


class ResearchIntelligenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class SourceType(StrEnum):
    PEER_REVIEWED_PAPER = "PEER_REVIEWED_PAPER"
    PREPRINT = "PREPRINT"
    BENCHMARK_CHALLENGE_RESULT = "BENCHMARK_CHALLENGE_RESULT"
    OFFICIAL_DOCUMENTATION = "OFFICIAL_DOCUMENTATION"
    IMPLEMENTATION_CODE_EVIDENCE = "IMPLEMENTATION_CODE_EVIDENCE"


class TrustClassification(StrEnum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    UNVERIFIED = "UNVERIFIED"


class Availability(StrEnum):
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class EvidenceQuality(StrEnum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    UNASSESSED = "UNASSESSED"


class EvidenceCategory(StrEnum):
    ACTIONABLE_PRIOR = "ACTIONABLE_PRIOR"
    STRONG_BASELINE = "STRONG_BASELINE"
    ESTABLISHED_CHOICE = "ESTABLISHED_CHOICE"
    FAILURE_MODE = "FAILURE_MODE"
    UNDEREXPLORED_HYPOTHESIS = "UNDEREXPLORED_HYPOTHESIS"
    GENERAL_FINDING = "GENERAL_FINDING"


class FindingStance(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    MIXED = "MIXED"


class ApplicabilityLevel(StrEnum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvidenceRelationType(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONFLICTS = "CONFLICTS"


class BriefSection(StrEnum):
    ACTIONABLE_PRIORS = "ACTIONABLE_PRIORS"
    STRONG_BASELINES = "STRONG_BASELINES"
    ESTABLISHED_CHOICES = "ESTABLISHED_CHOICES"
    LIKELY_FAILURE_MODES = "LIKELY_FAILURE_MODES"
    UNDEREXPLORED_HYPOTHESES = "UNDEREXPLORED_HYPOTHESES"
    UNRESOLVED_CONFLICTS = "UNRESOLVED_CONFLICTS"
    EXPERIMENT_DESIGN_IMPLICATIONS = "EXPERIMENT_DESIGN_IMPLICATIONS"


class SourceCandidate(ResearchIntelligenceModel):
    """Already-retrieved bibliographic and provenance material."""

    title: str = Field(min_length=1, max_length=500)
    authors: tuple[str, ...] = ()
    organisation: str | None = Field(default=None, min_length=1, max_length=300)
    source_type: SourceType
    publication_or_update_date: date | None = None
    retrieved_at: AwareDatetime
    reference_identity: str = Field(min_length=1, max_length=1_000)
    uri: str | None = Field(default=None, min_length=1, max_length=2_000)
    source_version: str = Field(min_length=1, max_length=200)
    task_contexts: tuple[str, ...] = ()
    dataset_contexts: tuple[str, ...] = ()
    domains: tuple[str, ...] = Field(min_length=1)
    code_availability: Availability = Availability.UNKNOWN
    data_availability: Availability = Availability.UNKNOWN
    trust_classification: TrustClassification
    quality_score: float = Field(ge=0, le=1)
    ingestion_method: Literal["ALREADY_RETRIEVED"] = "ALREADY_RETRIEVED"
    provenance_version: str = Field(default="offline-scout-v1", min_length=1)

    @field_validator(
        "authors",
        "task_contexts",
        "dataset_contexts",
        "domains",
        mode="before",
    )
    @classmethod
    def tuple_values_are_unique(cls, value):
        values = tuple(str(item).strip() for item in value)
        if any(not item for item in values) or len(values) != len(set(values)):
            raise ValueError(
                "research intelligence values must be non-empty and unique"
            )
        return values

    @model_validator(mode="after")
    def dates_are_causal(self) -> "SourceCandidate":
        if (
            self.publication_or_update_date is not None
            and self.publication_or_update_date > self.retrieved_at.date()
        ):
            raise ValueError("source publication date cannot follow retrieval")
        return self


class SourceRecord(SourceCandidate):
    record_version: Literal["research-intelligence-source-v1"] = SOURCE_RECORD_VERSION
    evidence_boundary: Literal["EXTERNAL_RESEARCH_INTELLIGENCE"] = (
        EXTERNAL_EVIDENCE_BOUNDARY
    )
    source_id: str = Field(pattern=r"^source-[0-9a-f]{24}$")
    source_version_id: str = Field(pattern=r"^source-version-[0-9a-f]{24}$")
    source_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def identity_is_canonical(self) -> "SourceRecord":
        candidate = SourceCandidate.model_validate(
            self.model_dump(
                mode="python",
                exclude={
                    "record_version",
                    "evidence_boundary",
                    "source_id",
                    "source_version_id",
                    "source_content_hash",
                },
            )
        )
        expected_hash = source_candidate_content_hash(candidate)
        expected_source = source_identity(candidate)
        expected_version = source_version_identity(
            expected_source, candidate, expected_hash
        )
        if (
            self.source_content_hash != expected_hash
            or self.source_id != expected_source
            or self.source_version_id != expected_version
        ):
            raise ValueError("research_intelligence_source_identity_mismatch")
        return self


class QuantitativeResult(ResearchIntelligenceModel):
    metric: str = Field(min_length=1, max_length=200)
    value: float
    unit: str | None = Field(default=None, min_length=1, max_length=100)
    comparator: str | None = Field(default=None, min_length=1, max_length=300)
    uncertainty: str | None = Field(default=None, min_length=1, max_length=300)

    @field_validator("value")
    @classmethod
    def value_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("quantitative evidence must be finite")
        return value


class ApplicabilityAssessment(ResearchIntelligenceModel):
    task_id: str | None = Field(default=None, min_length=1)
    domain: str | None = Field(default=None, min_length=1)
    dataset_id: str | None = Field(default=None, min_length=1)
    level: ApplicabilityLevel
    score: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def has_a_target(self) -> "ApplicabilityAssessment":
        if self.task_id is None and self.domain is None and self.dataset_id is None:
            raise ValueError("applicability assessment requires a target")
        return self


class FindingCandidate(ResearchIntelligenceModel):
    claim_key: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,199}$")
    claim: str = Field(min_length=1, max_length=2_000)
    category: EvidenceCategory
    stance: FindingStance = FindingStance.SUPPORTS
    quantitative_results: tuple[QuantitativeResult, ...] = ()
    method_or_intervention: str = Field(min_length=1, max_length=1_000)
    dataset_or_population: str = Field(min_length=1, max_length=1_000)
    conditions: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    applicability_assessments: tuple[ApplicabilityAssessment, ...] = Field(min_length=1)
    evidence_quality: EvidenceQuality
    confidence: float = Field(ge=0, le=1)
    hypothesis_tags: tuple[str, ...] = ()
    experiment_design_implications: tuple[str, ...] = ()

    @field_validator(
        "conditions",
        "limitations",
        "hypothesis_tags",
        "experiment_design_implications",
        mode="before",
    )
    @classmethod
    def text_tuples_are_unique(cls, value):
        values = tuple(str(item).strip() for item in value)
        if any(not item for item in values) or len(values) != len(set(values)):
            raise ValueError("finding values must be non-empty and unique")
        return values


class RetrievedSourceMaterial(ResearchIntelligenceModel):
    source: SourceCandidate
    findings: tuple[FindingCandidate, ...] = Field(min_length=1)


class SourceReference(ResearchIntelligenceModel):
    source_id: str = Field(pattern=r"^source-[0-9a-f]{24}$")
    source_version_id: str = Field(pattern=r"^source-version-[0-9a-f]{24}$")


class EvidenceRanking(ResearchIntelligenceModel):
    quality_score: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)
    freshness_score: float = Field(ge=0, le=1)
    combined_score: float = Field(ge=0, le=1)
    ranking_version: Literal["quality-relevance-freshness-v1"] = (
        "quality-relevance-freshness-v1"
    )


class ResearchProgrammeContext(ResearchIntelligenceModel):
    task_id: str = Field(min_length=1)
    domains: tuple[str, ...] = Field(min_length=1)
    dataset_ids: tuple[str, ...] = ()
    hypothesis_ids: tuple[str, ...] = ()
    as_of_date: date

    @field_validator("domains", "dataset_ids", "hypothesis_ids", mode="before")
    @classmethod
    def context_values_are_unique(cls, value):
        values = tuple(str(item).strip() for item in value)
        if any(not item for item in values) or len(values) != len(set(values)):
            raise ValueError("research context values must be non-empty and unique")
        return values


class EvidenceCard(ResearchIntelligenceModel):
    card_version: Literal["research-intelligence-evidence-card-v1"] = (
        EVIDENCE_CARD_VERSION
    )
    evidence_boundary: Literal["EXTERNAL_RESEARCH_INTELLIGENCE"] = (
        EXTERNAL_EVIDENCE_BOUNDARY
    )
    evidence_id: str = Field(pattern=r"^evidence-[0-9a-f]{24}$")
    evidence_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_references: tuple[SourceReference, ...] = Field(min_length=1)
    claim_key: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,199}$")
    claim: str = Field(min_length=1, max_length=2_000)
    category: EvidenceCategory
    stance: FindingStance
    quantitative_results: tuple[QuantitativeResult, ...] = ()
    method_or_intervention: str = Field(min_length=1, max_length=1_000)
    dataset_or_population: str = Field(min_length=1, max_length=1_000)
    conditions: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    applicability_assessments: tuple[ApplicabilityAssessment, ...] = Field(min_length=1)
    current_applicability: ApplicabilityAssessment
    programme_context: ResearchProgrammeContext
    evidence_quality: EvidenceQuality
    confidence: float = Field(ge=0, le=1)
    ranking: EvidenceRanking
    hypothesis_tags: tuple[str, ...] = ()
    experiment_design_implications: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    conflicting_evidence_ids: tuple[str, ...] = ()
    synthesised_at: AwareDatetime
    synthesiser_version: Literal["deterministic-evidence-synthesis-v1"] = (
        SYNTHESIS_VERSION
    )

    @model_validator(mode="after")
    def references_and_identity_are_valid(self) -> "EvidenceCard":
        source_pairs = tuple(
            (item.source_id, item.source_version_id) for item in self.source_references
        )
        if len(source_pairs) != len(set(source_pairs)):
            raise ValueError("evidence source references must be unique")
        supporting = set(self.supporting_evidence_ids)
        conflicting = set(self.conflicting_evidence_ids)
        if (
            self.evidence_id in supporting
            or self.evidence_id in conflicting
            or supporting & conflicting
            or len(supporting) != len(self.supporting_evidence_ids)
            or len(conflicting) != len(self.conflicting_evidence_ids)
        ):
            raise ValueError("evidence relationships are invalid")
        expected_hash = evidence_card_content_hash(self)
        expected_id = stable_identifier(
            "evidence", self.claim_key, self.stance.value, expected_hash
        )
        if (
            self.evidence_content_hash != expected_hash
            or self.evidence_id != expected_id
        ):
            raise ValueError("research_intelligence_evidence_identity_mismatch")
        return self


class SynthesisResult(ResearchIntelligenceModel):
    synthesis_version: Literal["deterministic-evidence-synthesis-v1"] = (
        SYNTHESIS_VERSION
    )
    programme_context: ResearchProgrammeContext
    source_records: tuple[SourceRecord, ...]
    evidence_cards: tuple[EvidenceCard, ...]
    duplicate_sources_ignored: int = Field(ge=0)
    duplicate_findings_ignored: int = Field(ge=0)
    synthesised_at: AwareDatetime


class ResearchIntelligenceRefreshRecord(ResearchIntelligenceModel):
    refresh_id: str = Field(pattern=r"^research-refresh-[0-9a-f]{24}$")
    refresh_version: Literal["research-intelligence-refresh-v1"] = (
        "research-intelligence-refresh-v1"
    )
    programme_context: ResearchProgrammeContext
    source_version_ids: tuple[str, ...]
    evidence_card_ids: tuple[str, ...]
    source_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    completed_at: AwareDatetime
    synthesiser_version: Literal["deterministic-evidence-synthesis-v1"] = (
        SYNTHESIS_VERSION
    )

    @model_validator(mode="after")
    def counts_and_identity_are_valid(self) -> "ResearchIntelligenceRefreshRecord":
        if (
            self.source_count != len(self.source_version_ids)
            or self.evidence_count != len(self.evidence_card_ids)
            or len(set(self.source_version_ids)) != len(self.source_version_ids)
            or len(set(self.evidence_card_ids)) != len(self.evidence_card_ids)
        ):
            raise ValueError("research intelligence refresh counts are invalid")
        expected = stable_identifier(
            "research-refresh",
            content_hash(self.programme_context),
            *self.source_version_ids,
            *self.evidence_card_ids,
        )
        if self.refresh_id != expected:
            raise ValueError("research_intelligence_refresh_identity_mismatch")
        return self


class EvidenceQuery(ResearchIntelligenceModel):
    task_id: str | None = Field(default=None, min_length=1)
    domain: str | None = Field(default=None, min_length=1)
    dataset_id: str | None = Field(default=None, min_length=1)
    hypothesis_id: str | None = Field(default=None, min_length=1)
    categories: frozenset[EvidenceCategory] = frozenset()
    minimum_applicability_score: float = Field(default=0, ge=0, le=1)
    minimum_rank_score: float = Field(default=0, ge=0, le=1)


class BriefEntry(ResearchIntelligenceModel):
    section: BriefSection
    statement: str = Field(min_length=1, max_length=2_000)
    evidence_card_ids: tuple[str, ...] = Field(min_length=1)
    priority_score: float = Field(ge=0, le=1)
    experiment_design_implications: tuple[str, ...] = ()

    @field_validator("evidence_card_ids")
    @classmethod
    def evidence_references_are_unique(cls, value: tuple[str, ...]):
        if len(value) != len(set(value)):
            raise ValueError("brief evidence references must be unique")
        return value


class ResearchIntelligenceBrief(ResearchIntelligenceModel):
    brief_version: Literal["research-intelligence-brief-v1"] = BRIEF_VERSION
    evidence_boundary: Literal["EXTERNAL_RESEARCH_INTELLIGENCE"] = (
        EXTERNAL_EVIDENCE_BOUNDARY
    )
    programme_context: ResearchProgrammeContext
    actionable_priors: tuple[BriefEntry, ...] = ()
    known_strong_baselines: tuple[BriefEntry, ...] = ()
    established_choices: tuple[BriefEntry, ...] = ()
    likely_failure_modes: tuple[BriefEntry, ...] = ()
    underexplored_hypotheses: tuple[BriefEntry, ...] = ()
    unresolved_uncertainties: tuple[BriefEntry, ...] = ()
    experiment_design_implications: tuple[BriefEntry, ...] = ()
    generated_at: AwareDatetime
    generator_version: Literal["deterministic-brief-builder-v1"] = (
        "deterministic-brief-builder-v1"
    )

    @model_validator(mode="after")
    def every_entry_is_traceable(self) -> "ResearchIntelligenceBrief":
        sections = (
            self.actionable_priors,
            self.known_strong_baselines,
            self.established_choices,
            self.likely_failure_modes,
            self.underexplored_hypotheses,
            self.unresolved_uncertainties,
            self.experiment_design_implications,
        )
        if any(not item.evidence_card_ids for section in sections for item in section):
            raise ValueError("brief entries require evidence card references")
        return self


def source_identity(candidate: SourceCandidate) -> str:
    return stable_identifier(
        "source",
        candidate.source_type.value,
        candidate.reference_identity.strip().casefold(),
    )


def source_candidate_content_hash(candidate: SourceCandidate) -> str:
    payload = candidate.model_dump(mode="python", exclude={"retrieved_at"})
    return content_hash(
        {
            "domain": "research-intelligence-source-content",
            "version": SOURCE_RECORD_VERSION,
            "payload": payload,
        }
    )


def source_version_identity(
    source_id: str, candidate: SourceCandidate, candidate_hash: str
) -> str:
    return stable_identifier(
        "source-version", source_id, candidate.source_version, candidate_hash
    )


def materialise_source(candidate: SourceCandidate) -> SourceRecord:
    candidate_hash = source_candidate_content_hash(candidate)
    source_id = source_identity(candidate)
    return SourceRecord(
        **candidate.model_dump(mode="python"),
        source_id=source_id,
        source_version_id=source_version_identity(source_id, candidate, candidate_hash),
        source_content_hash=candidate_hash,
    )


def evidence_card_content_hash(card: EvidenceCard | dict) -> str:
    payload = (
        card.model_dump(mode="python") if isinstance(card, EvidenceCard) else dict(card)
    )
    for field in (
        "card_version",
        "evidence_boundary",
        "evidence_id",
        "evidence_content_hash",
        "supporting_evidence_ids",
        "conflicting_evidence_ids",
        "synthesised_at",
        "synthesiser_version",
    ):
        payload.pop(field, None)
    return content_hash(
        {
            "domain": "research-intelligence-evidence-content",
            "version": EVIDENCE_CARD_VERSION,
            "payload": payload,
        }
    )


def materialise_evidence_card(**payload) -> EvidenceCard:
    candidate = {
        **payload,
        "evidence_id": "evidence-" + "0" * 24,
        "evidence_content_hash": "0" * 64,
    }
    candidate_hash = evidence_card_content_hash(candidate)
    evidence_id = stable_identifier(
        "evidence",
        str(candidate["claim_key"]),
        FindingStance(candidate["stance"]).value,
        candidate_hash,
    )
    return EvidenceCard.model_validate(
        {
            **candidate,
            "evidence_id": evidence_id,
            "evidence_content_hash": candidate_hash,
        }
    )
