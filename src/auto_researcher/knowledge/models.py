"""Immutable contracts for deterministic knowledge retrieval and citation."""

from __future__ import annotations

import math
import re
from enum import StrEnum

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from auto_researcher.contracts.enums import KnowledgeRetrievalStatus, ReadSafetyMode
from auto_researcher.contracts.models import FrozenJsonDict
from auto_researcher.knowledge.read_safety import ReadSafetyAttestation

CURIE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*:[^\s:/][^\s]*$")
PROHIBITED_PROPERTY_NAMES = frozenset(
    {
        "id",
        "_id",
        "element_id",
        "password",
        "secret",
        "credential",
        "credentials",
        "patient_id",
        "participant_id",
        "subject_id",
        "clinical_row",
        "clinical_rows",
        "mutation_value",
        "mutation_values",
        "matrix",
    }
)


class KnowledgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class KnowledgeTrustTier(StrEnum):
    CURATED = "CURATED"
    CORPUS = "CORPUS"
    LIVE = "LIVE"
    UNVERIFIED = "UNVERIFIED"


class KnowledgeSourceType(StrEnum):
    ONTOLOGY_RELEASE = "ONTOLOGY_RELEASE"
    CURATED_DATABASE = "CURATED_DATABASE"
    LITERATURE = "LITERATURE"
    CURATED_ASSERTION = "CURATED_ASSERTION"
    CORPUS_ASSERTION = "CORPUS_ASSERTION"
    LIVE_ASSERTION = "LIVE_ASSERTION"


class KnowledgeErrorCode(StrEnum):
    KNOWLEDGE_DISABLED = "KNOWLEDGE_DISABLED"
    TASK_NOT_KNOWLEDGE_CAPABLE = "TASK_NOT_KNOWLEDGE_CAPABLE"
    PROVIDER_NOT_INSTALLED = "PROVIDER_NOT_INSTALLED"
    PROVIDER_NOT_CONFIGURED = "PROVIDER_NOT_CONFIGURED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    CONNECTIVITY_FAILED = "CONNECTIVITY_FAILED"
    READ_ONLY_NOT_VERIFIED = "READ_ONLY_NOT_VERIFIED"
    READ_SAFETY_MODE_NOT_PERMITTED = "READ_SAFETY_MODE_NOT_PERMITTED"
    ATTESTATION_INVALID = "ATTESTATION_INVALID"
    ATTESTATION_CANONICALIZATION_FAILED = "ATTESTATION_CANONICALIZATION_FAILED"
    ATTESTATION_DUPLICATE_UNORDERED_VALUE = "ATTESTATION_DUPLICATE_UNORDERED_VALUE"
    CONFIGURATION_HASH_MISMATCH = "CONFIGURATION_HASH_MISMATCH"
    LEGACY_ATTESTATION_REGENERATION_REQUIRED = (
        "LEGACY_ATTESTATION_REGENERATION_REQUIRED"
    )
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    CONTENT_VERSION_MISMATCH = "CONTENT_VERSION_MISMATCH"
    UNKNOWN_QUERY_TEMPLATE = "UNKNOWN_QUERY_TEMPLATE"
    INVALID_QUERY_PARAMETERS = "INVALID_QUERY_PARAMETERS"
    QUERY_TIMEOUT = "QUERY_TIMEOUT"
    TRANSIENT_QUERY_FAILURE = "TRANSIENT_QUERY_FAILURE"
    FORBIDDEN_WRITE_DETECTED = "FORBIDDEN_WRITE_DETECTED"
    OPERATOR_ATTESTED_WRITE_BARRIER_VIOLATION = (
        "OPERATOR_ATTESTED_WRITE_BARRIER_VIOLATION"
    )
    RESULT_LIMIT_EXCEEDED = "RESULT_LIMIT_EXCEEDED"
    INVALID_PROVENANCE = "INVALID_PROVENANCE"
    INVALID_IDENTIFIER = "INVALID_IDENTIFIER"
    EMPTY_GROUNDING_RESULT = "EMPTY_GROUNDING_RESULT"
    BUNDLE_VALIDATION_FAILED = "BUNDLE_VALIDATION_FAILED"
    RETRIEVAL_INDETERMINATE = "RETRIEVAL_INDETERMINATE"


class KnowledgeProviderConfiguration(KnowledgeModel):
    provider_id: str = Field(min_length=1)
    graph_alias: str = Field(min_length=1)
    database: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    content_version: str = Field(min_length=1)
    query_timeout_seconds: float = Field(default=20, gt=0, le=300)
    maximum_records: int = Field(default=100, ge=1, le=10_000)
    maximum_attempts: int = Field(default=2, ge=1, le=3)
    maximum_graph_hops: int = Field(default=3, ge=0, le=6)
    minimum_assertion_confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    allowed_trust_tiers: frozenset[KnowledgeTrustTier] | None = None
    read_safety_mode: ReadSafetyMode = ReadSafetyMode.PRIVILEGE_VERIFIED
    read_safety_attestation: ReadSafetyAttestation | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def identifiers_are_safe_for_output(self) -> "KnowledgeProviderConfiguration":
        simple = re.compile(r"^[A-Za-z0-9_.-]+$")
        if not simple.fullmatch(self.provider_id):
            raise ValueError("provider_id must be a safe identifier")
        if not simple.fullmatch(self.database):
            raise ValueError("database must be a safe explicit name")
        if not simple.fullmatch(self.schema_version):
            raise ValueError("schema_version must be a safe identifier")
        if not simple.fullmatch(self.content_version):
            raise ValueError("content_version must be a safe identifier")
        if not simple.fullmatch(self.graph_alias):
            raise ValueError("graph_alias must not contain connection details")
        if self.read_safety_mode == ReadSafetyMode.OPERATOR_ATTESTED:
            if self.provider_id != "neo4j" or self.read_safety_attestation is None:
                raise ValueError(
                    "operator-attested mode requires Neo4j and an attestation"
                )
        elif self.read_safety_attestation is not None:
            raise ValueError("an attestation is valid only in OPERATOR_ATTESTED mode")
        return self


class KnowledgeReadinessCheck(KnowledgeModel):
    code: str = Field(min_length=1)
    passed: bool
    message: str = Field(min_length=1)


class KnowledgeReadinessResult(KnowledgeModel):
    ready: bool
    checks: tuple[KnowledgeReadinessCheck, ...]
    errors: tuple[KnowledgeErrorCode, ...] = ()
    warnings: tuple[str, ...] = ()
    provider_id: str = Field(min_length=1)
    provider_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    content_version: str = Field(min_length=1)
    read_safety_mode: ReadSafetyMode = ReadSafetyMode.PRIVILEGE_VERIFIED
    privilege_verified: bool = False
    attestation_valid: bool = False
    attestation_id: str | None = None
    attestation_version: str | None = None
    attestation_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    residual_risk: str | None = None

    @model_validator(mode="after")
    def ready_has_no_failed_checks(self) -> "KnowledgeReadinessResult":
        if self.ready and (
            self.errors or any(not check.passed for check in self.checks)
        ):
            raise ValueError("ready knowledge provider cannot contain failed checks")
        return self


class KnowledgeTemplateRequest(KnowledgeModel):
    template_id: str = Field(min_length=1)
    template_version: str = Field(min_length=1)
    parameters: FrozenJsonDict
    maximum_records: int = Field(ge=1)
    rationale: str = Field(min_length=1)


class KnowledgeQueryPlan(KnowledgeModel):
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    query_plan_version: str = Field(min_length=1)
    template_requests: tuple[KnowledgeTemplateRequest, ...]
    grounding_policy_id: str = Field(min_length=1)
    maximum_total_records: int = Field(ge=1)
    maximum_references: int = Field(ge=0)
    required: bool = False


class KnowledgeRetrievalRequest(KnowledgeModel):
    retrieval_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    cycle: int = Field(ge=1)
    provider_id: str = Field(min_length=1)
    graph_alias: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    content_version: str = Field(min_length=1)
    query_plan: KnowledgeQueryPlan
    query_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    grounding_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_hashes: FrozenJsonDict
    task_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)


class KnowledgeSource(KnowledgeModel):
    source_id: str = Field(min_length=1)
    source_type: KnowledgeSourceType
    title: str = Field(min_length=1, max_length=300)
    version: str | None = None
    pmid: str | None = None
    doi: str | None = None
    accession: str | None = None
    curie: str | None = None
    publisher_or_database: str = Field(min_length=1, max_length=300)
    asserted_by: str = Field(min_length=1, max_length=100)
    retrieval_reference: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def source_has_stable_provenance(self) -> "KnowledgeSource":
        stable = self.pmid or self.doi or self.accession or self.curie
        if not stable and not self.version:
            raise ValueError("knowledge source requires a stable identifier or version")
        return self

    @field_validator("curie")
    @classmethod
    def source_curie_is_stable(cls, value: str | None) -> str | None:
        return _validate_curie(value) if value is not None else None

    _safe_title = field_validator("title", "publisher_or_database")(
        lambda value: _validate_safe_text(value)
    )


def _validate_curie(value: str) -> str:
    if not CURIE_PATTERN.fullmatch(value):
        raise ValueError("invalid stable CURIE")
    return value


def _validate_safe_text(value: str) -> str:
    if "<" in value or ">" in value or any(ord(item) < 32 for item in value):
        raise ValueError("unsafe control or delimiter in knowledge text")
    return value


def _safe_properties(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    lowered = {item.casefold() for item in _nested_field_names(value)}
    if lowered & PROHIBITED_PROPERTY_NAMES:
        raise ValueError("prohibited or internal property")
    return value


def _nested_field_names(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _nested_field_names(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_field_names(item)


class KnowledgeEntity(KnowledgeModel):
    entity_id: str = Field(min_length=1)
    curie: str
    entity_type: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=300)
    aliases: tuple[str, ...] = ()
    safe_properties: FrozenJsonDict = Field(default_factory=dict)
    source_references: tuple[str, ...]

    _curie = field_validator("curie")(_validate_curie)
    _properties = field_validator("safe_properties")(_safe_properties)


class KnowledgeAssertion(KnowledgeModel):
    assertion_id: str = Field(min_length=1)
    subject_entity_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_entity_id: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    source_references: tuple[str, ...]
    method: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    asserted_by: str = Field(min_length=1)
    trust_tier: KnowledgeTrustTier
    safe_properties: FrozenJsonDict = Field(default_factory=dict)

    _properties = field_validator("safe_properties")(_safe_properties)

    @field_validator("confidence")
    @classmethod
    def confidence_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("assertion confidence must be finite")
        return value


class KnowledgeReference(KnowledgeModel):
    reference_id: str = Field(min_length=1)
    reference_type: str = Field(min_length=1)
    concise_claim: str = Field(min_length=1, max_length=500)
    subject_curie: str
    predicate: str = Field(min_length=1)
    object_curie: str
    source_references: tuple[str, ...]
    trust_tier: KnowledgeTrustTier
    confidence: float = Field(ge=0, le=1)
    citation_label: str = Field(min_length=1, max_length=400)
    bundle_id: str = Field(min_length=1)
    relevant_parameters: tuple[str, ...] = ()
    prior_weight_cap: float = Field(default=0.3, ge=0, le=1)

    _subject = field_validator("subject_curie")(_validate_curie)
    _object = field_validator("object_curie")(_validate_curie)
    _safe_text = field_validator("concise_claim", "citation_label")(_validate_safe_text)


class KnowledgeGraphSnapshot(KnowledgeModel):
    provider_id: str = Field(min_length=1)
    graph_alias: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    configured_content_version: str = Field(min_length=1)
    retrieval_timestamp: AwareDatetime
    safe_graph_metadata: FrozenJsonDict = Field(default_factory=dict)
    returned_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class KnowledgeValidationResult(KnowledgeModel):
    passed: bool
    accepted_reference_count: int = Field(ge=0)
    rejected_assertion_count: int = Field(ge=0)
    reason_codes: tuple[str, ...] = ()
    trust_tier_summary: FrozenJsonDict = Field(default_factory=dict)


class KnowledgeBundle(KnowledgeModel):
    bundle_id: str = Field(min_length=1)
    retrieval_id: str = Field(min_length=1)
    query_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_snapshot: KnowledgeGraphSnapshot
    sources: tuple[KnowledgeSource, ...]
    entities: tuple[KnowledgeEntity, ...]
    assertions: tuple[KnowledgeAssertion, ...]
    references: tuple[KnowledgeReference, ...]
    validation_result: KnowledgeValidationResult
    artefact_references: tuple[str, ...] = ()
    bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class KnowledgeBundleReference(KnowledgeModel):
    bundle_id: str | None = None
    bundle_hash: str | None = None
    retrieval_id: str | None = None
    provider_id: str | None = None
    reference_ids: tuple[str, ...] = ()
    trust_summary: FrozenJsonDict = Field(default_factory=dict)
    artefact_reference: str | None = None
    status: KnowledgeRetrievalStatus


class KnowledgeContextReference(KnowledgeModel):
    reference_id: str
    concise_claim: str
    citation_label: str
    trust_tier: KnowledgeTrustTier
    confidence: float = Field(ge=0, le=1)
    entity_curies: tuple[str, str]
    source_ids: tuple[str, ...]
    bundle_id: str
    relevant_parameters: tuple[str, ...] = ()
    prior_weight_cap: float = Field(ge=0, le=1)


class KnowledgeGroundingPolicy(KnowledgeModel):
    policy_id: str = Field(min_length=1)
    allowed_entity_types: frozenset[str]
    allowed_predicates: frozenset[str]
    allowed_source_types: frozenset[KnowledgeSourceType]
    allowed_asserted_by: frozenset[str]
    allowed_trust_tiers: frozenset[KnowledgeTrustTier]
    minimum_assertion_confidence: float = Field(ge=0, le=1)
    require_stable_subject_identifier: bool = True
    require_stable_object_identifier: bool = True
    require_source_version_for_ontology: bool = True
    require_publication_for_assertions: bool = True
    maximum_references: int = Field(ge=0, le=100)
    maximum_assertions: int = Field(ge=0, le=10_000)
    maximum_entities: int = Field(ge=0, le=10_000)
    tier_prior_weight_caps: FrozenJsonDict = Field(
        default_factory=lambda: {
            "CURATED": 0.9,
            "CORPUS": 0.7,
            "LIVE": 0.3,
            "UNVERIFIED": 0.3,
        }
    )


class KnowledgeRetrievalRecord(KnowledgeModel):
    record_id: str
    retrieval_id: str
    run_id: str
    cycle: int = Field(ge=1)
    status: KnowledgeRetrievalStatus
    request: KnowledgeRetrievalRequest
    bundle: KnowledgeBundle | None = None
    errors: tuple[KnowledgeErrorCode, ...] = ()
    retry_of_retrieval_id: str | None = None
    provider_request_started: bool = False
    created_at: AwareDatetime

    @model_validator(mode="after")
    def status_payload_consistent(self) -> "KnowledgeRetrievalRecord":
        if self.status == KnowledgeRetrievalStatus.COMPLETED and self.bundle is None:
            raise ValueError("completed retrieval requires a bundle")
        if self.status == KnowledgeRetrievalStatus.FAILED and not self.errors:
            raise ValueError("failed retrieval requires a safe error code")
        return self
