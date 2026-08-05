"""Operator-attested read-safety contracts for managed Neo4j services."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from auto_researcher.knowledge.identity import (
    CanonicalizationError,
    canonical_json,
    domain_separated_hash,
)


READ_SAFETY_CONTRACT_VERSION = "knowledge-read-safety-v2"
CANONICAL_HASH_ALGORITHM = "canonical-json-sha256-v1"
CANONICAL_HASH_VERSION = "1"
ATTESTATION_HASH_DOMAIN = "auto-researcher-read-safety-attestation"
CONFIGURATION_HASH_DOMAIN = "auto-researcher-read-safety-configuration"
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
TEMPLATE_IDENTIFIER = re.compile(r"^[a-z0-9_.-]+@\d+\.\d+\.\d+$")


class ReadSafetyPlatform(StrEnum):
    NEO4J_AURA = "NEO4J_AURA"


class ReadSafetyServiceTier(StrEnum):
    PROFESSIONAL = "PROFESSIONAL"


class ReadSafetyIdentityClass(StrEnum):
    NATIVE_INSTANCE_CREDENTIAL = "NATIVE_INSTANCE_CREDENTIAL"


class ReadSafetyCredentialClass(StrEnum):
    MANAGED_INSTANCE_PRIMARY = "MANAGED_INSTANCE_PRIMARY"


class ProhibitedCapability(StrEnum):
    GRAPH_WRITE = "GRAPH_WRITE"
    SCHEMA_WRITE = "SCHEMA_WRITE"
    ADMIN_PROCEDURE = "ADMIN_PROCEDURE"
    ARBITRARY_CYPHER = "ARBITRARY_CYPHER"
    MODEL_GENERATED_CYPHER = "MODEL_GENERATED_CYPHER"


class ReadSafetyResidualRisk(StrEnum):
    DATABASE_CREDENTIAL_NOT_ENFORCED_READ_ONLY = (
        "DATABASE_CREDENTIAL_NOT_ENFORCED_READ_ONLY"
    )


REQUIRED_PROHIBITED_CAPABILITIES = frozenset(ProhibitedCapability)


class ReadSafetyAttestation(BaseModel):
    """Immutable, credential-free operator review bound to one safe configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    read_safety_contract_version: str = READ_SAFETY_CONTRACT_VERSION
    attestation_hash_algorithm: Literal["canonical-json-sha256-v1"]
    configuration_hash_algorithm: Literal["canonical-json-sha256-v1"]
    attestation_id: str = Field(min_length=1)
    attestation_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    platform: ReadSafetyPlatform
    service_tier: ReadSafetyServiceTier
    provider_id: str = Field(min_length=1)
    graph_alias: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    content_version: str = Field(min_length=1)
    identity_class: ReadSafetyIdentityClass
    credential_class: ReadSafetyCredentialClass
    reviewed_at: AwareDatetime
    expires_at: AwareDatetime
    reviewer: str = Field(min_length=1)
    evidence_references: frozenset[str] = Field(min_length=1)
    permitted_query_template_ids: frozenset[str] = Field(min_length=1)
    prohibited_capabilities: frozenset[ProhibitedCapability] = Field(min_length=1)
    residual_risk_statement: str = Field(min_length=1, max_length=1000)
    residual_risk_code: ReadSafetyResidualRisk = (
        ReadSafetyResidualRisk.DATABASE_CREDENTIAL_NOT_ENFORCED_READ_ONLY
    )
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    attestation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator(
        "read_safety_contract_version",
        "attestation_id",
        "provider_id",
        "graph_alias",
        "schema_version",
        "content_version",
        "reviewer",
    )
    @classmethod
    def identifiers_are_safe(cls, value: str) -> str:
        if not SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError("attestation identifiers must be safe")
        return value

    @field_validator(
        "evidence_references",
        "permitted_query_template_ids",
        "prohibited_capabilities",
        mode="before",
    )
    @classmethod
    def unordered_values_are_unambiguous(cls, value: Any) -> Any:
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError("ATTESTATION_CANONICALIZATION_FAILED")
        try:
            items = list(value)
            identities = [canonical_json(item) for item in items]
        except (CanonicalizationError, TypeError):
            raise ValueError("ATTESTATION_CANONICALIZATION_FAILED") from None
        if len(identities) != len(set(identities)):
            raise ValueError("ATTESTATION_DUPLICATE_UNORDERED_VALUE")
        return value

    @field_validator("evidence_references")
    @classmethod
    def evidence_references_are_safe(cls, value: frozenset[str]) -> frozenset[str]:
        if any(not SAFE_IDENTIFIER.fullmatch(item) for item in value):
            raise ValueError("evidence references must be safe identifiers")
        return value

    @field_validator("permitted_query_template_ids")
    @classmethod
    def templates_are_versioned(cls, value: frozenset[str]) -> frozenset[str]:
        if any(not TEMPLATE_IDENTIFIER.fullmatch(item) for item in value):
            raise ValueError("permitted templates must be versioned safe identifiers")
        return value

    @field_validator("residual_risk_statement")
    @classmethod
    def residual_risk_is_safe(cls, value: str) -> str:
        lowered = value.casefold()
        if (
            "://" in value
            or "@" in value
            or any(ord(character) < 32 for character in value)
            or any(token in lowered for token in ("password=", "token=", "secret="))
        ):
            raise ValueError("residual risk statement contains sensitive content")
        return value

    @model_validator(mode="after")
    def attestation_is_honest_and_bounded(self) -> "ReadSafetyAttestation":
        if self.read_safety_contract_version != READ_SAFETY_CONTRACT_VERSION:
            raise ValueError("unsupported read-safety contract version")
        if self.reviewed_at >= self.expires_at:
            raise ValueError("attestation expiry must follow its review time")
        if self.prohibited_capabilities != REQUIRED_PROHIBITED_CAPABILITIES:
            raise ValueError("attestation must prohibit every required capability")
        if self.provider_id != "neo4j":
            raise ValueError("operator attestation is restricted to Neo4j")
        return self


def attestation_content_hash(attestation: ReadSafetyAttestation) -> str:
    payload = attestation.model_dump(
        mode="python",
        exclude={"attestation_hash"},
    )
    return domain_separated_hash(
        payload,
        hash_domain=ATTESTATION_HASH_DOMAIN,
        hash_version=CANONICAL_HASH_VERSION,
        schema_version=READ_SAFETY_CONTRACT_VERSION,
    )


def seal_attestation(attestation: ReadSafetyAttestation) -> ReadSafetyAttestation:
    """Return a copy carrying its deterministic content hash."""

    return attestation.model_copy(
        update={"attestation_hash": attestation_content_hash(attestation)}
    )


def read_safety_configuration_hash(
    configuration: Any,
    template_registry: Any,
    attestation: ReadSafetyAttestation,
) -> str:
    templates: dict[str, str] = {}
    for identity in attestation.permitted_query_template_ids:
        template_id, version = identity.rsplit("@", 1)
        template = template_registry.get(template_id, version)
        templates[identity] = template.cypher_sha256
    return domain_separated_hash(
        {
            "read_safety_contract_version": READ_SAFETY_CONTRACT_VERSION,
            "configuration_hash_algorithm": attestation.configuration_hash_algorithm,
            "provider_id": configuration.provider_id,
            "platform": attestation.platform,
            "service_tier": attestation.service_tier,
            "graph_alias": configuration.graph_alias,
            "database": configuration.database,
            "schema_version": configuration.schema_version,
            "content_version": configuration.content_version,
            "query_timeout_seconds": configuration.query_timeout_seconds,
            "maximum_records": configuration.maximum_records,
            "maximum_attempts": configuration.maximum_attempts,
            "maximum_graph_hops": configuration.maximum_graph_hops,
            "minimum_assertion_confidence": configuration.minimum_assertion_confidence,
            "allowed_trust_tiers": configuration.allowed_trust_tiers or frozenset(),
            "enabled": configuration.enabled,
            "identity_class": attestation.identity_class,
            "credential_class": attestation.credential_class,
            "templates": templates,
        },
        hash_domain=CONFIGURATION_HASH_DOMAIN,
        hash_version=CANONICAL_HASH_VERSION,
        schema_version=READ_SAFETY_CONTRACT_VERSION,
    )


def parse_read_safety_attestation(value: Any) -> ReadSafetyAttestation:
    """Parse only corrected canonical-hash attestations; legacy files fail closed."""

    if isinstance(value, Mapping) and (
        "attestation_hash_algorithm" not in value
        or "configuration_hash_algorithm" not in value
    ):
        raise ValueError("LEGACY_ATTESTATION_REGENERATION_REQUIRED")
    try:
        return ReadSafetyAttestation.model_validate(value)
    except ValidationError as exc:
        message = str(exc)
        for code in (
            "ATTESTATION_DUPLICATE_UNORDERED_VALUE",
            "ATTESTATION_CANONICALIZATION_FAILED",
        ):
            if code in message:
                raise ValueError(code) from None
        raise


def validate_operator_attestation(
    attestation: ReadSafetyAttestation,
    configuration: Any,
    template_registry: Any,
    *,
    now: datetime,
) -> tuple[str, ...]:
    """Return stable validation codes; an empty tuple means valid."""

    errors: list[str] = []
    if now.tzinfo is None:
        raise ValueError("attestation validation clock must be timezone-aware")
    if now < attestation.reviewed_at:
        errors.append("ATTESTATION_NOT_YET_VALID")
    if now >= attestation.expires_at:
        errors.append("ATTESTATION_EXPIRED")
    try:
        calculated_attestation_hash = attestation_content_hash(attestation)
    except CanonicalizationError:
        errors.append("ATTESTATION_CANONICALIZATION_FAILED")
    else:
        if attestation.attestation_hash != calculated_attestation_hash:
            errors.append("ATTESTATION_HASH_MISMATCH")
    expected = {
        "provider_id": configuration.provider_id,
        "graph_alias": configuration.graph_alias,
        "schema_version": configuration.schema_version,
        "content_version": configuration.content_version,
    }
    for field, expected_value in expected.items():
        if getattr(attestation, field) != expected_value:
            errors.append(f"ATTESTATION_{field.upper()}_MISMATCH")
    if attestation.platform != ReadSafetyPlatform.NEO4J_AURA:
        errors.append("ATTESTATION_PLATFORM_MISMATCH")
    if attestation.service_tier != ReadSafetyServiceTier.PROFESSIONAL:
        errors.append("ATTESTATION_SERVICE_TIER_MISMATCH")
    if attestation.identity_class != ReadSafetyIdentityClass.NATIVE_INSTANCE_CREDENTIAL:
        errors.append("ATTESTATION_IDENTITY_CLASS_MISMATCH")
    if (
        attestation.credential_class
        != ReadSafetyCredentialClass.MANAGED_INSTANCE_PRIMARY
    ):
        errors.append("ATTESTATION_CREDENTIAL_CLASS_MISMATCH")
    try:
        expected_configuration_hash = read_safety_configuration_hash(
            configuration,
            template_registry,
            attestation,
        )
    except CanonicalizationError:
        errors.append("ATTESTATION_CANONICALIZATION_FAILED")
    except KeyError:
        errors.append("ATTESTATION_TEMPLATE_SET_MISMATCH")
    else:
        if attestation.configuration_hash != expected_configuration_hash:
            errors.append("ATTESTATION_CONFIGURATION_HASH_MISMATCH")
    return tuple(dict.fromkeys(errors))
