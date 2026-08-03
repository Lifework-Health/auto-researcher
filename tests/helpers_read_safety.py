from __future__ import annotations

from datetime import UTC, datetime

from auto_researcher.contracts.enums import ReadSafetyMode
from auto_researcher.knowledge.models import KnowledgeProviderConfiguration
from auto_researcher.knowledge.read_safety import (
    CANONICAL_HASH_ALGORITHM,
    ProhibitedCapability,
    ReadSafetyAttestation,
    read_safety_configuration_hash,
    seal_attestation,
)
from auto_researcher.knowledge.templates import default_template_registry


DEFAULT_ATTESTED_TEMPLATES = (
    "generic.schema_preflight@1.0.0",
    "icca_nbs.gene_signature_pathway@1.0.0",
)


def operator_configuration(
    *,
    expires_at: datetime = datetime(2026, 8, 30, tzinfo=UTC),
    template_ids: tuple[str, ...] = DEFAULT_ATTESTED_TEMPLATES,
) -> KnowledgeProviderConfiguration:
    registry = default_template_registry()
    draft = ReadSafetyAttestation(
        attestation_id="aura-professional-review",
        attestation_version="1.0.0",
        attestation_hash_algorithm=CANONICAL_HASH_ALGORITHM,
        configuration_hash_algorithm=CANONICAL_HASH_ALGORITHM,
        platform="NEO4J_AURA",
        service_tier="PROFESSIONAL",
        provider_id="neo4j",
        graph_alias="cell-biology",
        schema_version="knowledge-graph-auto-v0.1",
        content_version="backbone-test",
        identity_class="NATIVE_INSTANCE_CREDENTIAL",
        credential_class="MANAGED_INSTANCE_PRIMARY",
        reviewed_at=datetime(2026, 7, 29, tzinfo=UTC),
        expires_at=expires_at,
        reviewer="authorised-operator",
        evidence_references=("aura-viewer-driver-probe",),
        permitted_query_template_ids=template_ids,
        prohibited_capabilities=frozenset(ProhibitedCapability),
        residual_risk_statement=(
            "The managed primary credential is not database-enforced read-only; "
            "registered query barriers reduce but do not remove risk."
        ),
        configuration_hash="0" * 64,
        attestation_hash="0" * 64,
    )
    configuration = KnowledgeProviderConfiguration(
        provider_id="neo4j",
        graph_alias="cell-biology",
        database="neo4j",
        schema_version="knowledge-graph-auto-v0.1",
        content_version="backbone-test",
        query_timeout_seconds=5,
        maximum_records=10,
        maximum_graph_hops=3,
        minimum_assertion_confidence=0.6,
        allowed_trust_tiers=frozenset({"CURATED", "CORPUS"}),
        read_safety_mode=ReadSafetyMode.OPERATOR_ATTESTED,
        read_safety_attestation=draft,
    )
    bound = draft.model_copy(
        update={
            "configuration_hash": read_safety_configuration_hash(
                configuration,
                registry,
                draft,
            )
        }
    )
    sealed = seal_attestation(bound)
    return configuration.model_copy(update={"read_safety_attestation": sealed})
