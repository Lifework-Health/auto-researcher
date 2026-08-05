"""Replay-safe provenance events for append-only knowledge retrieval records."""

from __future__ import annotations

import hashlib

from auto_researcher.contracts.enums import (
    EventType,
    KnowledgeRetrievalStatus,
    ProvenanceKind,
)
from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.knowledge.store import KnowledgeRetrievalStore
from auto_researcher.provenance.protocols import ProvenanceStore

CODE_VERSION = "auto-researcher-v2.1-pr5.5"


def _read_safety_outputs(bundle) -> tuple[str, ...]:
    if bundle is None:
        return ()
    metadata = bundle.graph_snapshot.safe_graph_metadata
    outputs = (
        f"read_safety_mode:{metadata.get('read_safety_mode', 'none')}",
        f"attestation_id:{metadata.get('attestation_id', 'none')}",
        f"attestation_version:{metadata.get('attestation_version', 'none')}",
        f"attestation_hash:{metadata.get('attestation_hash', 'none')}",
        "attestation_hash_algorithm:"
        f"{metadata.get('attestation_hash_algorithm', 'none')}",
        "configuration_hash_algorithm:"
        f"{metadata.get('configuration_hash_algorithm', 'none')}",
        f"platform:{metadata.get('platform', 'none')}",
        f"service_tier:{metadata.get('service_tier', 'none')}",
        f"credential_class:{metadata.get('credential_class', 'none')}",
        f"privilege_introspection:{metadata.get('privilege_introspection', 'none')}",
        f"residual_risk:{metadata.get('residual_risk', 'none')}",
    )
    audit = metadata.get("query_execution_audit", ())
    if not isinstance(audit, (list, tuple)):
        return outputs
    executions = []
    for item in audit:
        if not isinstance(item, dict):
            continue
        executions.extend(
            (
                f"executed_template:{item.get('template_id', 'none')}",
                f"executed_template_hash:{item.get('template_hash', 'none')}",
                "zero_updates_confirmed:"
                f"{str(bool(item.get('zero_updates_confirmed'))).lower()}",
                "zero_system_updates_confirmed:"
                f"{str(bool(item.get('zero_system_updates_confirmed'))).lower()}",
            )
        )
    return outputs + tuple(executions)


def _event_type(status: KnowledgeRetrievalStatus) -> EventType | None:
    if status == KnowledgeRetrievalStatus.RESERVED:
        return EventType.KNOWLEDGE_RETRIEVAL_RESERVED
    if status == KnowledgeRetrievalStatus.COMPLETED:
        return EventType.KNOWLEDGE_RETRIEVAL_COMPLETED
    if status in {
        KnowledgeRetrievalStatus.FAILED,
        KnowledgeRetrievalStatus.INDETERMINATE,
    }:
        return EventType.KNOWLEDGE_RETRIEVAL_FAILED
    return None


def append_knowledge_retrieval_events(
    provenance_store: ProvenanceStore,
    retrieval_store: KnowledgeRetrievalStore,
    *,
    run_id: str,
    cycle: int,
) -> list[str]:
    """Mirror durable retrieval snapshots without leaking query data or secrets."""
    event_ids: list[str] = []
    for record in retrieval_store.list_records(run_id):
        if record.cycle != cycle:
            continue
        event_type = _event_type(record.status)
        if event_type is None:
            continue
        digest = hashlib.sha256(record.record_id.encode()).hexdigest()[:24]
        event_id = f"event-knowledge-{digest}"
        bundle = record.bundle
        retrieval_provenance = (
            ProvenanceKind.SIMULATED
            if record.request.provider_id == "static"
            else ProvenanceKind.REAL
        )
        outputs = (
            f"status:{record.status.value}",
            f"provider:{record.request.provider_id}",
            f"graph_alias:{record.request.graph_alias}",
            f"schema_version:{record.request.schema_version}",
            f"content_version:{record.request.content_version}",
            f"query_plan_hash:{record.request.query_plan_hash}",
            f"grounding_policy:{record.request.query_plan.grounding_policy_id}",
            f"grounding_policy_hash:{record.request.grounding_policy_hash}",
            *(
                f"template:{item.template_id}@{item.template_version}"
                for item in record.request.query_plan.template_requests
            ),
            *(
                f"template_hash:{template}={digest}"
                for template, digest in sorted(record.request.template_hashes.items())
            ),
            f"bundle_id:{bundle.bundle_id if bundle else 'none'}",
            f"bundle_hash:{bundle.bundle_hash if bundle else 'none'}",
            *_read_safety_outputs(bundle),
            f"error_codes:{','.join(item.value for item in record.errors) or 'none'}",
            f"retry_of:{record.retry_of_retrieval_id or 'none'}",
        )
        event = DecisionEvent(
            event_id=event_id,
            run_id=run_id,
            cycle=cycle,
            event_type=event_type,
            actor="knowledge_provider",
            input_references=(record.retrieval_id,),
            output_references=outputs,
            rationale=(
                "Bounded knowledge retrieval "
                f"{record.status.value.casefold()} under a fixed query plan."
            ),
            timestamp=record.created_at,
            code_version=CODE_VERSION,
            provenance=retrieval_provenance,
        )
        provenance_store.append_event_idempotent(event)
        event_ids.append(event_id)
        if bundle is not None:
            validated_id = f"{event_id}-validated"
            validated = DecisionEvent(
                event_id=validated_id,
                run_id=run_id,
                cycle=cycle,
                event_type=EventType.KNOWLEDGE_BUNDLE_VALIDATED,
                actor="knowledge_bundle_validator",
                input_references=(record.retrieval_id, bundle.bundle_id),
                output_references=(
                    f"bundle_hash:{bundle.bundle_hash}",
                    f"accepted_references:{bundle.validation_result.accepted_reference_count}",
                    f"rejected_assertions:{bundle.validation_result.rejected_assertion_count}",
                    *(
                        f"trust_tier:{tier}={count}"
                        for tier, count in sorted(
                            bundle.validation_result.trust_tier_summary.items()
                        )
                    ),
                    *(
                        f"artefact:{reference}"
                        for reference in bundle.artefact_references
                    ),
                    *(
                        f"reference:{reference.reference_id}"
                        for reference in bundle.references
                    ),
                ),
                rationale="Validated identifiers, provenance, trust, and task policy.",
                timestamp=record.created_at,
                code_version=CODE_VERSION,
                provenance=retrieval_provenance,
            )
            provenance_store.append_event_idempotent(validated)
            event_ids.append(validated_id)
    return event_ids
