"""Deterministic evidence, provenance, identifier, and bundle validation."""

from __future__ import annotations

import re
from collections import Counter

from auto_researcher.knowledge.identity import (
    assertion_id,
    bundle_content_hash,
    entity_id,
    reference_id,
)
from auto_researcher.knowledge.models import (
    CURIE_PATTERN,
    KnowledgeAssertion,
    KnowledgeBundle,
    KnowledgeGroundingPolicy,
    KnowledgeReference,
    KnowledgeSourceType,
    KnowledgeTrustTier,
    KnowledgeValidationResult,
)
from auto_researcher.knowledge.policy import validate_policy_ceiling
from auto_researcher.knowledge.policy import prior_cap

PATIENT_LIKE = re.compile(
    r"\b(patient|participant|subject|mrn|nhs)[-_:\s]*[A-Za-z0-9]{2,}\b",
    re.IGNORECASE,
)
ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'])(?:/[A-Za-z0-9_.-]+/|[A-Za-z]:\\)")
MAXIMUM_BUNDLE_BYTES = 2_000_000


class KnowledgeBundleValidator:
    def validate(
        self,
        bundle: KnowledgeBundle,
        policy: KnowledgeGroundingPolicy,
        *,
        provider_id: str,
        schema_version: str,
        content_version: str,
        maximum_records: int,
        query_plan_hash: str,
    ) -> KnowledgeBundle:
        validate_policy_ceiling(policy)
        reasons: list[str] = []
        if bundle.graph_snapshot.provider_id != provider_id:
            reasons.append("provider_identity_mismatch")
        if bundle.graph_snapshot.schema_version != schema_version:
            reasons.append("schema_version_mismatch")
        if bundle.graph_snapshot.configured_content_version != content_version:
            reasons.append("content_version_mismatch")
        if bundle.query_plan_hash != query_plan_hash:
            reasons.append("query_plan_hash_mismatch")
        if bundle.bundle_hash != bundle_content_hash(bundle):
            reasons.append("bundle_hash_mismatch")
        if len(bundle.entities) > min(policy.maximum_entities, maximum_records):
            reasons.append("entity_limit_exceeded")
        if len(bundle.assertions) > min(policy.maximum_assertions, maximum_records):
            reasons.append("assertion_limit_exceeded")
        payload = bundle.model_dump(mode="json")
        if len(bundle.model_dump_json().encode()) > MAXIMUM_BUNDLE_BYTES:
            reasons.append("bundle_size_exceeded")
        string_values = tuple(_string_values(payload))
        if any(PATIENT_LIKE.search(value) for value in string_values):
            reasons.append("patient_like_identifier")
        if any(ABSOLUTE_PATH.search(value) for value in string_values):
            reasons.append("absolute_runtime_path")
        field_names = {item.casefold() for item in _field_names(payload)}
        if field_names & {
            "password",
            "neo4j_uri",
            "credential",
            "credentials",
            "username",
            "access_token",
            "secret",
            "uri",
        } or any(
            any(
                token in value.casefold()
                for token in (
                    "password=",
                    "neo4j_uri",
                    "access_token",
                    "secret=",
                    "neo4j+s://",
                    "bolt://",
                )
            )
            for value in string_values
        ):
            reasons.append("credential_or_infrastructure_field")
        if field_names & {
            "id",
            "_id",
            "element_id",
            "patient_id",
            "participant_id",
            "subject_id",
            "clinical_row",
            "clinical_rows",
            "mutation_value",
            "mutation_values",
            "matrix",
        }:
            reasons.append("prohibited_property")

        sources = {item.source_id: item for item in bundle.sources}
        entities = {item.entity_id: item for item in bundle.entities}
        if len(sources) != len(bundle.sources):
            reasons.append("duplicate_source_id")
        if len(entities) != len(bundle.entities):
            reasons.append("duplicate_entity_id")
        for entity in bundle.entities:
            if entity.entity_type not in policy.allowed_entity_types:
                reasons.append("disallowed_entity_type")
            if entity.entity_id != entity_id(entity.curie, entity.entity_type):
                reasons.append("unstable_entity_id")
            if not CURIE_PATTERN.fullmatch(entity.curie):
                reasons.append("invalid_curie")
            if set(entity.source_references) - set(sources):
                reasons.append("missing_entity_source")

        accepted_assertions: list[KnowledgeAssertion] = []
        rejected = 0
        for assertion in bundle.assertions:
            assertion_reasons = self._assertion_reasons(
                assertion,
                policy,
                sources,
                entities,
            )
            if assertion_reasons:
                reasons.extend(assertion_reasons)
                rejected += 1
            else:
                accepted_assertions.append(assertion)

        accepted_keys = {
            (
                item.subject_entity_id,
                item.predicate,
                item.object_entity_id,
            ): item
            for item in accepted_assertions
        }
        accepted_references: list[KnowledgeReference] = []
        for reference in bundle.references:
            subject = next(
                (
                    entity
                    for entity in bundle.entities
                    if entity.curie == reference.subject_curie
                ),
                None,
            )
            object_ = next(
                (
                    entity
                    for entity in bundle.entities
                    if entity.curie == reference.object_curie
                ),
                None,
            )
            key = (
                subject.entity_id if subject else "",
                reference.predicate,
                object_.entity_id if object_ else "",
            )
            assertion = accepted_keys.get(key)
            if (
                assertion is None
                or reference.trust_tier not in policy.allowed_trust_tiers
                or reference.confidence < policy.minimum_assertion_confidence
                or reference.reference_id
                != reference_id(assertion.assertion_id, bundle.bundle_id)
                or set(reference.source_references) - set(sources)
            ):
                continue
            accepted_references.append(
                reference.model_copy(
                    update={
                        "prior_weight_cap": prior_cap(
                            policy,
                            reference.trust_tier,
                        )
                    }
                )
            )
        accepted_references.sort(key=lambda item: item.reference_id)
        accepted_references = accepted_references[: policy.maximum_references]
        trust = Counter(item.trust_tier.value for item in accepted_references)
        fatal = {
            "provider_identity_mismatch",
            "schema_version_mismatch",
            "content_version_mismatch",
            "query_plan_hash_mismatch",
            "bundle_hash_mismatch",
            "entity_limit_exceeded",
            "assertion_limit_exceeded",
            "patient_like_identifier",
            "absolute_runtime_path",
            "credential_or_infrastructure_field",
            "prohibited_property",
            "bundle_size_exceeded",
            "duplicate_source_id",
            "duplicate_entity_id",
            "unstable_entity_id",
            "invalid_curie",
        }
        passed = not (set(reasons) & fatal)
        validation = KnowledgeValidationResult(
            passed=passed,
            accepted_reference_count=len(accepted_references),
            rejected_assertion_count=rejected,
            reason_codes=tuple(sorted(set(reasons))),
            trust_tier_summary=dict(sorted(trust.items())),
        )
        validated = bundle.model_copy(
            update={
                "assertions": tuple(accepted_assertions),
                "references": tuple(accepted_references),
                "validation_result": validation,
            }
        )
        return validated.model_copy(
            update={"bundle_hash": bundle_content_hash(validated)}
        )

    def _assertion_reasons(
        self,
        assertion: KnowledgeAssertion,
        policy: KnowledgeGroundingPolicy,
        sources: dict,
        entities: dict,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        subject = entities.get(assertion.subject_entity_id)
        object_ = entities.get(assertion.object_entity_id)
        evidence = [
            sources[item] for item in assertion.source_references if item in sources
        ]
        if subject is None or object_ is None:
            reasons.append("missing_assertion_endpoint")
        if set(assertion.source_references) - set(sources) or not evidence:
            reasons.append("missing_assertion_source")
        if assertion.predicate not in policy.allowed_predicates:
            reasons.append("disallowed_predicate")
        if assertion.asserted_by not in policy.allowed_asserted_by:
            reasons.append("disallowed_asserted_by")
        if assertion.trust_tier not in policy.allowed_trust_tiers:
            reasons.append("disallowed_trust_tier")
        if assertion.confidence < policy.minimum_assertion_confidence:
            reasons.append("low_assertion_confidence")
        if assertion.assertion_id != assertion_id(
            subject.curie if subject else "",
            assertion.predicate,
            object_.curie if object_ else "",
            assertion.source_references,
        ):
            reasons.append("unstable_assertion_id")
        if evidence and any(
            item.source_type not in policy.allowed_source_types for item in evidence
        ):
            reasons.append("disallowed_source_type")
        if evidence and any(
            item.asserted_by not in policy.allowed_asserted_by for item in evidence
        ):
            reasons.append("disallowed_source_asserted_by")
        if evidence and (
            (
                assertion.trust_tier == KnowledgeTrustTier.CURATED
                and any(
                    item.source_type
                    in {
                        KnowledgeSourceType.CORPUS_ASSERTION,
                        KnowledgeSourceType.LIVE_ASSERTION,
                    }
                    for item in evidence
                )
            )
            or (
                assertion.trust_tier == KnowledgeTrustTier.CORPUS
                and any(
                    item.source_type == KnowledgeSourceType.LIVE_ASSERTION
                    for item in evidence
                )
            )
        ):
            reasons.append("source_trust_mismatch")
        ontology = any(
            item.source_type
            in {
                KnowledgeSourceType.ONTOLOGY_RELEASE,
                KnowledgeSourceType.CURATED_DATABASE,
            }
            for item in evidence
        )
        if (
            ontology
            and policy.require_source_version_for_ontology
            and any(not item.version for item in evidence)
        ):
            reasons.append("ontology_source_version_missing")
        biological = assertion.predicate not in {
            "IS_A",
            "PART_OF",
            "INCLUDES",
            "PARTICIPATES_IN",
            "IDENTIFIES",
            "CATALOGUED_AS",
            "BOUNDS",
            "SUBTYPE_OF",
            "DEFINED_BY",
        }
        if biological and policy.require_publication_for_assertions:
            if not any(
                item.pmid
                or item.doi
                or (
                    item.accession
                    and item.source_type
                    in {
                        KnowledgeSourceType.LITERATURE,
                        KnowledgeSourceType.CURATED_ASSERTION,
                        KnowledgeSourceType.CORPUS_ASSERTION,
                    }
                )
                for item in evidence
            ):
                reasons.append("publication_or_accession_missing")
        if (
            assertion.trust_tier
            in {KnowledgeTrustTier.LIVE, KnowledgeTrustTier.UNVERIFIED}
            or assertion.asserted_by.casefold() == "llm"
        ):
            reasons.append("live_or_unverified_not_grounding")
        return tuple(reasons)


def _string_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_values(item)
    elif isinstance(value, str):
        yield value


def _field_names(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _field_names(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _field_names(item)
