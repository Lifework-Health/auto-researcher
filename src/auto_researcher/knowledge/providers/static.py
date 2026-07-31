"""Deterministic static provider used by offline tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from auto_researcher.knowledge.identity import (
    assertion_id,
    bundle_content_hash,
    content_hash,
    entity_id,
    reference_id,
    stable_identifier,
)
from auto_researcher.knowledge.models import (
    KnowledgeAssertion,
    KnowledgeBundle,
    KnowledgeErrorCode,
    KnowledgeGraphSnapshot,
    KnowledgeProviderConfiguration,
    KnowledgeReadinessCheck,
    KnowledgeReadinessResult,
    KnowledgeReference,
    KnowledgeRetrievalRequest,
    KnowledgeSource,
    KnowledgeSourceType,
    KnowledgeTrustTier,
    KnowledgeValidationResult,
)


class StaticKnowledgeProvider:
    provider_id = "static"
    provider_version = "1.0.0"

    def __init__(
        self,
        configuration: KnowledgeProviderConfiguration,
        *,
        clock: Callable[[], datetime],
    ) -> None:
        self.configuration = configuration
        self._clock = clock
        self.calls = 0

    def readiness(
        self,
        configuration: KnowledgeProviderConfiguration,
    ) -> KnowledgeReadinessResult:
        identity_ok = configuration.provider_id == self.provider_id
        enabled = configuration.enabled
        checks = (
            KnowledgeReadinessCheck(
                code="provider_identity",
                passed=identity_ok,
                message="Static provider identity matches configuration.",
            ),
            KnowledgeReadinessCheck(
                code="provider_enabled",
                passed=enabled,
                message="Static provider is explicitly enabled.",
            ),
        )
        return KnowledgeReadinessResult(
            ready=identity_ok and enabled,
            checks=checks,
            errors=(
                ()
                if identity_ok and enabled
                else (KnowledgeErrorCode.PROVIDER_NOT_CONFIGURED,)
            ),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            schema_version=configuration.schema_version,
            content_version=configuration.content_version,
        )

    def execution_template_hashes(self) -> dict[str, str]:
        return {}

    def retrieve(self, request: KnowledgeRetrievalRequest) -> KnowledgeBundle:
        self.calls += 1
        bundle_id = stable_identifier("knowledge-bundle", request.retrieval_id)
        sources = (
            KnowledgeSource(
                source_id="source:synthetic-ontology-v1",
                source_type=KnowledgeSourceType.ONTOLOGY_RELEASE,
                title="Synthetic parameter ontology",
                version="1.0",
                curie="SYNTHSRC:ontology-v1",
                publisher_or_database="Auto Researcher synthetic fixture",
                asserted_by="curator",
                retrieval_reference=request.retrieval_id,
            ),
            KnowledgeSource(
                source_id="source:synthetic-publication-001",
                source_type=KnowledgeSourceType.LITERATURE,
                title="Synthetic source-backed objective assertion",
                version="1.0",
                accession="SYNTHPUB:001",
                publisher_or_database="Auto Researcher synthetic fixture",
                asserted_by="curator",
                retrieval_reference=request.retrieval_id,
            ),
            KnowledgeSource(
                source_id="source:synthetic-live-001",
                source_type=KnowledgeSourceType.LIVE_ASSERTION,
                title="Synthetic unverified diagnostic assertion",
                version="1.0",
                accession="SYNTHLIVE:001",
                publisher_or_database="Auto Researcher synthetic fixture",
                asserted_by="llm",
                retrieval_reference=request.retrieval_id,
            ),
        )
        entity_specs = (
            (
                "SYNTH:model-family",
                "Parameter",
                "model family",
                ("source:synthetic-ontology-v1",),
            ),
            (
                "SYNTH:complexity",
                "Parameter",
                "complexity",
                ("source:synthetic-ontology-v1",),
            ),
            (
                "SYNTH:objective-score",
                "Metric",
                "objective score",
                ("source:synthetic-publication-001",),
            ),
        )
        from auto_researcher.knowledge.models import KnowledgeEntity

        entities = tuple(
            KnowledgeEntity(
                entity_id=entity_id(curie, kind),
                curie=curie,
                entity_type=kind,
                name=name,
                source_references=source_refs,
            )
            for curie, kind, name, source_refs in entity_specs
        )
        assertion_specs = (
            (
                entities[0],
                "BOUNDS",
                entities[1],
                ("source:synthetic-ontology-v1",),
                "synthetic ontology structure",
                1.0,
                "curator",
                KnowledgeTrustTier.CURATED,
                ("complexity", "model_family"),
                "The synthetic model family bounds the permitted complexity.",
            ),
            (
                entities[1],
                "ASSOCIATED_WITH",
                entities[2],
                ("source:synthetic-publication-001",),
                "deterministic fixture study",
                0.8,
                "curator",
                KnowledgeTrustTier.CURATED,
                ("complexity",),
                "A source-backed synthetic assertion links complexity to objective score.",
            ),
            (
                entities[0],
                "ASSOCIATED_WITH",
                entities[2],
                ("source:synthetic-live-001",),
                "unverified model assertion",
                0.9,
                "llm",
                KnowledgeTrustTier.UNVERIFIED,
                ("model_family",),
                "An unverified diagnostic assertion links model family to objective score.",
            ),
        )
        assertions = []
        references = []
        for (
            subject,
            predicate,
            object_,
            source_refs,
            method,
            confidence,
            asserted_by,
            tier,
            parameters,
            claim,
        ) in assertion_specs:
            assertion_identifier = assertion_id(
                subject.curie,
                predicate,
                object_.curie,
                source_refs,
            )
            assertions.append(
                KnowledgeAssertion(
                    assertion_id=assertion_identifier,
                    subject_entity_id=subject.entity_id,
                    predicate=predicate,
                    object_entity_id=object_.entity_id,
                    direction="FORWARD",
                    source_references=source_refs,
                    method=method,
                    confidence=confidence,
                    asserted_by=asserted_by,
                    trust_tier=tier,
                )
            )
            references.append(
                KnowledgeReference(
                    reference_id=reference_id(assertion_identifier, bundle_id),
                    reference_type="SYNTHETIC_ASSERTION",
                    concise_claim=claim,
                    subject_curie=subject.curie,
                    predicate=predicate,
                    object_curie=object_.curie,
                    source_references=source_refs,
                    trust_tier=tier,
                    confidence=confidence,
                    citation_label=f"[{source_refs[0]}]",
                    bundle_id=bundle_id,
                    relevant_parameters=parameters,
                )
            )
        safe_content = {
            "sources": [item.model_dump(mode="json") for item in sources],
            "entities": [item.model_dump(mode="json") for item in entities],
            "assertions": [item.model_dump(mode="json") for item in assertions],
        }
        bundle = KnowledgeBundle(
            bundle_id=bundle_id,
            retrieval_id=request.retrieval_id,
            query_plan_hash=request.query_plan_hash,
            graph_snapshot=KnowledgeGraphSnapshot(
                provider_id=self.provider_id,
                graph_alias=request.graph_alias,
                schema_version=request.schema_version,
                configured_content_version=request.content_version,
                retrieval_timestamp=self._clock(),
                safe_graph_metadata={"fixture": "synthetic", "record_count": 3},
                returned_content_hash=content_hash(safe_content),
            ),
            sources=sources,
            entities=entities,
            assertions=tuple(assertions),
            references=tuple(references),
            validation_result=KnowledgeValidationResult(
                passed=False,
                accepted_reference_count=0,
                rejected_assertion_count=0,
                reason_codes=("not_validated",),
            ),
            bundle_hash="0" * 64,
        )
        return bundle.model_copy(update={"bundle_hash": bundle_content_hash(bundle)})

    def close(self) -> None:
        return None
