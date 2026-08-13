"""Deterministic, contradiction-preserving evidence synthesis."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from auto_researcher.knowledge.identity import content_hash
from auto_researcher.research_intelligence.models import (
    ApplicabilityAssessment,
    ApplicabilityLevel,
    EvidenceCard,
    EvidenceQuality,
    EvidenceRanking,
    FindingCandidate,
    FindingStance,
    ResearchProgrammeContext,
    RetrievedSourceMaterial,
    SourceRecord,
    SourceReference,
    SynthesisResult,
    materialise_evidence_card,
    materialise_source,
    materialise_source_retrieval,
    source_identity,
)
from auto_researcher.research_intelligence.protocols import ResearchScout

QUALITY_WEIGHT = {
    EvidenceQuality.HIGH: 1.0,
    EvidenceQuality.MODERATE: 0.7,
    EvidenceQuality.LOW: 0.4,
    EvidenceQuality.UNASSESSED: 0.25,
}


def _finding_identity(finding: FindingCandidate) -> str:
    return content_hash(finding)


def _matches(
    assessment: ApplicabilityAssessment, context: ResearchProgrammeContext
) -> bool:
    return (
        (assessment.task_id is None or assessment.task_id == context.task_id)
        and (assessment.domain is None or assessment.domain in context.domains)
        and (
            assessment.dataset_id is None
            or assessment.dataset_id in context.dataset_ids
        )
    )


def _current_applicability(
    finding: FindingCandidate, context: ResearchProgrammeContext
) -> ApplicabilityAssessment:
    matching = tuple(
        assessment
        for assessment in finding.applicability_assessments
        if _matches(assessment, context)
    )
    if matching:
        return max(
            matching,
            key=lambda item: (
                item.score,
                item.task_id is not None,
                item.dataset_id is not None,
                item.domain is not None,
                item.rationale,
            ),
        )
    return ApplicabilityAssessment(
        task_id=context.task_id,
        level=ApplicabilityLevel.NOT_APPLICABLE,
        score=0,
        rationale="No supplied applicability assessment matches this programme.",
    )


def _freshness(source: SourceRecord, context: ResearchProgrammeContext) -> float:
    publication = source.publication_or_update_date
    if publication is None:
        return 0.25
    days = max(0, (context.as_of_date - publication).days)
    if days <= 730:
        return 1.0
    if days <= 1_825:
        return 0.75
    if days <= 3_650:
        return 0.5
    return 0.25


def _ranking(
    finding: FindingCandidate,
    sources: tuple[SourceRecord, ...],
    applicability: ApplicabilityAssessment,
    context: ResearchProgrammeContext,
) -> EvidenceRanking:
    source_quality = sum(item.quality_score for item in sources) / len(sources)
    quality = round((source_quality + QUALITY_WEIGHT[finding.evidence_quality]) / 2, 6)
    freshness = round(max(_freshness(item, context) for item in sources), 6)
    relevance = applicability.score
    return EvidenceRanking(
        quality_score=quality,
        relevance_score=relevance,
        freshness_score=freshness,
        combined_score=round(0.45 * quality + 0.4 * relevance + 0.15 * freshness, 6),
    )


class DeterministicEvidenceSynthesiser:
    synthesiser_id = "deterministic-evidence-synthesiser"
    synthesiser_version = "deterministic-evidence-synthesis-v1"

    def synthesise(
        self,
        scout: ResearchScout,
        programme_context: ResearchProgrammeContext,
        *,
        synthesised_at: datetime,
    ) -> SynthesisResult:
        timestamp = synthesised_at.astimezone(UTC)
        selected: dict[str, RetrievedSourceMaterial] = {}
        observed: list[RetrievedSourceMaterial] = []
        duplicate_sources = 0
        duplicate_findings = 0
        for material in scout.collect():
            observed.append(material)
            identity = source_identity(material.source)
            current = selected.get(identity)
            if current is None:
                selected[identity] = material
                continue
            duplicate_sources += 1
            if content_hash(
                current.source.model_dump(
                    mode="python",
                    exclude={"retrieved_at", "ingestion_method", "provenance_version"},
                )
            ) != content_hash(
                material.source.model_dump(
                    mode="python",
                    exclude={"retrieved_at", "ingestion_method", "provenance_version"},
                )
            ):
                raise ValueError("research_intelligence_source_version_conflict")
            findings = {
                _finding_identity(item): item
                for item in (*current.findings, *material.findings)
            }
            duplicate_findings += (
                len(current.findings) + len(material.findings) - len(findings)
            )
            source = max(
                (current.source, material.source),
                key=lambda item: item.retrieved_at,
            )
            selected[identity] = RetrievedSourceMaterial(
                source=source,
                findings=tuple(findings[key] for key in sorted(findings)),
            )

        records = {
            identity: materialise_source(material.source)
            for identity, material in selected.items()
        }
        retrievals = {
            retrieval.retrieval_id: retrieval
            for material in observed
            for retrieval in (
                materialise_source_retrieval(
                    material.source, records[source_identity(material.source)]
                ),
            )
        }
        grouped: dict[str, list[tuple[SourceRecord, FindingCandidate]]] = defaultdict(
            list
        )
        seen: set[tuple[str, str]] = set()
        for identity, material in selected.items():
            source = records[identity]
            for finding in material.findings:
                finding_identity = _finding_identity(finding)
                occurrence = (source.source_version_id, finding_identity)
                if occurrence in seen:
                    duplicate_findings += 1
                    continue
                seen.add(occurrence)
                grouped[finding_identity].append((source, finding))

        cards: list[EvidenceCard] = []
        for finding_identity in sorted(grouped):
            items = grouped[finding_identity]
            finding = items[0][1]
            sources = tuple(
                sorted(
                    {item[0].source_version_id: item[0] for item in items}.values(),
                    key=lambda item: item.source_version_id,
                )
            )
            applicability = _current_applicability(finding, programme_context)
            cards.append(
                materialise_evidence_card(
                    source_references=tuple(
                        SourceReference(
                            source_id=source.source_id,
                            source_version_id=source.source_version_id,
                        )
                        for source in sources
                    ),
                    claim_key=finding.claim_key,
                    claim=finding.claim,
                    category=finding.category,
                    stance=finding.stance,
                    quantitative_results=finding.quantitative_results,
                    method_or_intervention=finding.method_or_intervention,
                    dataset_or_population=finding.dataset_or_population,
                    conditions=finding.conditions,
                    limitations=finding.limitations,
                    applicability_assessments=finding.applicability_assessments,
                    current_applicability=applicability,
                    programme_context=programme_context,
                    evidence_quality=finding.evidence_quality,
                    confidence=finding.confidence,
                    ranking=_ranking(
                        finding, sources, applicability, programme_context
                    ),
                    hypothesis_tags=finding.hypothesis_tags,
                    experiment_design_implications=(
                        finding.experiment_design_implications
                    ),
                    supporting_evidence_ids=(),
                    conflicting_evidence_ids=(),
                    synthesised_at=timestamp,
                )
            )

        linked: list[EvidenceCard] = []
        for card in cards:
            support = tuple(
                sorted(
                    other.evidence_id
                    for other in cards
                    if other.evidence_id != card.evidence_id
                    and other.claim_key == card.claim_key
                    and other.stance == card.stance
                )
            )
            conflict = tuple(
                sorted(
                    other.evidence_id
                    for other in cards
                    if other.claim_key == card.claim_key
                    and other.stance != card.stance
                    and FindingStance.MIXED not in {other.stance, card.stance}
                )
            )
            linked.append(
                EvidenceCard.model_validate(
                    {
                        **card.model_dump(mode="python"),
                        "supporting_evidence_ids": support,
                        "conflicting_evidence_ids": conflict,
                    }
                )
            )

        return SynthesisResult(
            programme_context=programme_context,
            source_records=tuple(
                sorted(records.values(), key=lambda item: item.source_version_id)
            ),
            source_retrievals=tuple(
                sorted(retrievals.values(), key=lambda item: item.retrieval_id)
            ),
            evidence_cards=tuple(
                sorted(
                    linked,
                    key=lambda item: (-item.ranking.combined_score, item.evidence_id),
                )
            ),
            duplicate_sources_ignored=duplicate_sources,
            duplicate_findings_ignored=duplicate_findings,
            synthesised_at=timestamp,
        )
