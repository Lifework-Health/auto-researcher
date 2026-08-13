from __future__ import annotations

from datetime import UTC, date, datetime

from auto_researcher.research_intelligence.models import (
    ApplicabilityLevel,
    ResearchProgrammeContext,
)
from auto_researcher.research_intelligence.scout import OfflineResearchScout
from auto_researcher.research_intelligence.synthesis import (
    DeterministicEvidenceSynthesiser,
)
from tests.research_intelligence_fixtures import feta_nnunet_corpus, tabular_corpus

NOW = datetime(2026, 8, 13, 14, tzinfo=UTC)
FETA_CONTEXT = ResearchProgrammeContext(
    task_id="feta_seg_search@1.0",
    domains=("fetal-mri-segmentation",),
    dataset_ids=("FeTA",),
    hypothesis_ids=("fetal-mri-segmentation.baseline",),
    as_of_date=date(2026, 8, 13),
)


def _synthesise(materials, context=FETA_CONTEXT):
    return DeterministicEvidenceSynthesiser().synthesise(
        OfflineResearchScout(materials), context, synthesised_at=NOW
    )


def test_synthesis_is_deterministic_under_insertion_order_and_preserves_conflicts():
    corpus = feta_nnunet_corpus()
    forward = _synthesise(corpus)
    reverse = _synthesise(tuple(reversed(corpus)))
    assert forward == reverse
    spacing = [
        card
        for card in forward.evidence_cards
        if card.claim_key == "spacing.fixed_target"
    ]
    assert len(spacing) == 2
    assert all(card.conflicting_evidence_ids for card in spacing)
    assert {card.conflicting_evidence_ids[0] for card in spacing} == {
        card.evidence_id for card in spacing
    }


def test_duplicate_sources_and_findings_are_idempotently_ignored():
    item = feta_nnunet_corpus()[0]
    duplicate_finding = item.model_copy(
        update={"findings": (item.findings[0], item.findings[0])}
    )
    result = _synthesise((duplicate_finding, duplicate_finding))
    assert len(result.source_records) == 1
    assert len(result.evidence_cards) == 1
    assert result.duplicate_sources_ignored == 1
    assert result.duplicate_findings_ignored == 3


def test_same_source_with_disjoint_findings_merges_deterministically():
    item = feta_nnunet_corpus()[0]
    left = item.model_copy(update={"findings": (item.findings[0],)})
    right = item.model_copy(update={"findings": (item.findings[1],)})
    forward = _synthesise((left, right))
    reverse = _synthesise((right, left))
    assert forward == reverse
    assert len(forward.source_records) == 1
    assert len(forward.evidence_cards) == 2
    assert all(card.source_references for card in forward.evidence_cards)


def test_high_quality_external_evidence_can_have_low_task_applicability():
    context = ResearchProgrammeContext(
        task_id="unrelated_task@1.0",
        domains=("unrelated-domain",),
        as_of_date=date(2026, 8, 13),
    )
    result = _synthesise((feta_nnunet_corpus()[-1],), context)
    card = result.evidence_cards[0]
    assert card.ranking.quality_score > 0.9
    assert card.current_applicability.level == ApplicabilityLevel.NOT_APPLICABLE
    assert card.ranking.relevance_score == 0


def test_second_non_imaging_domain_uses_same_task_agnostic_pipeline():
    context = ResearchProgrammeContext(
        task_id="credit_risk_fixture@1.0",
        domains=("imbalanced-tabular-classification",),
        as_of_date=date(2026, 8, 13),
    )
    result = _synthesise(tabular_corpus(), context)
    assert len(result.evidence_cards) == 2
    assert {card.programme_context.task_id for card in result.evidence_cards} == {
        "credit_risk_fixture@1.0"
    }
