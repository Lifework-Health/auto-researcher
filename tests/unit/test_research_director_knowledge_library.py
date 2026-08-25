from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from auto_researcher.research_intelligence import (
    DeterministicEvidenceSynthesiser,
    OfflineResearchScout,
    ResearchProgrammeContext,
    build_research_director_knowledge_library,
    knowledge_library_as_landscape_evidence,
)
from tests.research_intelligence_fixtures import feta_nnunet_corpus

NOW = datetime(2026, 8, 24, 12, tzinfo=UTC)
PROGRAMME_ID = "feta_seg_search@1.0"


def _cards():
    context = ResearchProgrammeContext(
        task_id=PROGRAMME_ID,
        domains=("fetal-mri-segmentation",),
        dataset_ids=("FeTA",),
        as_of_date=date(2026, 8, 24),
    )
    return (
        DeterministicEvidenceSynthesiser()
        .synthesise(
            OfflineResearchScout(feta_nnunet_corpus()),
            context,
            synthesised_at=NOW,
        )
        .evidence_cards
    )


def test_library_is_bounded_deterministic_and_advisory_only():
    cards = _cards()
    first = build_research_director_knowledge_library(
        cards,
        programme_id=PROGRAMME_ID,
        content_version="feta-v9-curated-v1",
        evidence_cutoff=date(2026, 8, 24),
        maximum_cards=4,
    )
    second = build_research_director_knowledge_library(
        tuple(reversed(cards)),
        programme_id=PROGRAMME_ID,
        content_version="feta-v9-curated-v1",
        evidence_cutoff=date(2026, 8, 24),
        maximum_cards=4,
    )

    assert first == second
    assert len(first.cards) == 4
    assert tuple(card.evidence_id for card in first.cards) == tuple(
        sorted(card.evidence_id for card in first.cards)
    )
    evidence = knowledge_library_as_landscape_evidence(first)
    assert evidence.evidence_type == "KNOWLEDGE_CARD_LIBRARY"
    assert evidence.evidence_hash == first.library_hash
    assert evidence.reference_ids == tuple(card.evidence_id for card in first.cards)
    assert evidence.safe_payload["experiment_authority_exercised"] is False


def test_library_rejects_wrong_programme_and_duplicate_cards():
    cards = _cards()
    with pytest.raises(ValueError, match="programme_mismatch"):
        build_research_director_knowledge_library(
            cards,
            programme_id="another_task@1.0",
            content_version="v1",
            evidence_cutoff=date(2026, 8, 24),
        )
    with pytest.raises(ValueError, match="card_duplicate"):
        build_research_director_knowledge_library(
            (cards[0], cards[0]),
            programme_id=PROGRAMME_ID,
            content_version="v1",
            evidence_cutoff=date(2026, 8, 24),
        )


def test_library_identity_detects_tampering():
    library = build_research_director_knowledge_library(
        _cards(),
        programme_id=PROGRAMME_ID,
        content_version="feta-v9-curated-v1",
        evidence_cutoff=date(2026, 8, 24),
        maximum_cards=3,
    )
    with pytest.raises(ValueError, match="library_hash_mismatch"):
        library.model_copy(update={"content_version": "tampered"}).model_validate(
            library.model_copy(update={"content_version": "tampered"}).model_dump()
        )
