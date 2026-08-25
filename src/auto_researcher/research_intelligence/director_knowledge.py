"""Frozen knowledge-card libraries for the Research Director.

The existing :class:`EvidenceCard` is the canonical knowledge-card format.  This
module creates a compact, hash-bound campaign snapshot from those cards.  The
snapshot is advisory evidence only: it cannot create a search request or an
experiment configuration.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from auto_researcher.agents.models import ResearchLandscapeEvidence
from auto_researcher.research_intelligence.models import EvidenceCard
from auto_researcher.runtime.identity import payload_hash

KNOWLEDGE_LIBRARY_SCHEMA_VERSION = "research-director-knowledge-library-v1"


class _KnowledgeLibraryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class ResearchDirectorKnowledgeCard(_KnowledgeLibraryModel):
    """Compact, cited projection of one canonical EvidenceCard."""

    evidence_id: str = Field(pattern=r"^evidence-[0-9a-f]{24}$")
    evidence_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_key: str
    claim: str
    category: str
    stance: str
    conditions: tuple[str, ...]
    limitations: tuple[str, ...]
    applicability_score: float = Field(ge=0, le=1)
    applicability_rationale: str
    evidence_quality: str
    confidence: float = Field(ge=0, le=1)
    priority_score: float = Field(ge=0, le=1)
    hypothesis_tags: tuple[str, ...] = ()
    experiment_design_implications: tuple[str, ...] = ()
    source_reference_ids: tuple[str, ...] = Field(min_length=1)
    trust_boundary: Literal["CURATED_EXTERNAL_RESEARCH_INTELLIGENCE"] = (
        "CURATED_EXTERNAL_RESEARCH_INTELLIGENCE"
    )


class ResearchDirectorKnowledgeLibrary(_KnowledgeLibraryModel):
    """Immutable collection identity bound into a campaign configuration."""

    schema_version: Literal["research-director-knowledge-library-v1"] = (
        KNOWLEDGE_LIBRARY_SCHEMA_VERSION
    )
    library_id: str = Field(pattern=r"^knowledge-library-[0-9a-f]{24}$")
    library_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    programme_id: str = Field(min_length=1)
    content_version: str = Field(min_length=1)
    evidence_cutoff: date
    cards: tuple[ResearchDirectorKnowledgeCard, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identity_is_canonical(self) -> "ResearchDirectorKnowledgeLibrary":
        if len({item.evidence_id for item in self.cards}) != len(self.cards):
            raise ValueError("research_director_knowledge_card_duplicate")
        if tuple(item.evidence_id for item in self.cards) != tuple(
            sorted(item.evidence_id for item in self.cards)
        ):
            raise ValueError("research_director_knowledge_cards_not_canonical")
        content = self.model_dump(mode="python", exclude={"library_id", "library_hash"})
        expected_hash = payload_hash(content)
        if self.library_hash != expected_hash:
            raise ValueError("research_director_knowledge_library_hash_mismatch")
        if self.library_id != f"knowledge-library-{expected_hash[:24]}":
            raise ValueError("research_director_knowledge_library_id_mismatch")
        return self


def _compact_card(card: EvidenceCard) -> ResearchDirectorKnowledgeCard:
    sources = tuple(
        sorted(
            f"{item.source_id}:{item.source_version_id}"
            for item in card.source_references
        )
    )
    return ResearchDirectorKnowledgeCard(
        evidence_id=card.evidence_id,
        evidence_content_hash=card.evidence_content_hash,
        claim_key=card.claim_key,
        claim=card.claim,
        category=card.category.value,
        stance=card.stance.value,
        conditions=card.conditions,
        limitations=card.limitations,
        applicability_score=card.current_applicability.score,
        applicability_rationale=card.current_applicability.rationale,
        evidence_quality=card.evidence_quality.value,
        confidence=card.confidence,
        priority_score=card.ranking.combined_score,
        hypothesis_tags=card.hypothesis_tags,
        experiment_design_implications=card.experiment_design_implications,
        source_reference_ids=sources,
    )


def build_research_director_knowledge_library(
    cards: tuple[EvidenceCard, ...],
    *,
    programme_id: str,
    content_version: str,
    evidence_cutoff: date,
    maximum_cards: int = 24,
) -> ResearchDirectorKnowledgeLibrary:
    """Select a deterministic, bounded set of reviewed knowledge cards."""

    if not 1 <= maximum_cards <= 64:
        raise ValueError("research_director_knowledge_card_budget_invalid")
    if not cards:
        raise ValueError("research_director_knowledge_cards_empty")
    if any(card.programme_context.task_id != programme_id for card in cards):
        raise ValueError("research_director_knowledge_programme_mismatch")

    by_id = {card.evidence_id: card for card in cards}
    if len(by_id) != len(cards):
        raise ValueError("research_director_knowledge_card_duplicate")
    ranked = sorted(
        cards,
        key=lambda item: (
            -item.ranking.combined_score,
            -item.current_applicability.score,
            item.evidence_id,
        ),
    )[:maximum_cards]
    compact = tuple(
        sorted(map(_compact_card, ranked), key=lambda item: item.evidence_id)
    )
    content = {
        "schema_version": KNOWLEDGE_LIBRARY_SCHEMA_VERSION,
        "programme_id": programme_id,
        "content_version": content_version,
        "evidence_cutoff": evidence_cutoff,
        "cards": compact,
    }
    library_hash = payload_hash(content)
    return ResearchDirectorKnowledgeLibrary(
        **content,
        library_id=f"knowledge-library-{library_hash[:24]}",
        library_hash=library_hash,
    )


def knowledge_library_as_landscape_evidence(
    library: ResearchDirectorKnowledgeLibrary,
) -> ResearchLandscapeEvidence:
    """Expose the frozen library as evidence, never executable authority."""

    return ResearchLandscapeEvidence(
        evidence_id=library.library_id,
        evidence_type="KNOWLEDGE_CARD_LIBRARY",
        evidence_hash=library.library_hash,
        source_reference=f"knowledge_cards:{library.content_version}",
        summary=(
            f"Frozen curated research library with {len(library.cards)} cited "
            "knowledge cards. Cards are advisory priors, not campaign results."
        ),
        reference_ids=tuple(item.evidence_id for item in library.cards),
        safe_payload={
            "schema_version": library.schema_version,
            "programme_id": library.programme_id,
            "content_version": library.content_version,
            "evidence_cutoff": library.evidence_cutoff.isoformat(),
            "trust_boundary": "CURATED_EXTERNAL_RESEARCH_INTELLIGENCE",
            "cards": [item.model_dump(mode="json") for item in library.cards],
            "experiment_authority_exercised": False,
        },
    )


__all__ = [
    "KNOWLEDGE_LIBRARY_SCHEMA_VERSION",
    "ResearchDirectorKnowledgeCard",
    "ResearchDirectorKnowledgeLibrary",
    "build_research_director_knowledge_library",
    "knowledge_library_as_landscape_evidence",
]
