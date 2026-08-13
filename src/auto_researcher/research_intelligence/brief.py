"""Deterministic, traceable views over stored external evidence."""

from __future__ import annotations

from datetime import UTC, datetime

from auto_researcher.research_intelligence.models import (
    BriefEntry,
    BriefSection,
    EvidenceCard,
    EvidenceCategory,
    EvidenceQuery,
    ResearchIntelligenceBrief,
    ResearchProgrammeContext,
)
from auto_researcher.research_intelligence.protocols import EvidenceStore

_SECTIONS = {
    EvidenceCategory.ACTIONABLE_PRIOR: (
        "actionable_priors",
        BriefSection.ACTIONABLE_PRIORS,
    ),
    EvidenceCategory.STRONG_BASELINE: (
        "known_strong_baselines",
        BriefSection.STRONG_BASELINES,
    ),
    EvidenceCategory.ESTABLISHED_CHOICE: (
        "established_choices",
        BriefSection.ESTABLISHED_CHOICES,
    ),
    EvidenceCategory.FAILURE_MODE: (
        "likely_failure_modes",
        BriefSection.LIKELY_FAILURE_MODES,
    ),
    EvidenceCategory.UNDEREXPLORED_HYPOTHESIS: (
        "underexplored_hypotheses",
        BriefSection.UNDEREXPLORED_HYPOTHESES,
    ),
}


def _entry(card: EvidenceCard, section: BriefSection) -> BriefEntry:
    return BriefEntry(
        section=section,
        statement=card.claim,
        evidence_card_ids=(card.evidence_id,),
        priority_score=card.ranking.combined_score,
        experiment_design_implications=card.experiment_design_implications,
    )


class DeterministicBriefBuilder:
    """Builds an advisory view without creating new scientific claims."""

    def build(
        self,
        store: EvidenceStore,
        programme_context: ResearchProgrammeContext,
        *,
        generated_at: datetime,
    ) -> ResearchIntelligenceBrief:
        cards = store.query_evidence(EvidenceQuery(task_id=programme_context.task_id))
        cards = tuple(
            card for card in cards if card.programme_context == programme_context
        )
        values: dict[str, list[BriefEntry]] = {
            name: [] for name, _ in _SECTIONS.values()
        }
        for card in cards:
            target = _SECTIONS.get(card.category)
            if target is not None:
                values[target[0]].append(_entry(card, target[1]))

        conflicts: dict[str, set[str]] = {}
        by_id = {card.evidence_id: card for card in cards}
        for card in cards:
            if card.conflicting_evidence_ids:
                conflicts.setdefault(card.claim_key, set()).update(
                    (card.evidence_id, *card.conflicting_evidence_ids)
                )
        unresolved = []
        for claim_key, identifiers in sorted(conflicts.items()):
            relevant = tuple(
                by_id[item] for item in sorted(identifiers) if item in by_id
            )
            if relevant:
                unresolved.append(
                    BriefEntry(
                        section=BriefSection.UNRESOLVED_CONFLICTS,
                        statement=f"Conflicting external evidence remains for {claim_key}.",
                        evidence_card_ids=tuple(card.evidence_id for card in relevant),
                        priority_score=max(
                            card.ranking.combined_score for card in relevant
                        ),
                    )
                )

        implications = []
        for card in cards:
            for implication in card.experiment_design_implications:
                implications.append(
                    BriefEntry(
                        section=BriefSection.EXPERIMENT_DESIGN_IMPLICATIONS,
                        statement=implication,
                        evidence_card_ids=(card.evidence_id,),
                        priority_score=card.ranking.combined_score,
                    )
                )

        return ResearchIntelligenceBrief(
            programme_context=programme_context,
            **{name: tuple(entries) for name, entries in values.items()},
            unresolved_uncertainties=tuple(unresolved),
            experiment_design_implications=tuple(implications),
            generated_at=generated_at.astimezone(UTC),
        )
