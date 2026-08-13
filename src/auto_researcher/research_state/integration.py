"""Narrow integration helpers for PR #49 Research Intelligence views."""

from __future__ import annotations

from datetime import datetime

from auto_researcher.research_intelligence.models import ResearchIntelligenceBrief
from auto_researcher.research_intelligence.protocols import EvidenceStore
from auto_researcher.research_state.models import (
    ExternalEvidenceReference,
    external_evidence_reference,
)


def external_evidence_references_from_brief(
    programme_id: str,
    brief: ResearchIntelligenceBrief,
    evidence_store: EvidenceStore,
    *,
    evidence_store_reference: str,
    recorded_at: datetime,
) -> tuple[ExternalEvidenceReference, ...]:
    """Resolve a brief view back to its authoritative Evidence Cards."""

    sections = (
        brief.actionable_priors,
        brief.known_strong_baselines,
        brief.established_choices,
        brief.likely_failure_modes,
        brief.underexplored_hypotheses,
        brief.unresolved_uncertainties,
        brief.experiment_design_implications,
    )
    evidence_ids = sorted(
        {
            evidence_id
            for section in sections
            for entry in section
            for evidence_id in entry.evidence_card_ids
        }
    )
    references = []
    for evidence_id in evidence_ids:
        card = evidence_store.get_evidence(evidence_id)
        if card is None:
            raise ValueError("research_intelligence_brief_evidence_missing")
        references.append(
            external_evidence_reference(
                programme_id,
                card,
                evidence_store_reference=evidence_store_reference,
                recorded_at=recorded_at,
            )
        )
    return tuple(references)
