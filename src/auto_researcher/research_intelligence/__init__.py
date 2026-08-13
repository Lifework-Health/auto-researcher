"""Offline-first, task-agnostic Research Intelligence contracts."""

from auto_researcher.research_intelligence.brief import DeterministicBriefBuilder
from auto_researcher.research_intelligence.models import (
    ApplicabilityAssessment,
    ApplicabilityLevel,
    Availability,
    BriefEntry,
    BriefSection,
    EvidenceCard,
    EvidenceCategory,
    EvidenceQuality,
    EvidenceQuery,
    EvidenceRanking,
    FindingCandidate,
    FindingStance,
    QuantitativeResult,
    ResearchIntelligenceBrief,
    ResearchIntelligenceRefreshRecord,
    ResearchProgrammeContext,
    RetrievedSourceMaterial,
    SourceCandidate,
    SourceRecord,
    SourceReference,
    SourceType,
    SynthesisResult,
    TrustClassification,
)
from auto_researcher.research_intelligence.protocols import EvidenceStore, ResearchScout
from auto_researcher.research_intelligence.scout import OfflineResearchScout
from auto_researcher.research_intelligence.store import SQLiteEvidenceStore
from auto_researcher.research_intelligence.synthesis import (
    DeterministicEvidenceSynthesiser,
)

__all__ = [
    "DeterministicBriefBuilder",
    "DeterministicEvidenceSynthesiser",
    "ApplicabilityAssessment",
    "ApplicabilityLevel",
    "Availability",
    "BriefEntry",
    "BriefSection",
    "EvidenceCard",
    "EvidenceCategory",
    "EvidenceQuality",
    "EvidenceQuery",
    "EvidenceRanking",
    "EvidenceStore",
    "FindingCandidate",
    "FindingStance",
    "OfflineResearchScout",
    "QuantitativeResult",
    "ResearchIntelligenceBrief",
    "ResearchIntelligenceRefreshRecord",
    "ResearchProgrammeContext",
    "ResearchScout",
    "RetrievedSourceMaterial",
    "SQLiteEvidenceStore",
    "SourceCandidate",
    "SourceRecord",
    "SourceReference",
    "SourceType",
    "SynthesisResult",
    "TrustClassification",
]
