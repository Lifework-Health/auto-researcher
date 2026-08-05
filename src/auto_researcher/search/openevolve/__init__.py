"""Bounded, task-owned program evolution behind the generic search lifecycle."""

from auto_researcher.search.openevolve.models import (
    CandidatePreparationResult,
    CandidateStatus,
    CandidateValidationResult,
    OpenEvolveCandidate,
    OpenEvolvePopulationState,
    OpenEvolveSearchContract,
    OpenEvolveSearchResult,
    SandboxPolicy,
)

__all__ = [
    "CandidatePreparationResult",
    "CandidateStatus",
    "CandidateValidationResult",
    "OpenEvolveCandidate",
    "OpenEvolvePopulationState",
    "OpenEvolveSearchContract",
    "OpenEvolveSearchResult",
    "SandboxPolicy",
]


def __getattr__(name: str):
    if name == "OpenEvolveBackend":
        from auto_researcher.search.openevolve.backend import OpenEvolveBackend

        return OpenEvolveBackend
    raise AttributeError(name)
