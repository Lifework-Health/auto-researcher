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
    "EmbeddedOpenEvolveSearch",
    "NativeEvolutionConfiguration",
    "NativeEvolutionLimits",
    "OpenEvolveCandidate",
    "OpenEvolvePopulationState",
    "OpenEvolveSearchContract",
    "OpenEvolveSearchResult",
    "SandboxPolicy",
    "TaskOwnedCandidateNormalizer",
    "TaskOwnedScientificEvaluator",
    "native_configuration_from_search_space",
    "native_limits_from_search_space",
]


def __getattr__(name: str):
    if name == "OpenEvolveBackend":
        from auto_researcher.search.openevolve.backend import OpenEvolveBackend

        return OpenEvolveBackend
    native_exports = {
        "EmbeddedOpenEvolveSearch",
        "NativeEvolutionConfiguration",
        "NativeEvolutionLimits",
        "TaskOwnedCandidateNormalizer",
        "TaskOwnedScientificEvaluator",
        "native_configuration_from_search_space",
        "native_limits_from_search_space",
    }
    if name in native_exports:
        from auto_researcher.search.openevolve import native_engine

        return getattr(native_engine, name)
    raise AttributeError(name)
