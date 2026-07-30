"""Optional generic Optuna ask/tell search package."""

from auto_researcher.search.optuna.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    IntParameterSpec,
    OptimisationDirection,
    OptunaStudyResult,
    OptunaStudySpec,
    OptunaStudyState,
    OptunaTrialReference,
    OptunaTrialOutcome,
    OptunaTrialStatus,
)

__all__ = [
    "CategoricalParameterSpec",
    "FloatParameterSpec",
    "IntParameterSpec",
    "OptimisationDirection",
    "OptunaStudyResult",
    "OptunaStudySpec",
    "OptunaStudyState",
    "OptunaTrialReference",
    "OptunaTrialOutcome",
    "OptunaTrialStatus",
]
