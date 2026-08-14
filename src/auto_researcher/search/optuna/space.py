"""Task-approved dynamic search spaces using public Trial.suggest_* APIs."""

from __future__ import annotations

from typing import Any

from auto_researcher.search.optuna.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    IntParameterSpec,
    OptunaStudySpec,
)


def uses_trial_suggestions(spec: OptunaStudySpec) -> bool:
    return not spec.is_v1 or any(
        parameter.condition is not None for parameter in spec.parameters
    )


def suggest_parameters(trial: Any, spec: OptunaStudySpec) -> dict[str, Any]:
    """Let Optuna choose values only inside the immutable task envelope."""

    suggested: dict[str, Any] = {}
    for parameter in spec.parameters:
        condition = parameter.condition
        if condition is not None:
            if condition.parameter not in suggested:
                raise RuntimeError("optuna_condition_predecessor_not_suggested")
            if suggested[condition.parameter] != condition.equals:
                continue
        if isinstance(parameter, FloatParameterSpec):
            value = trial.suggest_float(
                parameter.name,
                parameter.low,
                parameter.high,
                step=parameter.step,
                log=parameter.log,
            )
        elif isinstance(parameter, IntParameterSpec):
            value = trial.suggest_int(
                parameter.name,
                parameter.low,
                parameter.high,
                step=parameter.step,
                log=parameter.log,
            )
        elif isinstance(parameter, CategoricalParameterSpec):
            value = trial.suggest_categorical(parameter.name, parameter.choices)
        else:  # pragma: no cover - the discriminated model excludes this
            raise TypeError("unsupported Optuna parameter specification")
        suggested[parameter.name] = value
    return suggested
