"""Convert generic parameter contracts to Optuna distributions lazily."""

from __future__ import annotations

from typing import Any

from auto_researcher.search.optuna.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    IntParameterSpec,
    ParameterSpec,
)


def to_optuna_distribution(parameter: ParameterSpec) -> Any:
    try:
        from optuna.distributions import (
            CategoricalDistribution,
            FloatDistribution,
            IntDistribution,
        )
    except ImportError as exc:
        raise RuntimeError(
            "OPTUNA search requires the HPO dependency. "
            "Install with `pip install -e '.[hpo]'`."
        ) from exc

    if isinstance(parameter, FloatParameterSpec):
        return FloatDistribution(
            low=parameter.low,
            high=parameter.high,
            log=parameter.log,
            step=parameter.step,
        )
    if isinstance(parameter, IntParameterSpec):
        return IntDistribution(
            low=parameter.low,
            high=parameter.high,
            log=parameter.log,
            step=parameter.step,
        )
    if isinstance(parameter, CategoricalParameterSpec):
        return CategoricalDistribution(choices=parameter.choices)
    raise TypeError(f"unsupported parameter specification: {type(parameter).__name__}")


def fixed_distributions(parameters: tuple[ParameterSpec, ...]) -> dict[str, Any]:
    return {
        parameter.name: to_optuna_distribution(parameter)
        for parameter in parameters
    }
