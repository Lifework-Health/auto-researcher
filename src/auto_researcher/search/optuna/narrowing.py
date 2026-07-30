"""Structural narrowing of authoritative task-owned parameter spaces."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import JsonValue

from auto_researcher.search.optuna.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    IntParameterSpec,
    OptunaStudySpec,
    ParameterSpec,
)


def _encoded(value: JsonValue) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_fixed_value(parameter: ParameterSpec, value: JsonValue) -> None:
    if isinstance(parameter, FloatParameterSpec):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"fixed {parameter.name!r} must be numeric")
        numeric = float(value)
        if not parameter.low <= numeric <= parameter.high:
            raise ValueError(f"fixed {parameter.name!r} is outside registered bounds")
        return
    if isinstance(parameter, IntParameterSpec):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"fixed {parameter.name!r} must be an integer")
        if not parameter.low <= value <= parameter.high:
            raise ValueError(f"fixed {parameter.name!r} is outside registered bounds")
        if (value - parameter.low) % parameter.step:
            raise ValueError(f"fixed {parameter.name!r} does not respect registered step")
        return
    allowed = {_encoded(choice) for choice in parameter.choices}
    if _encoded(value) not in allowed:
        raise ValueError(f"fixed {parameter.name!r} is not a registered choice")


def _narrow_numeric(
    parameter: FloatParameterSpec | IntParameterSpec,
    proposal: dict[str, Any],
) -> FloatParameterSpec | IntParameterSpec:
    allowed_keys = {"type", "low", "high", "log", "step"}
    unknown = set(proposal) - allowed_keys
    if unknown:
        raise ValueError(
            f"unsupported narrowing fields for {parameter.name!r}: "
            f"{', '.join(sorted(unknown))}"
        )
    if proposal.get("type", parameter.type) != parameter.type:
        raise ValueError(f"parameter type mutation rejected for {parameter.name!r}")
    if "log" in proposal and bool(proposal["log"]) != parameter.log:
        raise ValueError(f"log mutation rejected for {parameter.name!r}")
    if "step" in proposal and proposal["step"] != parameter.step:
        raise ValueError(f"step mutation rejected for {parameter.name!r}")
    low = proposal.get("low", parameter.low)
    high = proposal.get("high", parameter.high)
    if isinstance(low, bool) or isinstance(high, bool):
        raise ValueError(f"bounds for {parameter.name!r} must be numeric")
    if isinstance(parameter, IntParameterSpec) and (
        not isinstance(low, int) or not isinstance(high, int)
    ):
        raise ValueError(f"bounds for {parameter.name!r} must be integers")
    if isinstance(parameter, FloatParameterSpec) and (
        not isinstance(low, (int, float))
        or not isinstance(high, (int, float))
        or not math.isfinite(float(low))
        or not math.isfinite(float(high))
    ):
        raise ValueError(f"bounds for {parameter.name!r} must be finite numbers")
    if low < parameter.low or high > parameter.high:
        raise ValueError(f"narrowing may not widen {parameter.name!r}")
    if low > high or (
        isinstance(parameter, FloatParameterSpec) and low == high
    ):
        raise ValueError(f"narrowing creates an empty range for {parameter.name!r}")
    if isinstance(parameter, IntParameterSpec) and (
        (low - parameter.low) % parameter.step
        or (high - parameter.low) % parameter.step
    ):
        raise ValueError(f"narrowed bounds for {parameter.name!r} violate its step")
    if isinstance(parameter, FloatParameterSpec) and parameter.step is not None:
        low_steps = (float(low) - parameter.low) / parameter.step
        high_steps = (float(high) - parameter.low) / parameter.step
        if not math.isclose(low_steps, round(low_steps), abs_tol=1e-9) or not (
            math.isclose(high_steps, round(high_steps), abs_tol=1e-9)
        ):
            raise ValueError(
                f"narrowed bounds for {parameter.name!r} violate its step"
            )
    payload = parameter.model_dump()
    payload.update(low=low, high=high)
    return type(parameter).model_validate(payload)


def _narrow_categorical(
    parameter: CategoricalParameterSpec,
    proposal: dict[str, Any],
) -> CategoricalParameterSpec:
    allowed_keys = {"type", "choices"}
    unknown = set(proposal) - allowed_keys
    if unknown:
        raise ValueError(
            f"unsupported narrowing fields for {parameter.name!r}: "
            f"{', '.join(sorted(unknown))}"
        )
    if proposal.get("type", parameter.type) != parameter.type:
        raise ValueError(f"parameter type mutation rejected for {parameter.name!r}")
    requested = tuple(proposal.get("choices", parameter.choices))
    if not requested:
        raise ValueError(f"categorical narrowing for {parameter.name!r} is empty")
    registered = {_encoded(choice) for choice in parameter.choices}
    if any(_encoded(choice) not in registered for choice in requested):
        raise ValueError(
            f"categorical narrowing introduces a new choice for {parameter.name!r}"
        )
    expected_order = tuple(
        choice
        for choice in parameter.choices
        if _encoded(choice) in {_encoded(item) for item in requested}
    )
    if requested != expected_order:
        raise ValueError(
            f"categorical narrowing may not reorder choices for {parameter.name!r}"
        )
    if len(requested) < 2:
        raise ValueError(
            f"use fixed configuration to select one choice for {parameter.name!r}"
        )
    payload = parameter.model_dump()
    payload["choices"] = requested
    return CategoricalParameterSpec.model_validate(payload)


def narrow_study_spec(
    registered: OptunaStudySpec,
    proposal: dict[str, JsonValue],
    *,
    request_experiment_budget: int,
) -> OptunaStudySpec:
    allowed_top_level = {
        "trial_budget",
        "seed",
        "fixed",
        "parameters",
        "n_startup_trials",
    }
    unknown = set(proposal) - allowed_top_level
    if unknown:
        raise ValueError(
            "unsupported Optuna search fields: "
            f"{', '.join(sorted(unknown))}"
        )
    raw_budget = proposal.get("trial_budget", request_experiment_budget)
    if isinstance(raw_budget, bool) or not isinstance(raw_budget, int):
        raise ValueError("Optuna trial_budget must be an integer")
    trial_budget = raw_budget
    if trial_budget <= 0:
        raise ValueError("Optuna trial_budget must be positive")
    if trial_budget > request_experiment_budget:
        raise ValueError("Optuna trial_budget exceeds SearchRequest experiment_budget")
    seed = proposal.get("seed", registered.seed)
    n_startup_trials = proposal.get(
        "n_startup_trials", registered.n_startup_trials
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Optuna seed must be an integer")
    if isinstance(n_startup_trials, bool) or not isinstance(n_startup_trials, int):
        raise ValueError("n_startup_trials must be an integer")
    if n_startup_trials < 0:
        raise ValueError("n_startup_trials must be non-negative")

    fixed_proposal = proposal.get("fixed", {})
    parameter_proposal = proposal.get("parameters", {})
    if not isinstance(fixed_proposal, dict) or not isinstance(parameter_proposal, dict):
        raise ValueError("Optuna fixed and parameters sections must be mappings")

    registered_parameters = {
        parameter.name: parameter for parameter in registered.parameters
    }
    known_names = set(registered_parameters) | set(registered.fixed_configuration)
    unknown_fixed = set(fixed_proposal) - known_names
    unknown_parameters = set(parameter_proposal) - set(registered_parameters)
    if unknown_fixed:
        raise ValueError(
            f"unknown fixed parameters: {', '.join(sorted(unknown_fixed))}"
        )
    if unknown_parameters:
        raise ValueError(
            f"planner attempted unregistered parameters: "
            f"{', '.join(sorted(unknown_parameters))}"
        )

    fixed = dict(registered.fixed_configuration)
    remaining: list[ParameterSpec] = []
    for name, parameter in registered_parameters.items():
        if name in fixed_proposal:
            value = fixed_proposal[name]
            _validate_fixed_value(parameter, value)
            fixed[name] = value
            if name in parameter_proposal:
                raise ValueError(f"{name!r} cannot be both fixed and sampled")
            continue
        raw = parameter_proposal.get(name, {})
        if not isinstance(raw, dict):
            raise ValueError(f"narrowing for {name!r} must be a mapping")
        if isinstance(parameter, (FloatParameterSpec, IntParameterSpec)):
            remaining.append(_narrow_numeric(parameter, raw))
        else:
            remaining.append(_narrow_categorical(parameter, raw))

    for name, registered_value in registered.fixed_configuration.items():
        if name in fixed_proposal and fixed_proposal[name] != registered_value:
            raise ValueError(
                f"fixed scientific context {name!r} does not match the task"
            )
    if not remaining:
        raise ValueError("Optuna study requires at least one sampled parameter")

    payload = registered.model_dump(mode="python")
    payload.update(
        parameters=tuple(remaining),
        fixed_configuration=fixed,
        trial_budget=trial_budget,
        seed=seed,
        n_startup_trials=n_startup_trials,
    )
    return OptunaStudySpec.model_validate(payload)
