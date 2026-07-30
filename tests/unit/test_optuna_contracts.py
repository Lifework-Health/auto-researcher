from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from auto_researcher.search.optuna.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    IntParameterSpec,
    OptimisationDirection,
    OptunaStudySpec,
)
from auto_researcher.search.optuna.narrowing import narrow_study_spec
from auto_researcher.search.optuna.selection import (
    SelectionCandidate,
    select_trials,
)


def study_spec() -> OptunaStudySpec:
    return OptunaStudySpec(
        schema_version="1.0",
        task_id="generic",
        task_version="1",
        search_space_version="1",
        direction=OptimisationDirection.MAXIMIZE,
        parameters=(
            FloatParameterSpec(name="rate", low=0.01, high=1.0, log=True),
            IntParameterSpec(name="depth", low=2, high=10, step=2),
            CategoricalParameterSpec(name="family", choices=("a", "b", "c")),
        ),
        fixed_configuration={"context": "fixed"},
        trial_budget=10,
        seed=7,
        objective_metric="score",
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FloatParameterSpec(name="x", low=0.1, high=1.0),
        lambda: IntParameterSpec(name="x", low=1, high=3),
        lambda: CategoricalParameterSpec(name="x", choices=("a", "b")),
    ],
)
def test_valid_parameter_models(factory):
    assert factory().name == "x"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FloatParameterSpec(name="x", low=1.0, high=1.0),
        lambda: FloatParameterSpec(name="x", low=0.0, high=1.0, log=True),
        lambda: IntParameterSpec(name="x", low=1, high=3, step=0),
        lambda: CategoricalParameterSpec(name="x", choices=("a", "a")),
    ],
)
def test_invalid_parameter_models(factory):
    with pytest.raises(ValidationError):
        factory()


def test_study_invariants_and_json_round_trip():
    original = study_spec()
    assert OptunaStudySpec.model_validate_json(original.model_dump_json()) == original
    with pytest.raises(ValidationError, match="unique"):
        OptunaStudySpec.model_validate(
            {
                **json.loads(original.model_dump_json()),
                "parameters": [
                    {"type": "int", "name": "x", "low": 1, "high": 2},
                    {"type": "int", "name": "x", "low": 1, "high": 2},
                ],
            }
        )
    with pytest.raises(ValidationError, match="overlap"):
        OptunaStudySpec.model_validate(
            {
                **json.loads(original.model_dump_json()),
                "fixed_configuration": {"rate": 0.2},
            }
        )


def test_structural_narrowing_accepts_subsets_and_pinning():
    narrowed = narrow_study_spec(
        study_spec(),
        {
            "trial_budget": 4,
            "fixed": {"family": "b"},
            "parameters": {
                "rate": {"type": "float", "low": 0.1, "high": 0.8, "log": True},
                "depth": {"type": "int", "low": 4, "high": 8, "step": 2},
            },
        },
        request_experiment_budget=5,
    )
    assert narrowed.trial_budget == 4
    assert narrowed.fixed_configuration["family"] == "b"
    assert [parameter.name for parameter in narrowed.parameters] == ["rate", "depth"]


@pytest.mark.parametrize(
    ("proposal", "message"),
    [
        ({"parameters": {"rate": {"low": 0.001}}}, "widen"),
        ({"parameters": {"depth": {"high": 12}}}, "widen"),
        ({"parameters": {"family": {"choices": ["a", "new"]}}}, "new choice"),
        ({"parameters": {"family": {"choices": []}}}, "empty"),
        ({"parameters": {"rate": {"log": False}}}, "log mutation"),
        ({"parameters": {"depth": {"step": 1}}}, "step mutation"),
        ({"parameters": {"unknown": {"type": "int"}}}, "unregistered"),
        ({"parameters": {"rate": "bad"}}, "mapping"),
    ],
)
def test_structural_narrowing_rejects_invalid_plans(proposal, message):
    with pytest.raises(ValueError, match=message):
        narrow_study_spec(
            study_spec(),
            proposal,
            request_experiment_budget=10,
        )


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (OptimisationDirection.MAXIMIZE, 2),
        (OptimisationDirection.MINIMIZE, 1),
    ],
)
def test_selection_prefers_best_feasible_and_preserves_overall(direction, expected):
    selected = select_trials(
        [
            SelectionCandidate(0, 99.0, False),
            SelectionCandidate(1, 1.0, True),
            SelectionCandidate(2, 2.0, True),
        ],
        direction,
    )
    assert selected.best_feasible.trial_number == expected
    assert selected.best_overall is not None


def test_selection_ties_use_lowest_trial_and_no_feasible_is_clean():
    selected = select_trials(
        [
            SelectionCandidate(4, 2.0, False),
            SelectionCandidate(2, 2.0, False),
        ],
        OptimisationDirection.MAXIMIZE,
    )
    assert selected.best_feasible is None
    assert selected.best_overall.trial_number == 2
