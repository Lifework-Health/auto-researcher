from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ValidationError

from auto_researcher.contracts.enums import ProvenanceKind
from auto_researcher.contracts.models import EvaluationResult
from auto_researcher.tasks.artifacts import strict_json_bytes
from auto_researcher.tasks.scientific_json import (
    ScientificJsonPolicy,
    normalise_scientific_json,
)


def test_finite_python_float_is_preserved():
    result = normalise_scientific_json(0.75, root_path="score")
    assert result.valid
    assert result.value == 0.75
    assert result.category_counts["finite_numbers"] == 1


def test_finite_numpy_scalar_is_preserved():
    numpy = pytest.importorskip("numpy")
    result = normalise_scientific_json(numpy.float64(0.25), root_path="metric")
    assert result.valid
    assert result.value == pytest.approx(0.25)


def test_permitted_nan_is_null_and_availability_is_recorded():
    result = normalise_scientific_json(
        {"c_index": float("nan"), "zero": 0.0},
        policy=ScientificJsonPolicy(
            permitted_nan_paths=frozenset({"scientific.c_index"})
        ),
        root_path="scientific",
    )
    assert result.valid
    assert result.value == {"c_index": None, "zero": 0.0}
    assert result.value["c_index"] is None
    assert result.value["zero"] == 0.0
    assert result.unavailable_paths == ("scientific.c_index",)
    assert result.category_counts["unavailable_nan"] == 1


@pytest.mark.parametrize(
    ("value", "category"),
    [
        (float("nan"), "rejected_nan"),
        (float("inf"), "rejected_positive_infinity"),
        (float("-inf"), "rejected_negative_infinity"),
    ],
)
def test_unknown_non_finite_values_are_rejected(value, category):
    result = normalise_scientific_json(value, root_path="unexpected")
    assert not result.valid
    assert result.rejected_paths == ("unexpected",)
    assert result.category_counts[category] == 1


def test_nested_numpy_array_is_normalised_recursively():
    numpy = pytest.importorskip("numpy")
    result = normalise_scientific_json(
        {"matrix": numpy.array([[1.0, 2.0], [3.0, 4.0]])}
    )
    assert result.valid
    assert result.value == {"matrix": [[1.0, 2.0], [3.0, 4.0]]}


def test_pydantic_models_and_dataclasses_are_supported():
    class Payload(BaseModel):
        value: float

    @dataclass
    class Wrapper:
        payload: Payload

    result = normalise_scientific_json(Wrapper(Payload(value=0.5)))
    assert result.valid
    assert result.value == {"payload": {"value": 0.5}}


def test_strict_json_never_emits_non_standard_numeric_tokens():
    normalised = normalise_scientific_json(
        {"metric": float("nan")},
        policy=ScientificJsonPolicy(permitted_nan_paths=frozenset({"metric"})),
    )
    payload = strict_json_bytes(normalised.value).decode("utf-8")
    assert '"metric": null' in payload
    assert "NaN" not in payload
    assert "Infinity" not in payload
    json.loads(payload, parse_constant=lambda token: pytest.fail(token))


def test_successful_evaluation_rejects_a_non_finite_primary_score():
    with pytest.raises(ValidationError, match="primary_score must be finite"):
        EvaluationResult(
            experiment_id="experiment",
            success=True,
            primary_score=float("nan"),
            metrics={"objective": 0.5},
            constraint_results={"gate": True},
            evaluator_version="v1",
            provenance=ProvenanceKind.REAL,
        )


def test_constraints_require_actual_booleans():
    with pytest.raises(ValidationError, match="explicit booleans"):
        EvaluationResult(
            experiment_id="experiment",
            success=True,
            primary_score=0.5,
            metrics={"objective": 0.5},
            constraint_results={"gate": 1},
            evaluator_version="v1",
            provenance=ProvenanceKind.REAL,
        )


def test_evaluation_metrics_are_strict_json_before_graph_state():
    with pytest.raises(ValidationError, match="strict finite JSON"):
        EvaluationResult(
            experiment_id="experiment",
            success=True,
            primary_score=0.5,
            metrics={"unexpected": float("nan")},
            constraint_results={"gate": True},
            evaluator_version="v1",
            provenance=ProvenanceKind.REAL,
        )
