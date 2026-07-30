"""Immutable contracts for generic single-objective Optuna search."""

from __future__ import annotations

import json
import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from auto_researcher.contracts.models import FrozenJsonDict


class OptimisationDirection(StrEnum):
    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"


class OptunaTrialStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
    PRUNED = "PRUNED"


class OptunaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class FloatParameterSpec(OptunaModel):
    type: Literal["float"] = "float"
    name: str = Field(min_length=1)
    low: float
    high: float
    log: bool = False
    step: float | None = None

    @model_validator(mode="after")
    def validate_distribution(self) -> "FloatParameterSpec":
        values = [self.low, self.high]
        if self.step is not None:
            values.append(self.step)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("float distribution values must be finite")
        if self.low >= self.high:
            raise ValueError("float low must be less than high")
        if self.log and self.low <= 0:
            raise ValueError("logarithmic float bounds must be positive")
        if self.step is not None and self.step <= 0:
            raise ValueError("float step must be positive")
        if self.log and self.step is not None:
            raise ValueError("Optuna does not support float log with step")
        if self.step is not None:
            intervals = (self.high - self.low) / self.step
            if not math.isclose(intervals, round(intervals), abs_tol=1e-9):
                raise ValueError("float bounds must align with step")
        return self


class IntParameterSpec(OptunaModel):
    type: Literal["int"] = "int"
    name: str = Field(min_length=1)
    low: int
    high: int
    step: int = 1
    log: bool = False

    @model_validator(mode="after")
    def validate_distribution(self) -> "IntParameterSpec":
        if self.low > self.high:
            raise ValueError("integer low must be less than or equal to high")
        if self.step <= 0:
            raise ValueError("integer step must be positive")
        if self.log and (self.low <= 0 or self.high <= 0):
            raise ValueError("logarithmic integer bounds must be positive")
        if self.log and self.step != 1:
            raise ValueError("Optuna logarithmic integer distributions require step=1")
        if (self.high - self.low) % self.step:
            raise ValueError("integer bounds must align with step")
        return self


class CategoricalParameterSpec(OptunaModel):
    type: Literal["categorical"] = "categorical"
    name: str = Field(min_length=1)
    choices: tuple[JsonValue, ...] = Field(min_length=2)

    @field_validator("choices")
    @classmethod
    def choices_must_be_unique_and_json_safe(
        cls,
        choices: tuple[JsonValue, ...],
    ) -> tuple[JsonValue, ...]:
        encoded: list[str] = []
        for choice in choices:
            try:
                encoded.append(
                    json.dumps(
                        choice,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("categorical choices must be JSON serialisable") from exc
        if len(set(encoded)) != len(encoded):
            raise ValueError("categorical choices must not contain duplicates")
        return choices


ParameterSpec = Annotated[
    FloatParameterSpec | IntParameterSpec | CategoricalParameterSpec,
    Field(discriminator="type"),
]


class OptunaStudySpec(OptunaModel):
    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    search_space_version: str = Field(min_length=1)
    direction: OptimisationDirection
    parameters: tuple[ParameterSpec, ...] = Field(min_length=1)
    fixed_configuration: FrozenJsonDict = Field(default_factory=dict)
    trial_budget: int = Field(gt=0)
    seed: int
    sampler: Literal["TPE"] = "TPE"
    n_startup_trials: int = Field(default=5, ge=0)
    objective_metric: str = Field(min_length=1)
    study_metadata: FrozenJsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def names_must_be_disjoint_and_unique(self) -> "OptunaStudySpec":
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("Optuna parameter names must be unique")
        overlap = set(names) & self.fixed_configuration.keys()
        if overlap:
            raise ValueError(
                "sampled and fixed configuration names overlap: "
                f"{', '.join(sorted(overlap))}"
            )
        return self


class OptunaTrialReference(OptunaModel):
    study_name: str = Field(min_length=1)
    trial_number: int = Field(ge=0)
    slot_index: int = Field(ge=0)
    parameters: FrozenJsonDict
    experiment_id: str | None = None
    status: OptunaTrialStatus


class OptunaStudyState(OptunaModel):
    study_name: str = Field(min_length=1)
    search_space_hash: str = Field(min_length=1)
    direction: OptimisationDirection
    trial_budget: int = Field(ge=0)
    trials_asked: int = Field(default=0, ge=0)
    trials_completed: int = Field(default=0, ge=0)
    trials_failed: int = Field(default=0, ge=0)
    current_trial: OptunaTrialReference | None = None
    best_feasible_trial_number: int | None = Field(default=None, ge=0)
    best_feasible_score: float | None = None
    best_overall_trial_number: int | None = Field(default=None, ge=0)
    best_overall_score: float | None = None
    finished: bool = False
    finish_reason: str | None = None

    @field_validator("best_feasible_score", "best_overall_score")
    @classmethod
    def best_scores_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("best study scores must be finite")
        return value


class OptunaStudyResult(OptunaModel):
    study_name: str = Field(min_length=1)
    direction: OptimisationDirection
    trial_budget: int = Field(ge=0)
    trials_asked: int = Field(ge=0)
    trials_completed: int = Field(ge=0)
    trials_failed: int = Field(ge=0)
    best_feasible_trial_number: int | None = Field(default=None, ge=0)
    best_feasible_score: float | None = None
    best_overall_trial_number: int | None = Field(default=None, ge=0)
    best_overall_score: float | None = None
    feasible_trial_found: bool
    finish_reason: str = Field(min_length=1)
    artefact_references: tuple[str, ...] = ()

    @field_validator("best_feasible_score", "best_overall_score")
    @classmethod
    def best_scores_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("best result scores must be finite")
        return value


class OptunaTrialOutcome(OptunaModel):
    trial_number: int = Field(ge=0)
    status: OptunaTrialStatus
    objective_value: float | None = None
    feasible: bool
    experiment_id: str
    parameters: FrozenJsonDict
    evaluation_artefact_references: tuple[str, ...] = ()
    verification_status: str

    @field_validator("objective_value")
    @classmethod
    def objective_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("trial objective must be finite")
        return value
