"""Immutable contracts for native Optuna studies and operational diagnostics."""

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


class OptunaConditionSpec(OptunaModel):
    """A typed conditional branch over an earlier task-approved parameter."""

    parameter: str = Field(min_length=1)
    equals: JsonValue


class FloatParameterSpec(OptunaModel):
    type: Literal["float"] = "float"
    name: str = Field(min_length=1)
    low: float
    high: float
    log: bool = False
    step: float | None = None
    condition: OptunaConditionSpec | None = None

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
    condition: OptunaConditionSpec | None = None

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
    condition: OptunaConditionSpec | None = None

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
                raise ValueError(
                    "categorical choices must be JSON serialisable"
                ) from exc
        if len(set(encoded)) != len(encoded):
            raise ValueError("categorical choices must not contain duplicates")
        return choices


ParameterSpec = Annotated[
    FloatParameterSpec | IntParameterSpec | CategoricalParameterSpec,
    Field(discriminator="type"),
]


class OptunaObjectiveSpec(OptunaModel):
    name: str = Field(min_length=1)
    direction: OptimisationDirection
    metric: str = Field(min_length=1)


class OptunaConstraintSpec(OptunaModel):
    """Task-approved projection into Optuna's <=0 feasible convention."""

    name: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    relation: Literal["LESS_THAN_OR_EQUAL", "GREATER_THAN_OR_EQUAL"]
    threshold: float

    @field_validator("threshold")
    @classmethod
    def threshold_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("constraint thresholds must be finite")
        return value


class OptunaSamplerSpec(OptunaModel):
    type: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_-]*$")
    options: FrozenJsonDict = Field(default_factory=dict)


class OptunaPrunerSpec(OptunaModel):
    type: str = Field(default="none", min_length=1, pattern=r"^[a-z][a-z0-9_-]*$")
    options: FrozenJsonDict = Field(default_factory=dict)


class OptunaDiagnosticsSpec(OptunaModel):
    parameter_importance: bool = False
    importance_evaluators: tuple[
        Literal["native_default", "fanova", "mdi", "ped_anova"], ...
    ] = ("native_default",)


class OptunaStudyDiagnostics(OptunaModel):
    """Search diagnostics only; never scientific evidence or Research State."""

    sampler: str = Field(min_length=1)
    pruner: str = Field(min_length=1)
    completed_trials: int = Field(ge=0)
    pruned_trials: int = Field(ge=0)
    failed_trials: int = Field(ge=0)
    best_trial_number: int | None = Field(default=None, ge=0)
    pareto_trial_numbers: tuple[int, ...] = ()
    parameter_importances: FrozenJsonDict = Field(default_factory=dict)
    epistemic_status: Literal["OPERATIONAL_SEARCH_DIAGNOSTIC"] = (
        "OPERATIONAL_SEARCH_DIAGNOSTIC"
    )


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
    sampler: Literal["TPE"] | OptunaSamplerSpec = "TPE"
    n_startup_trials: int = Field(default=5, ge=0)
    objective_metric: str = Field(min_length=1)
    objectives: tuple[OptunaObjectiveSpec, ...] = ()
    constraints: tuple[OptunaConstraintSpec, ...] = ()
    pruner: OptunaPrunerSpec = Field(default_factory=OptunaPrunerSpec)
    intermediate_reporting: bool = False
    diagnostics: OptunaDiagnosticsSpec = Field(default_factory=OptunaDiagnosticsSpec)
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
        positions = {name: index for index, name in enumerate(names)}
        for index, parameter in enumerate(self.parameters):
            condition = parameter.condition
            if condition is None:
                continue
            condition_index = positions.get(condition.parameter)
            if condition_index is None or condition_index >= index:
                raise ValueError(
                    "conditional parameters must reference an earlier approved parameter"
                )
        objective_specs = self.objective_specs
        objective_names = [objective.name for objective in objective_specs]
        if len(set(objective_names)) != len(objective_names):
            raise ValueError("Optuna objective names must be unique")
        if self.objectives and (
            self.objective_metric != self.objectives[0].metric
            or self.direction is not self.objectives[0].direction
        ):
            raise ValueError(
                "legacy objective fields must match the first versioned objective"
            )
        constraint_names = [constraint.name for constraint in self.constraints]
        if len(set(constraint_names)) != len(constraint_names):
            raise ValueError("Optuna constraint names must be unique")
        if self.pruner.type != "none" and not self.intermediate_reporting:
            raise ValueError("native pruning requires task intermediate reporting")
        if len(objective_specs) > 1 and self.pruner.type != "none":
            raise ValueError("optuna_4_9_multi_objective_pruning_not_supported")
        return self

    @property
    def objective_specs(self) -> tuple[OptunaObjectiveSpec, ...]:
        if self.objectives:
            return self.objectives
        return (
            OptunaObjectiveSpec(
                name=self.objective_metric,
                direction=self.direction,
                metric=self.objective_metric,
            ),
        )

    @property
    def sampler_spec(self) -> OptunaSamplerSpec:
        if self.sampler == "TPE":
            return OptunaSamplerSpec(
                type="tpe",
                options={"n_startup_trials": self.n_startup_trials},
            )
        return self.sampler

    @property
    def is_v1(self) -> bool:
        return self.schema_version.startswith("1.")


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
    objective_names: tuple[str, ...] = ()
    directions: tuple[OptimisationDirection, ...] = ()
    trial_budget: int = Field(ge=0)
    trials_asked: int = Field(default=0, ge=0)
    trials_completed: int = Field(default=0, ge=0)
    trials_failed: int = Field(default=0, ge=0)
    trials_pruned: int = Field(default=0, ge=0)
    current_trial: OptunaTrialReference | None = None
    best_feasible_trial_number: int | None = Field(default=None, ge=0)
    best_feasible_score: float | None = None
    best_overall_trial_number: int | None = Field(default=None, ge=0)
    best_overall_score: float | None = None
    pareto_trial_numbers: tuple[int, ...] = ()
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
    objective_names: tuple[str, ...] = ()
    directions: tuple[OptimisationDirection, ...] = ()
    trial_budget: int = Field(ge=0)
    trials_asked: int = Field(ge=0)
    trials_completed: int = Field(ge=0)
    trials_failed: int = Field(ge=0)
    trials_pruned: int = Field(default=0, ge=0)
    best_feasible_trial_number: int | None = Field(default=None, ge=0)
    best_feasible_score: float | None = None
    best_overall_trial_number: int | None = Field(default=None, ge=0)
    best_overall_score: float | None = None
    feasible_trial_found: bool
    pareto_trial_numbers: tuple[int, ...] = ()
    pareto_trials: tuple["OptunaTrialOutcome", ...] = ()
    diagnostics: OptunaStudyDiagnostics | None = None
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
    objective_values: tuple[float, ...] = ()
    objective_names: tuple[str, ...] = ()
    feasible: bool
    constraint_values: tuple[float, ...] = ()
    experiment_id: str
    parameters: FrozenJsonDict
    evaluation_artefact_references: tuple[str, ...] = ()
    verification_status: str
    pruned_at_step: int | None = Field(default=None, ge=0)
    intermediate_values: dict[int, float] = Field(default_factory=dict)
    evaluation_reused: bool = False

    @field_validator("objective_value")
    @classmethod
    def objective_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("trial objective must be finite")
        return value

    @field_validator("objective_values", "constraint_values")
    @classmethod
    def vectors_must_be_finite(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("trial objective and constraint vectors must be finite")
        return values

    @model_validator(mode="after")
    def scalar_convenience_is_single_objective_only(self) -> "OptunaTrialOutcome":
        if len(self.objective_values) == 1:
            if self.objective_value is None:
                object.__setattr__(self, "objective_value", self.objective_values[0])
            elif self.objective_value != self.objective_values[0]:
                raise ValueError("scalar objective does not match objective vector")
        elif len(self.objective_values) > 1 and self.objective_value is not None:
            raise ValueError("multi-objective trials do not have a scalar objective")
        return self
