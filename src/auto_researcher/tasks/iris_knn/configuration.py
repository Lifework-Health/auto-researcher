"""Canonical bounded configuration for the Iris weighted k-NN benchmark."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

FEATURE_WEIGHT_LOW = 0.1
FEATURE_WEIGHT_HIGH = 4.0
K_CHOICES = (1, 3, 5, 7, 9)
DISTANCE_POWER_CHOICES = (1, 2)
OPTUNA_WEIGHT_NAMES = tuple(f"feature_weight_{index}" for index in range(4))


class IrisKNNConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_weight_0: float
    feature_weight_1: float
    feature_weight_2: float
    feature_weight_3: float
    k: Literal[1, 3, 5, 7, 9]
    distance_power: Literal[1, 2]

    @model_validator(mode="before")
    @classmethod
    def accept_user_facing_weight_vector(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "feature_weights" not in value:
            return value
        payload = dict(value)
        weights = payload.pop("feature_weights")
        if not isinstance(weights, (list, tuple)) or len(weights) != 4:
            raise ValueError("feature_weights must contain exactly four values")
        if any(name in payload for name in OPTUNA_WEIGHT_NAMES):
            raise ValueError("feature weights must be supplied in exactly one form")
        payload.update(zip(OPTUNA_WEIGHT_NAMES, weights, strict=True))
        return payload

    @field_validator(*OPTUNA_WEIGHT_NAMES)
    @classmethod
    def weights_are_finite_and_bounded(cls, value: float) -> float:
        canonical = float(value)
        if not (
            math.isfinite(canonical)
            and FEATURE_WEIGHT_LOW <= canonical <= FEATURE_WEIGHT_HIGH
        ):
            raise ValueError("feature weights must be finite values in [0.1, 4.0]")
        return canonical

    @property
    def feature_weights(self) -> tuple[float, float, float, float]:
        return (
            self.feature_weight_0,
            self.feature_weight_1,
            self.feature_weight_2,
            self.feature_weight_3,
        )

    def scientific_configuration(self) -> dict:
        return {
            "feature_weights": list(self.feature_weights),
            "k": self.k,
            "distance_power": self.distance_power,
        }


def normalise_iris_configuration(configuration: dict[str, Any]) -> dict:
    """Accept canonical input or Optuna's four scalar weight parameters."""

    payload = dict(configuration)
    present = [name for name in OPTUNA_WEIGHT_NAMES if name in payload]
    if present and len(present) != len(OPTUNA_WEIGHT_NAMES):
        raise ValueError("all four Optuna feature weights are required exactly once")
    return IrisKNNConfiguration.model_validate(payload).model_dump(mode="json")


def baseline_configuration() -> dict:
    return {
        "feature_weights": [1.0, 1.0, 1.0, 1.0],
        "k": 3,
        "distance_power": 2,
    }


def configuration_schema() -> dict:
    return {
        "feature_weights": {
            "type": "array",
            "length": 4,
            "items": {
                "type": "number",
                "minimum": FEATURE_WEIGHT_LOW,
                "maximum": FEATURE_WEIGHT_HIGH,
            },
        },
        "k": {"type": "integer", "enum": list(K_CHOICES)},
        "distance_power": {
            "type": "integer",
            "enum": list(DISTANCE_POWER_CHOICES),
        },
    }
