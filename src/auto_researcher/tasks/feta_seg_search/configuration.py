"""Bounded fold-0 configuration for FeTA SegResNet development search."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

CONFIGURATION_SCHEMA_VERSION = "feta-segresnet-search-configuration-v1"
FIDELITY_LEVELS = (25, 50, 100, 150, 300)
LEARNING_RATE_BOUNDS = (3e-5, 5e-4)
WEIGHT_DECAY_BOUNDS = (1e-6, 3e-4)
DROPOUT_BOUNDS = (0.0, 0.4)
DICE_WEIGHT_BOUNDS = (0.5, 1.5)
POSITIVE_NEGATIVE_RATIOS = ("1:1", "2:1", "3:1")
AUGMENTATION_STRENGTHS = ("light", "baseline", "strong")


def validation_epochs(maximum_epochs: int) -> tuple[int, ...]:
    """Return the registered sparse native-space validation schedule."""

    schedules = {
        25: (25,),
        50: (25, 50),
        100: (50, 100),
        150: (50, 100, 150),
        300: tuple(range(25, 301, 25)),
    }
    try:
        return schedules[maximum_epochs]
    except KeyError as exc:
        raise ValueError("feta_search_fidelity_invalid") from exc


class FeTASegSearchConfiguration(BaseModel):
    """Immutable candidate plus the fixed audited scientific context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fold: Literal[0] = 0
    maximum_epochs: Literal[25, 50, 100, 150, 300] = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.2
    dice_weight: float = 1.0
    positive_negative_ratio: Literal["1:1", "2:1", "3:1"] = "1:1"
    augmentation_strength: Literal["light", "baseline", "strong"] = "baseline"

    spatial_dims: Literal[3] = 3
    in_channels: Literal[1] = 1
    out_channels: Literal[8] = 8
    init_filters: Literal[32] = 32
    blocks_down: tuple[int, int, int, int] = (1, 2, 2, 4)
    blocks_up: tuple[int, int, int] = (1, 1, 1)
    norm: Literal["GROUP"] = "GROUP"
    activation: Literal["RELU"] = "RELU"
    upsample_mode: Literal["deconv"] = "deconv"
    spacing_mm: tuple[float, float, float] = (0.5, 0.5, 0.5)
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: Literal[1] = 1
    samples_per_volume: Literal[2] = 2
    ce_weight: Literal[1.0] = 1.0
    inference_overlap: Literal[0.5] = 0.5
    inference_blending: Literal["gaussian"] = "gaussian"
    sliding_window_batch_size: Literal[1] = 1
    seed: Literal[20260807] = 20260807

    @field_validator("learning_rate")
    @classmethod
    def learning_rate_is_bounded(cls, value: float) -> float:
        return cls._bounded(value, LEARNING_RATE_BOUNDS, "learning_rate")

    @field_validator("weight_decay")
    @classmethod
    def weight_decay_is_bounded(cls, value: float) -> float:
        return cls._bounded(value, WEIGHT_DECAY_BOUNDS, "weight_decay")

    @field_validator("dropout")
    @classmethod
    def dropout_is_bounded(cls, value: float) -> float:
        return cls._bounded(value, DROPOUT_BOUNDS, "dropout")

    @field_validator("dice_weight")
    @classmethod
    def dice_weight_is_bounded(cls, value: float) -> float:
        return cls._bounded(value, DICE_WEIGHT_BOUNDS, "dice_weight")

    @staticmethod
    def _bounded(
        value: float, bounds: tuple[float, float], name: str
    ) -> float:
        result = float(value)
        if not math.isfinite(result) or not bounds[0] <= result <= bounds[1]:
            raise ValueError(f"feta_search_{name}_out_of_bounds")
        return result

    @model_validator(mode="after")
    def fixed_scientific_context_is_locked(self) -> "FeTASegSearchConfiguration":
        if (
            self.blocks_down != (1, 2, 2, 4)
            or self.blocks_up != (1, 1, 1)
            or self.spacing_mm != (0.5, 0.5, 0.5)
            or self.patch_size != (128, 128, 128)
        ):
            raise ValueError("feta_search_fixed_scientific_context_modified")
        return self

    def validation_epochs(self) -> tuple[int, ...]:
        return validation_epochs(self.maximum_epochs)

    def scientific_configuration(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def baseline_search_configuration(maximum_epochs: int = 50) -> dict[str, Any]:
    """Candidate matching the locked baseline hyperparameters at one fidelity."""

    return FeTASegSearchConfiguration(
        maximum_epochs=maximum_epochs,  # type: ignore[arg-type]
    ).model_dump(mode="json")


def normalise_search_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    return FeTASegSearchConfiguration.model_validate(configuration).model_dump(
        mode="json"
    )
