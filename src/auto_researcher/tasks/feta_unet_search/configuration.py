"""Bounded BasicUNet training configuration for planner-driven development."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

CONFIGURATION_SCHEMA_VERSION = "feta-basic-unet-search-configuration-v1"
FIDELITY_LEVELS = (5, 25, 50, 100, 150)
LEARNING_RATE_BOUNDS = (3e-5, 5e-4)
WEIGHT_DECAY_BOUNDS = (1e-6, 3e-4)
DROPOUT_BOUNDS = (0.0, 0.3)
DICE_WEIGHT_BOUNDS = (0.5, 1.5)
POSITIVE_NEGATIVE_RATIOS = ("1:1", "2:1", "3:1")
AUGMENTATION_STRENGTHS = ("light", "baseline", "strong")
CANDIDATE_CONFIGURATION_FIELDS = (
    "maximum_epochs",
    "learning_rate",
    "weight_decay",
    "dropout",
    "dice_weight",
    "positive_negative_ratio",
    "augmentation_strength",
)


class FeTAUNetSearchConfiguration(BaseModel):
    """A fold-0 BasicUNet candidate with a small registered mutable surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: Literal["development_baseline"] = "development_baseline"
    spatial_dims: Literal[3] = 3
    in_channels: Literal[1] = 1
    out_channels: Literal[8] = 8
    features: tuple[int, int, int, int, int, int] = (32, 32, 64, 128, 256, 32)
    activation: Literal["LeakyReLU"] = "LeakyReLU"
    negative_slope: float = 0.1
    activation_inplace: Literal[True] = True
    norm: Literal["instance"] = "instance"
    norm_affine: Literal[True] = True
    upsample: Literal["deconv"] = "deconv"
    spacing_mm: tuple[float, float, float] = (0.5, 0.5, 0.5)
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: Literal[1] = 1
    samples_per_volume: Literal[2] = 2
    maximum_epochs: Literal[5, 25, 50, 100, 150] = 25
    validation_every: Literal[5] = 5
    fold_count: Literal[1] = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.0
    dice_weight: float = 1.0
    positive_negative_ratio: Literal["1:1", "2:1", "3:1"] = "1:1"
    augmentation_strength: Literal["light", "baseline", "strong"] = "baseline"
    inference_overlap: float = 0.5
    inference_blending: Literal["gaussian"] = "gaussian"
    sliding_window_batch_size: Literal[1] = 1
    seed: Literal[20260807] = 20260807
    progress_milestone_epochs: tuple[
        Literal[25], Literal[50], Literal[100], Literal[150]
    ] = (
        25,
        50,
        100,
        150,
    )
    smoke_fold: Literal[0] = 0
    smoke_training_subjects: Literal[1] = 1
    smoke_validation_subjects: Literal[1] = 1

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
    def _bounded(value: float, bounds: tuple[float, float], name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or not bounds[0] <= result <= bounds[1]:
            raise ValueError(f"feta_unet_search_{name}_out_of_bounds")
        return result

    @model_validator(mode="after")
    def bounded_search_profile(self) -> "FeTAUNetSearchConfiguration":
        # Deliberately do not call the frozen DIRECT validator: this sibling
        # task varies only the registered training surface while retaining its
        # architecture, preprocessing, fold and inference identities.
        if (
            self.features != (32, 32, 64, 128, 256, 32)
            or self.negative_slope != 0.1
            or self.spacing_mm != (0.5, 0.5, 0.5)
            or self.patch_size != (128, 128, 128)
            or self.inference_overlap != 0.5
            or self.smoke_fold != 0
        ):
            raise ValueError("feta_unet_search_fixed_context_modified")
        return self

    def scientific_configuration(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def baseline_search_configuration(maximum_epochs: int = 25) -> dict[str, Any]:
    return FeTAUNetSearchConfiguration(
        maximum_epochs=maximum_epochs  # type: ignore[arg-type]
    ).model_dump(mode="json")


def normalise_search_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    validated = FeTAUNetSearchConfiguration.model_validate(configuration).model_dump(
        mode="json"
    )
    return {name: validated[name] for name in CANDIDATE_CONFIGURATION_FIELDS}
