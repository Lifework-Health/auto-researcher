"""Bounded host-interpreted training policy for FeTA OpenEvolve candidates."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from auto_researcher.tasks.feta_seg_search.configuration import (
    DICE_WEIGHT_BOUNDS,
    LEARNING_RATE_BOUNDS,
)

TRAINING_POLICY_VERSION = "feta-training-policy-v1"
AUGMENTATION_RECIPE_VERSION = "feta-evolve-bounded-numeric-v1"


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def reject_non_finite_floats(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("feta_evolve_policy_non_finite")
        return value


class LearningRatePolicy(PolicyModel):
    family: Literal["constant", "cosine", "linear"] = "constant"
    warmup_fraction: float = Field(default=0.0, ge=0.0, le=0.2)
    end_multiplier: float = Field(default=1.0, ge=0.05, le=1.0)

    @model_validator(mode="after")
    def constant_policy_is_constant(self) -> "LearningRatePolicy":
        if self.family == "constant" and (
            self.warmup_fraction != 0.0 or self.end_multiplier != 1.0
        ):
            raise ValueError("feta_evolve_constant_lr_policy_invalid")
        return self


class DiceWeightPolicy(PolicyModel):
    family: Literal["constant", "linear"] = "constant"
    start: float = Field(
        default=1.0, ge=DICE_WEIGHT_BOUNDS[0], le=DICE_WEIGHT_BOUNDS[1]
    )
    end: float = Field(default=1.0, ge=DICE_WEIGHT_BOUNDS[0], le=DICE_WEIGHT_BOUNDS[1])

    @model_validator(mode="after")
    def constant_policy_is_constant(self) -> "DiceWeightPolicy":
        if self.family == "constant" and self.start != self.end:
            raise ValueError("feta_evolve_constant_dice_policy_invalid")
        return self


class AugmentationRecipe(PolicyModel):
    flip_probability: float = Field(default=0.2, ge=0.1, le=0.3)
    intensity_probability: float = Field(default=0.2, ge=0.1, le=0.3)
    scale_factor: float = Field(default=0.1, ge=0.05, le=0.15)
    shift_offset: float = Field(default=0.1, ge=0.05, le=0.15)


class TrainingPolicy(PolicyModel):
    policy_version: Literal["feta-training-policy-v1"] = TRAINING_POLICY_VERSION
    learning_rate: LearningRatePolicy = Field(default_factory=LearningRatePolicy)
    dice_weight: DiceWeightPolicy = Field(default_factory=DiceWeightPolicy)
    augmentation: AugmentationRecipe = Field(default_factory=AugmentationRecipe)
    positive_negative_ratio: Literal["1:1", "2:1", "3:1"] = "1:1"

    def learning_rate_at(
        self, epoch: int, maximum_epochs: int, base_learning_rate: float
    ) -> float:
        if not 1 <= epoch <= maximum_epochs:
            raise ValueError("feta_evolve_epoch_out_of_range")
        progress = 1.0 if maximum_epochs == 1 else (epoch - 1) / (maximum_epochs - 1)
        policy = self.learning_rate
        if policy.family == "constant":
            multiplier = 1.0
        elif policy.warmup_fraction > 0 and progress < policy.warmup_fraction:
            multiplier = max(0.05, progress / policy.warmup_fraction)
        else:
            decay_progress = (
                progress
                if policy.warmup_fraction == 0
                else (progress - policy.warmup_fraction)
                / (1.0 - policy.warmup_fraction)
            )
            decay_progress = min(1.0, max(0.0, decay_progress))
            if policy.family == "linear":
                multiplier = 1.0 - (1.0 - policy.end_multiplier) * decay_progress
            else:
                cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                multiplier = (
                    policy.end_multiplier + (1.0 - policy.end_multiplier) * cosine
                )
        return min(
            LEARNING_RATE_BOUNDS[1],
            max(LEARNING_RATE_BOUNDS[0], base_learning_rate * multiplier),
        )

    def dice_weight_at(self, epoch: int, maximum_epochs: int) -> float:
        if not 1 <= epoch <= maximum_epochs:
            raise ValueError("feta_evolve_epoch_out_of_range")
        policy = self.dice_weight
        if policy.family == "constant" or maximum_epochs == 1:
            return policy.start
        progress = (epoch - 1) / (maximum_epochs - 1)
        return policy.start + (policy.end - policy.start) * progress


def default_training_policy(
    *,
    dice_weight: float = 1.0,
    augmentation_strength: str = "baseline",
    ratio: str = "1:1",
) -> TrainingPolicy:
    recipes = {
        "light": AugmentationRecipe(
            flip_probability=0.1,
            intensity_probability=0.1,
            scale_factor=0.05,
            shift_offset=0.05,
        ),
        "baseline": AugmentationRecipe(),
        "strong": AugmentationRecipe(
            flip_probability=0.3,
            intensity_probability=0.3,
            scale_factor=0.15,
            shift_offset=0.15,
        ),
    }
    try:
        recipe = recipes[augmentation_strength]
    except KeyError as exc:
        raise ValueError("feta_evolve_base_augmentation_invalid") from exc
    return TrainingPolicy(
        dice_weight=DiceWeightPolicy(start=dice_weight, end=dice_weight),
        augmentation=recipe,
        positive_negative_ratio=ratio,  # type: ignore[arg-type]
    )
