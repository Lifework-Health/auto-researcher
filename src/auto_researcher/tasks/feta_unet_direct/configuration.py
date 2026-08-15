"""Frozen configuration for the FeTA BasicUNet DIRECT baseline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class FeTAUNetDirectConfiguration(BaseModel):
    """One of two immutable execution profiles for the frozen architecture."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: Literal["engineering_smoke", "frozen_baseline"] = "frozen_baseline"
    spatial_dims: Literal[3] = 3
    in_channels: Literal[1] = 1
    out_channels: Literal[8] = 8
    features: tuple[int, int, int, int, int, int] = (32, 32, 64, 128, 256, 32)
    activation: Literal["LeakyReLU"] = "LeakyReLU"
    negative_slope: float = 0.1
    activation_inplace: Literal[True] = True
    norm: Literal["instance"] = "instance"
    norm_affine: Literal[True] = True
    dropout: float = 0.0
    upsample: Literal["deconv"] = "deconv"
    spacing_mm: tuple[float, float, float] = (0.5, 0.5, 0.5)
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: Literal[1] = 1
    samples_per_volume: Literal[2] = 2
    positive_negative_ratio: Literal["1:1"] = "1:1"
    learning_rate: float = 0.0001
    weight_decay: float = 0.00001
    maximum_epochs: int = 300
    validation_every: int = 5
    inference_overlap: float = 0.5
    inference_blending: Literal["gaussian"] = "gaussian"
    sliding_window_batch_size: Literal[1] = 1
    seed: Literal[20260807] = 20260807
    fold_count: int = 5
    smoke_fold: Literal[0] = 0
    smoke_training_subjects: int = 1
    smoke_validation_subjects: int = 1

    @model_validator(mode="after")
    def frozen_profile(self) -> "FeTAUNetDirectConfiguration":
        if self.features != (32, 32, 64, 128, 256, 32):
            raise ValueError("feta_unet_architecture_is_locked")
        if self.negative_slope != 0.1 or self.dropout != 0.0:
            raise ValueError("feta_unet_architecture_is_locked")
        if self.learning_rate != 0.0001 or self.weight_decay != 0.00001:
            raise ValueError("feta_unet_training_configuration_is_locked")
        if self.inference_overlap != 0.5:
            raise ValueError("feta_unet_inference_is_locked")
        if self.spacing_mm != (0.5, 0.5, 0.5) or self.patch_size != (
            128,
            128,
            128,
        ):
            raise ValueError("feta_unet_preprocessing_is_locked")
        if self.profile == "frozen_baseline" and (
            self.maximum_epochs != 300
            or self.validation_every != 5
            or self.fold_count != 5
            or self.smoke_training_subjects != 1
            or self.smoke_validation_subjects != 1
        ):
            raise ValueError("feta_unet_baseline_profile_is_locked")
        if self.profile == "engineering_smoke" and (
            self.maximum_epochs != 1
            or self.validation_every != 1
            or self.fold_count != 1
            or self.smoke_training_subjects != 1
            or self.smoke_validation_subjects != 1
        ):
            raise ValueError("feta_unet_smoke_profile_is_locked")
        return self

    def scientific_configuration(self) -> dict:
        return self.model_dump(mode="json")


def baseline_configuration() -> dict:
    return {
        "profile": "frozen_baseline",
        "maximum_epochs": 300,
        "validation_every": 5,
        "fold_count": 5,
    }


def engineering_smoke_configuration() -> dict:
    return {
        "profile": "engineering_smoke",
        "maximum_epochs": 1,
        "validation_every": 1,
        "fold_count": 1,
    }
