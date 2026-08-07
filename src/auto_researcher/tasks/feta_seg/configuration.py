"""Locked FeTA SegResNet baseline configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class FeTASegConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["full", "smoke"] = "full"
    spatial_dims: Literal[3] = 3
    in_channels: Literal[1] = 1
    out_channels: Literal[8] = 8
    init_filters: Literal[32] = 32
    blocks_down: tuple[int, int, int, int] = (1, 2, 2, 4)
    blocks_up: tuple[int, int, int] = (1, 1, 1)
    norm: Literal["GROUP"] = "GROUP"
    activation: Literal["RELU"] = "RELU"
    upsample_mode: Literal["deconv"] = "deconv"
    dropout: float = 0.2
    spacing_mm: tuple[float, float, float] = (0.5, 0.5, 0.5)
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: Literal[1] = 1
    samples_per_volume: Literal[2] = 2
    positive_negative_ratio: Literal["1:1"] = "1:1"
    learning_rate: float = 0.0001
    weight_decay: float = 0.00001
    maximum_epochs: int = 300
    validation_every: Literal[5] = 5
    inference_overlap: float = 0.5
    inference_blending: Literal["gaussian"] = "gaussian"
    sliding_window_batch_size: Literal[1] = 1
    seed: Literal[20260807] = 20260807
    fold_count: int = 5

    @model_validator(mode="after")
    def mode_budget(self) -> "FeTASegConfiguration":
        if (
            self.dropout != 0.2
            or self.learning_rate != 0.0001
            or self.weight_decay != 0.00001
            or self.inference_overlap != 0.5
        ):
            raise ValueError("feta_scientific_configuration_is_locked")
        if self.mode == "full" and (self.maximum_epochs != 300 or self.fold_count != 5):
            raise ValueError("feta_full_baseline_budget_is_locked")
        if self.mode == "smoke" and not (
            1 <= self.maximum_epochs <= 2 and self.fold_count == 1
        ):
            raise ValueError("feta_smoke_budget_invalid")
        return self

    def scientific_configuration(self) -> dict:
        return self.model_dump(mode="json")


def baseline_configuration() -> dict:
    # Vector-valued architecture constants are task-owned, not DIRECT search axes.
    return {"mode": "full", "maximum_epochs": 300, "fold_count": 5}


def smoke_configuration() -> dict:
    return {"mode": "smoke", "maximum_epochs": 1, "fold_count": 1}
