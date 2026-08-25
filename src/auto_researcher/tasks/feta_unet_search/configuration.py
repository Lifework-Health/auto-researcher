"""Bounded U-Net family configuration for planner-driven development."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

CONFIGURATION_SCHEMA_VERSION = "feta-unet-search-configuration-v6"
V7_CONFIGURATION_SCHEMA_VERSION = "feta-unet-search-configuration-v5"
V9_CONFIGURATION_SCHEMA_VERSION = "feta-unet-search-configuration-v7"
FIDELITY_LEVELS = (5, 10, 15, 25, 50, 100, 150)
LEARNING_RATE_BOUNDS = (3e-5, 5e-4)
WEIGHT_DECAY_BOUNDS = (1e-6, 3e-4)
DROPOUT_BOUNDS = (0.0, 0.3)
DICE_WEIGHT_BOUNDS = (0.5, 1.5)
POSITIVE_NEGATIVE_RATIOS = ("1:1", "2:1", "3:1")
MODEL_VARIANTS = ("basic_unet", "unet_plain", "unet_residual")
MODEL_VARIANT_CONTEXT = {
    "basic_unet": ("BasicUNet", 0),
    "unet_plain": ("UNet", 0),
    "unet_residual": ("UNet", 2),
    "structural_basic_unet": ("BasicUNet", 0),
    "dynunet": ("DynUNet", 0),
    "attention_unet": ("AttentionUnet", 0),
    "unetr": ("UNETR", 0),
    "swin_unetr": ("SwinUNETR", 0),
}
FEATURE_WIDTH_PROFILES = {
    "narrow": (24, 24, 48, 96, 192, 24),
    "baseline": (32, 32, 64, 128, 256, 32),
    "wide": (40, 40, 80, 160, 320, 40),
}
V6_BASIC_UNET_FEATURE_PROFILES = {
    "v6_balanced_64": (64, 64, 128, 256, 512, 64),
    "v6_balanced_80": (80, 80, 160, 320, 640, 80),
    "v6_balanced_96": (96, 96, 192, 384, 768, 96),
    "v6_balanced_112": (112, 112, 224, 448, 896, 112),
    "v6_balanced_128": (128, 128, 256, 512, 1024, 128),
    "v6_balanced_144": (144, 144, 288, 576, 1152, 144),
    "v6_deep_64": (48, 64, 128, 320, 640, 64),
    "v6_deep_80": (48, 80, 160, 400, 800, 80),
    "v6_decoder_96": (96, 96, 160, 256, 512, 128),
}
ALL_FEATURE_WIDTH_PROFILES = {
    **FEATURE_WIDTH_PROFILES,
    **V6_BASIC_UNET_FEATURE_PROFILES,
}
V6_ARCHITECTURE_BUDGET = "basicunet-15m-150m-v1"
V6_MINIMUM_TRAINABLE_PARAMETERS = 15_000_000
V6_MAXIMUM_TRAINABLE_PARAMETERS = 150_000_000
V6_UPSAMPLE_MODES = ("deconv", "pixelshuffle", "nontrainable")
V6_PIXELSHUFFLE_FEATURE_PROFILES = frozenset(
    {
        "v6_balanced_64",
        "v6_balanced_80",
        "v6_balanced_96",
        "v6_deep_64",
        "v6_deep_80",
        "v6_decoder_96",
    }
)
V6_OPTUNA_FEATURE_PROFILES = (
    "v6_balanced_64",
    "v6_balanced_80",
    "v6_balanced_96",
    "v6_deep_64",
    "v6_deep_80",
    "v6_decoder_96",
)
V7_MECHANISM_FEATURE_PROFILES = {
    # BasicUNet semantics: encoder widths followed by the terminal decoder width.
    # Five-value profiles have four resolution stages; six-value profiles have
    # the reference five stages.  This keeps every V7 candidate in one explicit
    # BasicUNet lineage while allowing depth and width asymmetry to evolve.
    "v7_compact_5": (48, 96, 192, 384, 48),
    "v7_balanced_5": (64, 128, 256, 512, 64),
    "v7_asymmetric_5": (64, 96, 192, 480, 64),
    "v7_deep_6": (40, 80, 160, 320, 640, 40),
    "v7_wide_6": (80, 160, 320, 640, 960, 80),
}
ALL_FEATURE_WIDTH_PROFILES.update(V7_MECHANISM_FEATURE_PROFILES)
V7_ARCHITECTURE_BUDGET = "basicunet-structural-15m-150m-v1"
V7_MINIMUM_TRAINABLE_PARAMETERS = 15_000_000
V7_MAXIMUM_TRAINABLE_PARAMETERS = 150_000_000
V7_MAXIMUM_PEAK_GPU_MEMORY_BYTES = 44 * 1024**3
V7_OPTUNA_FEATURE_PROFILES = tuple(V7_MECHANISM_FEATURE_PROFILES)
V7_KERNEL_PROFILES = (
    "standard",
    "large_front",
    "context_deep",
)
V7_DEEP_SUPERVISION_HEADS = (0, 1, 2)
V8_DYNUNET_FEATURE_PROFILES = {
    "v8_dyn_compact_5": (48, 96, 192, 384, 768),
    "v8_dyn_balanced_5": (64, 128, 256, 512, 768),
    "v8_dyn_context_5": (64, 96, 192, 480, 960),
    "v8_dyn_deep_6": (40, 80, 160, 320, 640, 960),
}
ALL_FEATURE_WIDTH_PROFILES.update(V8_DYNUNET_FEATURE_PROFILES)
V8_DYNUNET_ARCHITECTURE_BUDGET = "dynunet-15m-150m-v1"
V8_CAMPAIGN_ARCHITECTURE_MODE = "feta-unet-v8-mixed-bounded-v1"
V8_ARCHITECTURE_FAMILY_ID = "feta-unet-v8-mixed-bounded-family-v1"
V8_MINIMUM_TRAINABLE_PARAMETERS = 15_000_000
V8_MAXIMUM_TRAINABLE_PARAMETERS = 150_000_000
V8_MAXIMUM_PEAK_GPU_MEMORY_BYTES = 44 * 1024**3
V8_DYNUNET_KERNEL_PROFILES = ("standard", "large_front", "context_deep")
V8_DYNUNET_DEEP_SUPERVISION_HEADS = (0, 1, 2)
V8_STAGE_BLOCK_PROFILES = (
    "uniform",
    "shallow_to_deep",
    "deep_to_shallow",
    "bottleneck_heavy",
)
V8_RESIDUAL_PROFILES = (
    "uniform",
    "encoder_only",
    "decoder_only",
    "deep_only",
)
V9_ATTENTION_FEATURE_PROFILES = {
    "v9_attn_compact_5": (40, 80, 160, 320, 640),
    "v9_attn_balanced_5": (48, 96, 192, 384, 768),
}
V9_TRANSFORMER_FEATURE_PROFILES = {
    # The tuple remains part of the common scientific configuration identity.
    # Transformer internals are fixed by the named profile in model.py.
    "v9_unetr_base_16": (32, 64, 128, 256, 512),
    "v9_swin_tiny_24": (24, 48, 96, 192, 384),
}
ALL_FEATURE_WIDTH_PROFILES.update(V9_ATTENTION_FEATURE_PROFILES)
ALL_FEATURE_WIDTH_PROFILES.update(V9_TRANSFORMER_FEATURE_PROFILES)
V9_ATTENTION_ARCHITECTURE_BUDGET = "attention-unet-15m-150m-v1"
V9_TRANSFORMER_ARCHITECTURE_BUDGET = "transformer-pilot-15m-150m-v1"
V9_MINIMUM_TRAINABLE_PARAMETERS = 15_000_000
V9_MAXIMUM_TRAINABLE_PARAMETERS = 150_000_000
V9_MAXIMUM_PEAK_GPU_MEMORY_BYTES = 44 * 1024**3
RESIDUAL_CHANNEL_PROFILES = {
    "narrow": (24, 48, 96, 192, 384),
    "baseline": (32, 64, 128, 256, 512),
    "wide": (40, 80, 160, 320, 640),
}
ACTIVATIONS = ("LeakyReLU", "ReLU", "PReLU")
NORMALISATIONS = ("instance", "group")
OPTIMISERS = ("AdamW", "Adam")
LEARNING_RATE_SCHEDULES = ("constant", "cosine", "polynomial")
LOSS_VARIANTS = ("dice_ce", "dice_focal", "dice_tversky")
AUGMENTATION_POLICIES = (
    "reference_light",
    "geometric",
    "intensity",
    "combined",
)
SEARCH_ARCHITECTURE_FAMILY_ID = "basicunet-3d-structural-bounded-family-v1"
CANDIDATE_CONFIGURATION_FIELDS = (
    "maximum_epochs",
    "model_variant",
    "feature_width",
    "features",
    "architecture_budget",
    "upsample",
    "kernel_profile",
    "residual_blocks",
    "deep_supervision_heads",
    "convolutions_per_stage",
    "stage_block_profile",
    "residual_profile",
    "dilation_profile",
    "skip_fusion",
    "downsample",
    "activation",
    "norm",
    "optimizer",
    "lr_schedule",
    "loss_variant",
    "learning_rate",
    "weight_decay",
    "dropout",
    "dice_weight",
    "positive_negative_ratio",
    "augmentation_policy",
)


class FeTAUNetSearchConfiguration(BaseModel):
    """A fold-0 U-Net candidate with a bounded v5 mutable surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: Literal["development_baseline"] = "development_baseline"
    spatial_dims: Literal[3] = 3
    in_channels: Literal[1] = 1
    out_channels: Literal[8] = 8
    model_variant: Literal[
        "basic_unet",
        "unet_plain",
        "unet_residual",
        "structural_basic_unet",
        "dynunet",
        "attention_unet",
        "unetr",
        "swin_unetr",
    ] = "basic_unet"
    network_family: Literal[
        "BasicUNet", "UNet", "DynUNet", "AttentionUnet", "UNETR", "SwinUNETR"
    ] = "BasicUNet"
    residual_units: Literal[0, 2] = 0
    feature_width: str = "baseline"
    features: tuple[int, int, int, int, int] | tuple[int, int, int, int, int, int] = (
        FEATURE_WIDTH_PROFILES["baseline"]
    )
    channels: tuple[int, int, int, int, int] = RESIDUAL_CHANNEL_PROFILES["baseline"]
    strides: tuple[int, int, int, int] = (2, 2, 2, 2)
    activation: Literal["LeakyReLU", "ReLU", "PReLU"] = "LeakyReLU"
    negative_slope: float = 0.1
    activation_inplace: Literal[True] = True
    norm: Literal["instance", "group"] = "instance"
    norm_affine: Literal[True] = True
    norm_num_groups: Literal[8] = 8
    architecture_budget: Literal[
        "legacy",
        "basicunet-15m-150m-v1",
        "basicunet-structural-15m-150m-v1",
        "dynunet-15m-150m-v1",
        "attention-unet-15m-150m-v1",
        "transformer-pilot-15m-150m-v1",
    ] = "legacy"
    upsample: Literal["deconv", "pixelshuffle", "nontrainable"] = "deconv"
    kernel_profile: Literal["basic", "standard", "large_front", "context_deep"] = (
        "basic"
    )
    residual_blocks: bool = False
    deep_supervision_heads: Literal[0, 1, 2] = 0
    convolutions_per_stage: Literal[1, 2, 3] = 2
    stage_block_profile: Literal[
        "uniform", "shallow_to_deep", "deep_to_shallow", "bottleneck_heavy"
    ] = "uniform"
    residual_profile: Literal[
        "uniform", "encoder_only", "decoder_only", "deep_only"
    ] = "uniform"
    dilation_profile: Literal["none", "deep", "multiscale"] = "none"
    skip_fusion: Literal["concat", "add", "gated_concat"] = "concat"
    downsample: Literal["max_pool", "strided_conv"] = "max_pool"
    spacing_mm: tuple[float, float, float] = (0.5, 0.5, 0.5)
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: Literal[1] = 1
    samples_per_volume: Literal[2] = 2
    maximum_epochs: Literal[5, 10, 15, 25, 50, 100, 150] = 25
    validation_every: Literal[5] = 5
    fold_count: Literal[1] = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.0
    dice_weight: float = 1.0
    positive_negative_ratio: Literal["1:1", "2:1", "3:1"] = "1:1"
    augmentation_policy: Literal[
        "reference_light", "geometric", "intensity", "combined"
    ] = "reference_light"
    optimizer: Literal["AdamW", "Adam"] = "AdamW"
    lr_schedule: Literal["constant", "cosine", "polynomial"] = "constant"
    scheduler_horizon_epochs: Literal[150] = 150
    polynomial_power: Literal[0.9] = 0.9
    loss_variant: Literal["dice_ce", "dice_focal", "dice_tversky"] = "dice_ce"
    inference_overlap: float = 0.5
    inference_blending: Literal["gaussian"] = "gaussian"
    sliding_window_batch_size: Literal[1] = 1
    # The alternate value is reserved for the single predeclared V8 finalist
    # confirmation.  Seed is intentionally absent from every search space, so
    # it cannot become an optimisation dimension.
    seed: Literal[20260807, 20260824] = 20260807
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

    @model_validator(mode="before")
    @classmethod
    def derive_feature_tuple(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        profile = payload.get("feature_width")
        if (
            payload.get("architecture_budget") == V6_ARCHITECTURE_BUDGET
            and profile is None
        ):
            profile = "v6_balanced_64"
            payload["feature_width"] = profile
        if profile in V6_BASIC_UNET_FEATURE_PROFILES or profile == "custom":
            payload.setdefault("architecture_budget", V6_ARCHITECTURE_BUDGET)
            payload.setdefault("model_variant", "basic_unet")
        if profile in V7_MECHANISM_FEATURE_PROFILES:
            payload.setdefault("architecture_budget", V7_ARCHITECTURE_BUDGET)
            payload.setdefault("model_variant", "structural_basic_unet")
        if profile in V8_DYNUNET_FEATURE_PROFILES:
            payload.setdefault("architecture_budget", V8_DYNUNET_ARCHITECTURE_BUDGET)
            payload.setdefault("model_variant", "dynunet")
        if profile in V9_ATTENTION_FEATURE_PROFILES:
            payload.setdefault("architecture_budget", V9_ATTENTION_ARCHITECTURE_BUDGET)
            payload.setdefault("model_variant", "attention_unet")
        if profile == "v9_unetr_base_16":
            payload.setdefault("architecture_budget", V9_TRANSFORMER_ARCHITECTURE_BUDGET)
            payload.setdefault("model_variant", "unetr")
        if profile == "v9_swin_tiny_24":
            payload.setdefault("architecture_budget", V9_TRANSFORMER_ARCHITECTURE_BUDGET)
            payload.setdefault("model_variant", "swin_unetr")
        profile = payload.get("feature_width", "baseline")
        expected = ALL_FEATURE_WIDTH_PROFILES.get(profile)
        if expected is not None and "features" not in payload:
            payload["features"] = expected
        channels = RESIDUAL_CHANNEL_PROFILES.get(profile)
        if channels is not None and "channels" not in payload:
            payload["channels"] = channels
        variant = payload.get("model_variant", "basic_unet")
        context = MODEL_VARIANT_CONTEXT.get(variant)
        if context is not None:
            payload.setdefault("network_family", context[0])
            payload.setdefault("residual_units", context[1])
        return payload

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

    @field_validator("feature_width")
    @classmethod
    def feature_width_is_registered(cls, value: str) -> str:
        if value != "custom" and value not in ALL_FEATURE_WIDTH_PROFILES:
            raise ValueError("feta_unet_search_feature_width_unregistered")
        return value

    @staticmethod
    def _bounded(value: float, bounds: tuple[float, float], name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or not bounds[0] <= result <= bounds[1]:
            raise ValueError(f"feta_unet_search_{name}_out_of_bounds")
        return result

    @model_validator(mode="after")
    def bounded_search_profile(self) -> "FeTAUNetSearchConfiguration":
        # Deliberately do not call the frozen DIRECT validator: this sibling
        # task varies only the registered architecture/training surface while
        # retaining the preprocessing, fold and inference identities.
        expected_features = ALL_FEATURE_WIDTH_PROFILES.get(self.feature_width)
        legacy_architecture = self.architecture_budget == "legacy"
        if legacy_architecture and (
            self.feature_width not in FEATURE_WIDTH_PROFILES
            or self.features != expected_features
            or self.upsample != "deconv"
            or self.kernel_profile != "basic"
            or self.residual_blocks
            or self.deep_supervision_heads != 0
            or self.convolutions_per_stage != 2
            or self.stage_block_profile != "uniform"
            or self.residual_profile != "uniform"
            or self.dilation_profile != "none"
            or self.skip_fusion != "concat"
            or self.downsample != "max_pool"
        ):
            raise ValueError("feta_unet_search_fixed_context_modified")
        if self.architecture_budget == V6_ARCHITECTURE_BUDGET and (
            self.model_variant != "basic_unet"
            or self.feature_width not in {*V6_BASIC_UNET_FEATURE_PROFILES, "custom"}
            or (expected_features is not None and self.features != expected_features)
            or len(self.features) != 6
            or any(
                channel % 8 or channel < 32 or channel > 1_280
                for channel in self.features
            )
            or tuple(sorted(self.features[:5])) != self.features[:5]
            or not 32 <= self.features[5] <= 256
            or self.upsample not in V6_UPSAMPLE_MODES
            or self.kernel_profile != "basic"
            or self.residual_blocks
            or self.deep_supervision_heads != 0
            or self.convolutions_per_stage != 2
            or self.stage_block_profile != "uniform"
            or self.residual_profile != "uniform"
            or self.dilation_profile != "none"
            or self.skip_fusion != "concat"
            or self.downsample != "max_pool"
            or (
                self.upsample == "pixelshuffle"
                and self.feature_width != "custom"
                and self.feature_width not in V6_PIXELSHUFFLE_FEATURE_PROFILES
            )
        ):
            raise ValueError("feta_unet_search_v6_architecture_invalid")
        if self.architecture_budget == V7_ARCHITECTURE_BUDGET and (
            self.model_variant != "structural_basic_unet"
            or self.feature_width not in {*V7_MECHANISM_FEATURE_PROFILES, "custom"}
            or (expected_features is not None and self.features != expected_features)
            or len(self.features) not in (5, 6)
            or any(
                channel % 8 or channel < 32 or channel > 1_024
                for channel in self.features
            )
            or tuple(sorted(self.features[:-1])) != self.features[:-1]
            or not 32 <= self.features[-1] <= 256
            or self.upsample not in V6_UPSAMPLE_MODES
            or self.kernel_profile not in V7_KERNEL_PROFILES
            or self.deep_supervision_heads >= len(self.features) - 1
        ):
            raise ValueError("feta_unet_search_v7_architecture_invalid")
        if self.architecture_budget == V8_DYNUNET_ARCHITECTURE_BUDGET and (
            self.model_variant != "dynunet"
            or self.feature_width not in {*V8_DYNUNET_FEATURE_PROFILES, "custom"}
            or (expected_features is not None and self.features != expected_features)
            or len(self.features) not in (5, 6)
            or any(
                channel % 8 or channel < 32 or channel > 1_024
                for channel in self.features
            )
            or tuple(sorted(self.features)) != self.features
            or self.upsample != "deconv"
            or self.kernel_profile not in V8_DYNUNET_KERNEL_PROFILES
            or self.deep_supervision_heads not in V8_DYNUNET_DEEP_SUPERVISION_HEADS
            or self.deep_supervision_heads >= len(self.features) - 1
            or self.convolutions_per_stage != 2
            or self.stage_block_profile != "uniform"
            or self.residual_profile != "uniform"
            or self.dilation_profile != "none"
            or self.skip_fusion != "concat"
            or self.downsample != "max_pool"
        ):
            raise ValueError("feta_unet_search_v8_dynunet_architecture_invalid")
        if self.architecture_budget == V9_ATTENTION_ARCHITECTURE_BUDGET and (
            self.model_variant != "attention_unet"
            or self.feature_width not in V9_ATTENTION_FEATURE_PROFILES
            or self.features != expected_features
            or len(self.features) != 5
            or tuple(sorted(self.features)) != self.features
            or self.upsample != "deconv"
            or self.kernel_profile != "standard"
            or self.residual_blocks
            or self.deep_supervision_heads != 0
            or self.convolutions_per_stage != 2
            or self.stage_block_profile != "uniform"
            or self.residual_profile != "uniform"
            or self.dilation_profile != "none"
            or self.skip_fusion != "gated_concat"
            or self.downsample != "max_pool"
        ):
            raise ValueError("feta_unet_search_v9_attention_architecture_invalid")
        if self.architecture_budget == V9_TRANSFORMER_ARCHITECTURE_BUDGET and (
            self.feature_width not in V9_TRANSFORMER_FEATURE_PROFILES
            or self.features != expected_features
            or (
                self.feature_width == "v9_unetr_base_16"
                and self.model_variant != "unetr"
            )
            or (
                self.feature_width == "v9_swin_tiny_24"
                and self.model_variant != "swin_unetr"
            )
            or self.upsample != "deconv"
            or self.kernel_profile != "standard"
            or not self.residual_blocks
            or self.deep_supervision_heads != 0
            or self.convolutions_per_stage != 2
            or self.stage_block_profile != "uniform"
            or self.residual_profile != "uniform"
            or self.dilation_profile != "none"
            or self.skip_fusion != "concat"
            or self.downsample != "max_pool"
            or self.norm != "instance"
        ):
            raise ValueError("feta_unet_search_v9_transformer_architecture_invalid")
        expected_channels = RESIDUAL_CHANNEL_PROFILES.get(self.feature_width)
        if (
            (
                self.model_variant in {"unet_plain", "unet_residual"}
                and self.channels != expected_channels
            )
            or (self.network_family, self.residual_units)
            != MODEL_VARIANT_CONTEXT[self.model_variant]
            or self.strides != (2, 2, 2, 2)
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
