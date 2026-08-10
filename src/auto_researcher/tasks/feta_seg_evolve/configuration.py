"""Immutable FeTA context plus a bounded evolvable TrainingPolicy."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_seg_search.configuration import (
    DICE_WEIGHT_BOUNDS,
    DROPOUT_BOUNDS,
    LEARNING_RATE_BOUNDS,
    WEIGHT_DECAY_BOUNDS,
)
from auto_researcher.tasks.feta_seg_evolve.training_policy import (
    TrainingPolicy,
    default_training_policy,
)

EVOLVE_CONFIGURATION_VERSION = "feta-segresnet-evolve-configuration-v1"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvolveBaseConfiguration(FrozenModel):
    fold: Literal[0] = 0
    maximum_epochs: Literal[25, 50, 100] = 25
    learning_rate: float = Field(
        default=1e-4, ge=LEARNING_RATE_BOUNDS[0], le=LEARNING_RATE_BOUNDS[1]
    )
    weight_decay: float = Field(
        default=1e-5, ge=WEIGHT_DECAY_BOUNDS[0], le=WEIGHT_DECAY_BOUNDS[1]
    )
    dropout: float = Field(default=0.2, ge=DROPOUT_BOUNDS[0], le=DROPOUT_BOUNDS[1])
    dice_weight: float = Field(
        default=1.0, ge=DICE_WEIGHT_BOUNDS[0], le=DICE_WEIGHT_BOUNDS[1]
    )
    positive_negative_ratio: Literal["1:1", "2:1", "3:1"] = "1:1"
    augmentation_strength: Literal["light", "baseline", "strong"] = "baseline"
    spatial_dims: Literal[3] = 3
    in_channels: Literal[1] = 1
    out_channels: Literal[8] = 8
    init_filters: Literal[32] = 32
    blocks_down: tuple[Literal[1], Literal[2], Literal[2], Literal[4]] = (1, 2, 2, 4)
    blocks_up: tuple[Literal[1], Literal[1], Literal[1]] = (1, 1, 1)
    norm: Literal["GROUP"] = "GROUP"
    activation: Literal["RELU"] = "RELU"
    upsample_mode: Literal["deconv"] = "deconv"
    spacing_mm: tuple[Literal[0.5], Literal[0.5], Literal[0.5]] = (0.5, 0.5, 0.5)
    patch_size: tuple[Literal[128], Literal[128], Literal[128]] = (128, 128, 128)
    batch_size: Literal[1] = 1
    samples_per_volume: Literal[2] = 2
    ce_weight: Literal[1.0] = 1.0
    inference_overlap: Literal[0.5] = 0.5
    inference_blending: Literal["gaussian"] = "gaussian"
    sliding_window_batch_size: Literal[1] = 1
    seed: Literal[20260807] = 20260807


class CandidateProvenance(FrozenModel):
    candidate_id: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: int = Field(ge=0)
    parent_candidate_ids: tuple[str, ...] = ()
    creation_provenance: str = Field(min_length=1)


class FeTASegEvolveConfiguration(FrozenModel):
    configuration_version: Literal["feta-segresnet-evolve-configuration-v1"] = (
        EVOLVE_CONFIGURATION_VERSION
    )
    base_configuration: EvolveBaseConfiguration
    training_policy: TrainingPolicy
    seeding_mode: Literal["pure", "optuna"]
    base_configuration_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_policy_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_provenance: CandidateProvenance

    @model_validator(mode="after")
    def identities_match_payloads(self) -> "FeTASegEvolveConfiguration":
        if self.base_configuration_identity != payload_hash(self.base_configuration):
            raise ValueError("feta_evolve_base_configuration_identity_mismatch")
        if self.training_policy_identity != payload_hash(self.training_policy):
            raise ValueError("feta_evolve_training_policy_identity_mismatch")
        return self

    def scientific_configuration(self) -> dict[str, Any]:
        return {
            "configuration_version": self.configuration_version,
            "base_configuration": self.base_configuration.model_dump(mode="json"),
            "training_policy": self.training_policy.model_dump(mode="json"),
            "seeding_mode": self.seeding_mode,
            "base_configuration_identity": self.base_configuration_identity,
            "training_policy_identity": self.training_policy_identity,
        }

    def __getattr__(self, name: str) -> Any:
        base = object.__getattribute__(self, "base_configuration")
        if name in EvolveBaseConfiguration.model_fields:
            return getattr(base, name)
        raise AttributeError(name)


def base_configuration_from_runtime(
    options: dict[str, Any],
) -> tuple[EvolveBaseConfiguration, Literal["pure", "optuna"]]:
    raw = options.get("base_configuration")
    if raw is None:
        return EvolveBaseConfiguration(), "pure"
    return EvolveBaseConfiguration.model_validate(raw), "optuna"


def build_evolve_configuration(
    base: EvolveBaseConfiguration,
    policy: TrainingPolicy,
    *,
    seeding_mode: Literal["pure", "optuna"],
    candidate_provenance: CandidateProvenance,
) -> FeTASegEvolveConfiguration:
    return FeTASegEvolveConfiguration(
        base_configuration=base,
        training_policy=policy,
        seeding_mode=seeding_mode,
        base_configuration_identity=payload_hash(base),
        training_policy_identity=payload_hash(policy),
        candidate_provenance=candidate_provenance,
    )


def default_evolve_configuration() -> dict[str, Any]:
    base = EvolveBaseConfiguration()
    policy = default_training_policy()
    return build_evolve_configuration(
        base,
        policy,
        seeding_mode="pure",
        candidate_provenance=CandidateProvenance(
            candidate_id="direct-seed",
            source_hash="0" * 64,
            generation=0,
            creation_provenance="DIRECT",
        ),
    ).model_dump(mode="json")
