"""Immutable identities for development-only FeTA ensemble evaluation."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


SHA256_PATTERN = r"^[0-9a-f]{64}$"
ENSEMBLE_SCHEMA_VERSION = "feta-unet-ensemble-specification-v1"
PROBABILITY_CACHE_SCHEMA_VERSION = "feta-unet-probability-cache-v1"


class EnsembleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EnsembleMember(EnsembleModel):
    """One verified checkpoint and its compatibility-critical identities."""

    experiment_id: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    configuration_identity: str = Field(pattern=SHA256_PATTERN)
    architecture_identity: str = Field(min_length=1)
    dataset_manifest_hash: str = Field(min_length=1)
    split_hash: str = Field(min_length=1)
    fold_hash: str = Field(min_length=1)
    preprocessing_identity: str = Field(min_length=1)
    label_mapping_identity: str = Field(min_length=1)
    inference_identity: str = Field(min_length=1)
    output_classes: tuple[int, ...] = tuple(range(8))

    @model_validator(mode="after")
    def output_class_contract_is_exact(self) -> "EnsembleMember":
        if self.output_classes != tuple(range(8)):
            raise ValueError("feta_unet_ensemble_output_classes_invalid")
        return self

    def compatibility_identity(self) -> tuple[object, ...]:
        return (
            self.dataset_manifest_hash,
            self.split_hash,
            self.fold_hash,
            self.preprocessing_identity,
            self.label_mapping_identity,
            self.inference_identity,
            self.output_classes,
        )


class EnsembleSpecification(EnsembleModel):
    """A pre-specified two-to-four-member probability-mean ensemble."""

    schema_version: str = ENSEMBLE_SCHEMA_VERSION
    ensemble_id: str = Field(min_length=1)
    members: tuple[EnsembleMember, ...] = Field(min_length=2, max_length=4)
    weights: tuple[float, ...] = Field(min_length=2, max_length=4)
    aggregation: str = "per-class-probability-mean"
    selection_rule: str = Field(min_length=1)
    protected_development_only: bool = True
    sealed_holdout_evaluations: int = 0

    @model_validator(mode="after")
    def ensemble_contract_is_valid(self) -> "EnsembleSpecification":
        if self.schema_version != ENSEMBLE_SCHEMA_VERSION:
            raise ValueError("feta_unet_ensemble_schema_invalid")
        if self.aggregation != "per-class-probability-mean":
            raise ValueError("feta_unet_ensemble_aggregation_invalid")
        if not self.protected_development_only or self.sealed_holdout_evaluations != 0:
            raise ValueError("feta_unet_ensemble_holdout_boundary_invalid")
        if len(self.weights) != len(self.members):
            raise ValueError("feta_unet_ensemble_weight_count_mismatch")
        if any(not math.isfinite(value) or value < 0.0 for value in self.weights):
            raise ValueError("feta_unet_ensemble_weight_invalid")
        if not math.isclose(sum(self.weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("feta_unet_ensemble_weight_sum_invalid")
        experiment_ids = [item.experiment_id for item in self.members]
        checkpoint_ids = [item.checkpoint_sha256 for item in self.members]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("feta_unet_ensemble_member_duplicate")
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ValueError("feta_unet_ensemble_checkpoint_duplicate")
        identities = {item.compatibility_identity() for item in self.members}
        if len(identities) != 1:
            raise ValueError("feta_unet_ensemble_member_incompatible")
        return self


class ProbabilityCacheRecord(EnsembleModel):
    """Protected reference for one subject/member probability tensor."""

    schema_version: str = PROBABILITY_CACHE_SCHEMA_VERSION
    subject_id: str = Field(min_length=1)
    member_identity: str = Field(pattern=SHA256_PATTERN)
    probability_sha256: str = Field(pattern=SHA256_PATTERN)
    shape: tuple[int, int, int, int]
    dtype: str = "float32"
    size_bytes: int = Field(ge=1)
    contains_subject_identifier: bool = True

    @model_validator(mode="after")
    def cache_contract_is_valid(self) -> "ProbabilityCacheRecord":
        if self.schema_version != PROBABILITY_CACHE_SCHEMA_VERSION:
            raise ValueError("feta_unet_probability_cache_schema_invalid")
        if self.shape[0] != 8 or any(value < 1 for value in self.shape):
            raise ValueError("feta_unet_probability_cache_shape_invalid")
        if self.dtype != "float32" or not self.contains_subject_identifier:
            raise ValueError("feta_unet_probability_cache_contract_invalid")
        return self
