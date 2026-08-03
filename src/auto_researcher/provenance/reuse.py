"""Durable, per-run evaluation and verification reuse records."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from auto_researcher.contracts.models import (
    EvaluationResult,
    ImmutableDomainModel,
    VerificationResult,
)

EVALUATION_REUSE_PROTOCOL: Literal["evaluation-reuse-v2"] = "evaluation-reuse-v2"
LEGACY_EVALUATION_REUSE_PROTOCOL = "evaluation-reuse-v1"
VERIFICATION_REUSE_PROTOCOL = "verification-reuse-v1"


class EvaluationReuseRecord(ImmutableDomainModel):
    protocol_version: Literal["evaluation-reuse-v2"] = EVALUATION_REUSE_PROTOCOL
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    scientific_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    artefact_bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artefact_bundle_schema_version: str = Field(min_length=1)
    result_encoding_version: str = Field(min_length=1)
    expected_artefact_references: tuple[str, ...] = Field(min_length=1)
    evaluator_manifest_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_at: datetime
    result: EvaluationResult


class VerificationReuseRecord(ImmutableDomainModel):
    protocol_version: str = VERIFICATION_REUSE_PROTOCOL
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    scientific_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_version: str = Field(min_length=1)
    verification_policy_version: str = Field(min_length=1)
    evaluation_reuse_identity_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    result: VerificationResult
