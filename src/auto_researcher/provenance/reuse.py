"""Durable, per-run evaluation and verification reuse records."""

from __future__ import annotations

from pydantic import Field

from auto_researcher.contracts.models import (
    EvaluationResult,
    ImmutableDomainModel,
    VerificationResult,
)

EVALUATION_REUSE_PROTOCOL = "evaluation-reuse-v1"


class EvaluationReuseRecord(ImmutableDomainModel):
    protocol_version: str = EVALUATION_REUSE_PROTOCOL
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    scientific_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    result: EvaluationResult


class VerificationReuseRecord(ImmutableDomainModel):
    protocol_version: str = EVALUATION_REUSE_PROTOCOL
    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    scientific_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_version: str = Field(min_length=1)
    verification_policy_version: str = Field(min_length=1)
    result: VerificationResult
