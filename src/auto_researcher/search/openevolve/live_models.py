"""Immutable contracts for approved, bounded OpenEvolve mutation calls."""

from __future__ import annotations

import re
import math
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from auto_researcher.agents.models import ModelCallConfig
from auto_researcher.runtime.identity import payload_hash


class LiveMutationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LiveMutationApproval(LiveMutationModel):
    protocol_version: Literal["live-mutation-approval-v1"] = "live-mutation-approval-v1"
    approval_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    component_version: str = Field(min_length=1)
    adapter_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    adapter_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    mutation_operator_version: str = Field(min_length=1)
    maximum_model_calls: int = Field(ge=1)
    maximum_input_tokens: int = Field(ge=1)
    maximum_output_tokens: int = Field(ge=1)
    maximum_total_cost: float = Field(gt=0)
    currency: str = Field(min_length=1)
    pricing_version: str = Field(min_length=1)
    executor_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mutable_file: str = Field(pattern=r"^[A-Za-z0-9_.-]+\.py$")
    permitted_dataset_class: Literal["synthetic"] = "synthetic"
    prohibited_dataset_classes: tuple[str, ...] = (
        "aura",
        "genuine_icca",
        "mri",
        "patient_data",
    )
    created_at: AwareDatetime
    expires_at: AwareDatetime
    reviewer_identity: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    residual_risk_acknowledged: Literal[True]
    aura_access: Literal[False] = False
    patient_data_access: Literal[False] = False
    mri_access: Literal[False] = False
    direct_upstream_provider_access: Literal[False] = False
    local_subprocess_fallback: Literal[False] = False
    model_retries: Literal[False] = False
    package_installation: Literal[False] = False
    network_access: Literal[False] = False
    multiple_mutable_files: Literal[False] = False
    evaluator_or_verifier_mutation: Literal[False] = False
    approval_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("prohibited_dataset_classes", mode="before")
    @classmethod
    def canonical_prohibited_classes(cls, value):
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise ValueError("live_mutation_approval_mismatch")
        values = tuple(str(item) for item in value)
        if len(values) != len(set(values)):
            raise ValueError("live_mutation_approval_mismatch")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def immutable_identity_is_valid(
        self, info: ValidationInfo
    ) -> "LiveMutationApproval":
        if self.created_at.astimezone(UTC) >= self.expires_at.astimezone(UTC):
            raise ValueError("live_mutation_approval_expired")
        if self.expires_at.astimezone(UTC) - self.created_at.astimezone(
            UTC
        ) > timedelta(hours=24):
            raise ValueError("live_mutation_approval_expiry_too_long")
        if self.model_id.lower().endswith(("-latest", ":latest")):
            raise ValueError("live_mutation_model_not_approved")
        if set(self.prohibited_dataset_classes) != {
            "aura",
            "genuine_icca",
            "mri",
            "patient_data",
        }:
            raise ValueError("live_mutation_approval_mismatch")
        if not (info.context or {}).get("skip_approval_hash") and (
            approval_content_hash(self) != self.approval_hash
        ):
            raise ValueError("live_mutation_approval_tampered")
        return self


def approval_content_hash(approval: LiveMutationApproval | dict) -> str:
    if isinstance(approval, LiveMutationApproval):
        payload = approval.model_dump(mode="python")
    else:
        raw = dict(approval)
        raw.pop("approval_hash", None)
        # Validate and materialise defaults before hashing. Validation context
        # skips only the self-referential final hash comparison.
        payload = LiveMutationApproval.model_validate(
            {**raw, "approval_hash": "0" * 64},
            context={"skip_approval_hash": True},
        ).model_dump(mode="python")
    payload.pop("approval_hash", None)
    return payload_hash(
        {
            "domain": "auto-researcher-live-mutation-approval",
            "version": "canonical-json-sha256-v1",
            "payload": payload,
        }
    )


def parse_live_mutation_approval(payload: dict) -> LiveMutationApproval:
    return LiveMutationApproval.model_validate(payload)


class OpenEvolveModelBridgeContract(LiveMutationModel):
    protocol_version: Literal["openevolve-model-bridge-v1"] = (
        "openevolve-model-bridge-v1"
    )
    bridge_id: Literal["auto-researcher-openevolve-model-bridge"] = (
        "auto-researcher-openevolve-model-bridge"
    )
    bridge_version: Literal["1"] = "1"
    supported_adapter_version: Literal["1"] = "1"
    supported_search_contract_version: Literal["openevolve-search-v1"] = (
        "openevolve-search-v1"
    )
    mutation_operator_id: str = Field(min_length=1)
    mutation_operator_version: str = Field(min_length=1)
    prompt_id: Literal["openevolve-mutation"] = "openevolve-mutation"
    prompt_version: Literal["openevolve-mutation-prompt-v1"] = (
        "openevolve-mutation-prompt-v1"
    )
    response_schema_version: Literal["upstream-mutation-envelope-v1"] = (
        "upstream-mutation-envelope-v1"
    )
    maximum_input_bytes: int = Field(gt=0, le=1_000_000)
    model_config_contract: ModelCallConfig = Field(alias="model_config")
    reservation_policy: Literal["durable-before-dispatch-v1"] = (
        "durable-before-dispatch-v1"
    )
    uncertain_outcome_policy: Literal["no-automatic-redispatch-v1"] = (
        "no-automatic-redispatch-v1"
    )
    accounting_policy: Literal["model-call-accounting-v1"] = "model-call-accounting-v1"
    approval_policy_version: Literal["live-mutation-approval-v1"] = (
        "live-mutation-approval-v1"
    )
    model_store_protocol_version: Literal["agent-call-store-v2"] = "agent-call-store-v2"
    provider_mode: Literal["auto_researcher_owned"] = "auto_researcher_owned"

    @model_validator(mode="after")
    def one_attempt_only(self) -> "OpenEvolveModelBridgeContract":
        config = self.model_config_contract
        if config.maximum_attempts != 1:
            raise ValueError("live mutation model retries are prohibited")
        if config.provider not in {"anthropic", "fake-production"}:
            raise ValueError("live_mutation_provider_not_supported")
        if config.prompt_version != self.prompt_version:
            raise ValueError("model_call_prompt_not_approved")
        finite = (
            config.temperature,
            config.timeout_seconds,
            config.maximum_cost_per_call,
            config.pricing.input_cost_per_million_tokens,
            config.pricing.output_cost_per_million_tokens,
        )
        if not all(math.isfinite(item) for item in finite):
            raise ValueError("openevolve_model_bridge_finite_limits_required")
        return self


class OpenEvolveModelCallContext(LiveMutationModel):
    run_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    search_request_id: str = Field(min_length=1)
    generation: int = Field(ge=1)
    parent_candidate_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    component_version: str = Field(min_length=1)
    component_interface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    adapter_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    executor_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    mutable_file: str = Field(pattern=r"^[A-Za-z0-9_.-]+\.py$")
    dataset_class: Literal["synthetic"] = "synthetic"
    model_budget_identity: str = Field(min_length=1)
    maximum_model_calls: int = Field(ge=1)
    maximum_model_cost: float = Field(gt=0)


def validate_approval(
    approval: LiveMutationApproval,
    context: OpenEvolveModelCallContext,
    bridge: OpenEvolveModelBridgeContract,
    *,
    now: datetime,
) -> None:
    if now.astimezone(UTC) >= approval.expires_at.astimezone(UTC):
        raise ValueError("live_mutation_approval_expired")
    if now.astimezone(UTC) < approval.created_at.astimezone(UTC):
        raise ValueError("live_mutation_approval_mismatch")
    config = bridge.model_config_contract
    expected = (
        (approval.run_id, context.run_id),
        (approval.contract_id, context.contract_id),
        (approval.contract_hash, context.contract_hash),
        (approval.task_id, context.task_id),
        (approval.task_version, context.task_version),
        (approval.component_id, context.component_id),
        (approval.component_version, context.component_version),
        (approval.adapter_id, context.adapter_id),
        (approval.adapter_version, context.adapter_version),
        (approval.adapter_identity_hash, context.adapter_identity_hash),
        (approval.provider, config.provider),
        (approval.model_id, config.model_id),
        (approval.prompt_id, bridge.prompt_id),
        (approval.prompt_version, bridge.prompt_version),
        (approval.mutation_operator_version, bridge.mutation_operator_version),
        (approval.pricing_version, config.pricing.version),
        (approval.currency, config.pricing.currency),
        (approval.executor_policy_hash, context.executor_policy_hash),
        (approval.image_digest, context.image_digest),
        (approval.mutable_file, context.mutable_file),
        (approval.permitted_dataset_class, context.dataset_class),
    )
    if context.adapter_version != bridge.supported_adapter_version:
        raise ValueError("live_mutation_approval_mismatch")
    if any(left != right for left, right in expected):
        raise ValueError("live_mutation_approval_mismatch")
    if config.maximum_output_tokens > approval.maximum_output_tokens:
        raise ValueError("live_mutation_approval_mismatch")
    if context.maximum_model_calls > approval.maximum_model_calls:
        raise ValueError("live_mutation_approval_mismatch")
    if context.maximum_model_cost > approval.maximum_total_cost:
        raise ValueError("model_call_cost_limit_exceeded")
    if config.maximum_cost_per_call > approval.maximum_total_cost:
        raise ValueError("model_call_cost_limit_exceeded")
    if re.search(
        r"(?i)(api[_-]?key|bearer|secret|password)", approval.model_dump_json()
    ):
        raise ValueError("live_mutation_approval_sensitive_field")
