"""Durable, approval-bound production model bridge for OpenEvolve mutations."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from auto_researcher.agents.call_store import AgentCallStore, stable_record_id
from auto_researcher.agents.models import AgentCallRecord
from auto_researcher.contracts.enums import (
    AgentCallStatus,
    AgentRole,
    ProviderErrorCode,
)
from auto_researcher.providers.protocols import ProviderCallError, StructuredModelClient
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.live_models import (
    LiveMutationApprovalEnvelope,
    MetadataOnlyLiveMutationApproval,
    MetadataOnlyOpenEvolveModelCallContext,
    OPENEVOLVE_MUTATION_PROMPT_V1,
    OPENEVOLVE_MUTATION_PROMPT_V2,
    OpenEvolveModelBridgeContract,
    OpenEvolveModelCallContext,
    validate_approval,
    validate_metadata_only_approval,
)
from auto_researcher.search.openevolve.live_boundary import (
    MetadataOnlyMutationBoundary,
    assert_no_prohibited_dynamic_content,
    validate_metadata_only_request,
)
from auto_researcher.search.openevolve.upstream_models import (
    ModelBridgeReservation,
    MutationConstraints,
    UpstreamMutationEnvelope,
)

ProviderFactory = Callable[[], StructuredModelClient]


def mutation_input_hash(request: dict, prompt_version: str) -> str:
    payload: object = request
    version = "canonical-json-sha256-v1"
    if prompt_version == OPENEVOLVE_MUTATION_PROMPT_V2:
        payload = {"prompt_version": prompt_version, "request": request}
        version = "canonical-json-sha256-v2"
    return payload_hash(
        {
            "domain": "auto-researcher-openevolve-mutation-input",
            "version": version,
            "payload": payload,
        }
    )


class LiveMutationBridgeError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DurableOpenEvolveModelBridge:
    """At-most-once automatic dispatch and exactly-once completion reuse."""

    def __init__(
        self,
        *,
        contract: OpenEvolveModelBridgeContract,
        context: OpenEvolveModelCallContext | MetadataOnlyOpenEvolveModelCallContext,
        approval: LiveMutationApprovalEnvelope | None,
        store: AgentCallStore,
        provider_factory: ProviderFactory | None,
        now: Callable[[], datetime],
        system_prompt: str,
        metadata_only_boundary: MetadataOnlyMutationBoundary | None = None,
        crash_after_response: bool = False,
    ) -> None:
        self.contract = contract
        self.prompt_version = contract.prompt_version
        self.context = context
        self.approval = approval
        self.store = store
        self.provider_factory = provider_factory
        self.now = now
        self.system_prompt = system_prompt
        self.metadata_only_boundary = metadata_only_boundary
        self.crash_after_response = crash_after_response

    def complete(
        self, request: dict, mutation_reservation_id: str
    ) -> tuple[dict, ModelBridgeReservation]:
        if self.approval is None:
            raise LiveMutationBridgeError("live_mutation_approval_required")
        self._validate_request(request, mutation_reservation_id)
        parent = request.get("parent", {})
        call_context = self.context.model_copy(
            update={
                "generation": int(parent.get("generation", 0)) + 1,
                "parent_candidate_id": str(
                    parent.get("authoritative_candidate_id", parent.get("id", ""))
                ),
            }
        )
        self._validate_approval(call_context)
        input_hash = mutation_input_hash(request, self.contract.prompt_version)
        semantic_payload = {
            "domain": "auto-researcher-openevolve-model-call-semantic-key",
            "run_id": call_context.run_id,
            "thread_id": call_context.thread_id,
            "search_request_id": call_context.search_request_id,
            "generation": call_context.generation,
            "parent_candidate_id": call_context.parent_candidate_id,
            "component_id": call_context.component_id,
            "mutation_reservation_id": mutation_reservation_id,
        }
        if self.contract.prompt_version == OPENEVOLVE_MUTATION_PROMPT_V2:
            semantic_payload.update(
                {
                    "prompt_version": self.contract.prompt_version,
                    "input_payload_hash": input_hash,
                }
            )
        semantic_key = payload_hash(semantic_payload)
        identity_payload = {
            **call_context.model_dump(mode="json"),
            "bridge": self.contract.model_dump(mode="json", by_alias=True),
            "approval_id": self.approval.approval_id,
            "approval_hash": self.approval.approval_hash,
            "input_payload_hash": input_hash,
            "output_schema_version": self.contract.response_schema_version,
            "mutation_reservation_id": mutation_reservation_id,
        }
        call_id = "openevolve-call-" + payload_hash(
            {
                "domain": "auto-researcher-openevolve-model-call",
                "version": "canonical-json-sha256-v1",
                "payload": identity_payload,
            }
        )
        created = self.now().astimezone(UTC)
        config = self.contract.model_config_contract
        reserved = AgentCallRecord(
            record_id=stable_record_id(call_id, AgentCallStatus.RESERVED, 1),
            call_id=call_id,
            run_id=call_context.run_id,
            cycle=call_context.generation,
            role=AgentRole.OPENEVOLVE_MUTATION,
            provider=config.provider,
            model_id=config.model_id,
            prompt_name=self.contract.prompt_id,
            prompt_version=self.contract.prompt_version,
            prompt_hash=payload_hash(self.system_prompt),
            context_hash=payload_hash(call_context),
            response_schema_version=self.contract.response_schema_version,
            status=AgentCallStatus.RESERVED,
            pricing=config.pricing,
            pricing_version=config.pricing.version,
            pricing_currency=config.pricing.currency,
            created_at=created,
            semantic_key=semantic_key,
            approval_id=self.approval.approval_id,
            approval_hash=self.approval.approval_hash,
            budget_identity=call_context.model_budget_identity,
            input_payload_hash=input_hash,
            maximum_reserved_cost=config.maximum_cost_per_call,
            reservation_timestamp=created,
        )
        try:
            current, _ = self.store.reserve(
                reserved,
                maximum_calls=min(
                    self.approval.maximum_model_calls,
                    call_context.maximum_model_calls,
                ),
                maximum_total_cost=min(
                    self.approval.maximum_total_cost,
                    call_context.maximum_model_cost,
                ),
            )
        except ValueError as exc:
            raise LiveMutationBridgeError(str(exc)) from None
        if current.status == AgentCallStatus.COMPLETED:
            return self._reuse(current, request, mutation_reservation_id)
        if current.status == AgentCallStatus.DISPATCHING:
            raise LiveMutationBridgeError("model_call_already_dispatching")
        if current.status == AgentCallStatus.OUTCOME_UNKNOWN:
            raise LiveMutationBridgeError("model_call_outcome_unknown")
        if current.status != AgentCallStatus.RESERVED:
            raise LiveMutationBridgeError("model_call_identity_conflict")

        # Approval and budget are checked again immediately before ownership.
        self._validate_approval(call_context)
        dispatched_at = self.now().astimezone(UTC)
        dispatch = current.model_copy(
            update={
                "record_id": stable_record_id(call_id, AgentCallStatus.DISPATCHING, 2),
                "status": AgentCallStatus.DISPATCHING,
                "created_at": dispatched_at,
                "dispatch_timestamp": dispatched_at,
                "provider_request_started": True,
                "attempt_count": 1,
            }
        )
        if not self.store.transition(
            dispatch, expected_status=AgentCallStatus.RESERVED
        ):
            latest = self.store.latest(call_id)
            if latest is not None and latest.status == AgentCallStatus.COMPLETED:
                return self._reuse(latest, request, mutation_reservation_id)
            raise LiveMutationBridgeError("model_call_already_dispatching")
        if self.provider_factory is None:
            self._fail(
                dispatch,
                AgentCallStatus.FAILED_BEFORE_DISPATCH,
                ProviderErrorCode.PROVIDER_UNAVAILABLE_BEFORE_DISPATCH,
                ordinal=3,
            )
            raise LiveMutationBridgeError("model_call_provider_unavailable")
        try:
            client = self.provider_factory()
        except Exception:
            self._fail(
                dispatch,
                AgentCallStatus.FAILED_BEFORE_DISPATCH,
                ProviderErrorCode.PROVIDER_UNAVAILABLE_BEFORE_DISPATCH,
                ordinal=3,
            )
            raise LiveMutationBridgeError("model_call_provider_unavailable") from None
        try:
            response = client.generate_structured(
                call_id=call_id,
                system_prompt=self.system_prompt,
                user_prompt=json.dumps(request, sort_keys=True, separators=(",", ":")),
                response_model=UpstreamMutationEnvelope,
                call_config=config,
                context_hash=input_hash,
            )
        except ProviderCallError as exc:
            uncertain = exc.code in {
                ProviderErrorCode.TIMEOUT,
                ProviderErrorCode.TRANSIENT_PROVIDER_ERROR,
            }
            self._fail(
                dispatch,
                (
                    AgentCallStatus.OUTCOME_UNKNOWN
                    if uncertain
                    else AgentCallStatus.FAILED_CONFIRMED
                ),
                (
                    ProviderErrorCode.OUTCOME_UNKNOWN
                    if uncertain
                    else ProviderErrorCode.PROVIDER_CONFIRMED_FAILURE
                ),
                ordinal=3,
                cost=exc.estimated_cost,
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
            )
            raise LiveMutationBridgeError(
                "model_call_outcome_unknown"
                if uncertain
                else "model_call_provider_confirmed_failure"
            ) from None
        except Exception:
            self._fail(
                dispatch,
                AgentCallStatus.OUTCOME_UNKNOWN,
                ProviderErrorCode.OUTCOME_UNKNOWN,
                ordinal=3,
            )
            raise LiveMutationBridgeError("model_call_outcome_unknown") from None
        if self.crash_after_response:
            self._fail(
                dispatch,
                AgentCallStatus.OUTCOME_UNKNOWN,
                ProviderErrorCode.OUTCOME_UNKNOWN,
                ordinal=3,
                cost=response.estimated_cost,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            raise LiveMutationBridgeError("model_call_outcome_unknown")
        try:
            envelope = UpstreamMutationEnvelope.model_validate(
                response.structured_output
            )
            if (
                response.call_id != call_id
                or response.provider != config.provider
                or response.model_id != config.model_id
                or response.prompt_version != config.prompt_version
                or response.context_hash != input_hash
                or response.attempts != 1
            ):
                raise ValueError("model_call_response_invalid")
            self._validate_envelope(envelope, request)
            if response.estimated_cost > min(
                config.maximum_cost_per_call, self.approval.maximum_total_cost
            ):
                raise ValueError("model_call_cost_limit_exceeded")
            if response.input_tokens <= 0 or response.output_tokens <= 0:
                raise ValueError("model_call_response_invalid")
            if response.input_tokens > self.approval.maximum_input_tokens:
                raise ValueError("model_call_cost_limit_exceeded")
            if response.output_tokens > min(
                config.maximum_output_tokens,
                self.approval.maximum_output_tokens,
            ):
                raise ValueError("model_call_cost_limit_exceeded")
            if (
                response.cache_creation_input_tokens
                and config.pricing.cache_write_cost_per_million_tokens is None
            ) or (
                response.cache_read_input_tokens
                and config.pricing.cache_read_cost_per_million_tokens is None
            ):
                raise ValueError("model_call_response_invalid")
            calculated_cost = config.pricing.estimate(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cache_creation_input_tokens=response.cache_creation_input_tokens,
                cache_read_input_tokens=response.cache_read_input_tokens,
            )
            if abs(calculated_cost - response.estimated_cost) > 1e-12:
                raise ValueError("model_call_response_invalid")
        except Exception as exc:
            self._fail(
                dispatch,
                AgentCallStatus.FAILED_CONFIRMED,
                ProviderErrorCode.RESPONSE_INVALID,
                ordinal=3,
                cost=response.estimated_cost,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            code = (
                str(exc)
                if str(exc).startswith("model_call_")
                else "model_call_response_invalid"
            )
            raise LiveMutationBridgeError(code) from None
        completed_at = self.now().astimezone(UTC)
        output = envelope.model_dump(mode="json")
        output_hash = payload_hash(output)
        completion_identity = payload_hash(
            {
                "domain": "auto-researcher-openevolve-model-call-completion",
                "call_id": call_id,
                "input_hash": input_hash,
                "output_hash": output_hash,
                "approval_hash": self.approval.approval_hash,
                "provider": config.provider,
                "model_id": config.model_id,
            }
        )
        completed = dispatch.model_copy(
            update={
                "record_id": stable_record_id(call_id, AgentCallStatus.COMPLETED, 3),
                "status": AgentCallStatus.COMPLETED,
                "structured_output": output,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cache_creation_input_tokens": response.cache_creation_input_tokens,
                "cache_read_input_tokens": response.cache_read_input_tokens,
                "estimated_cost": response.estimated_cost,
                "latency_ms": response.latency_ms,
                "response_hash": output_hash,
                "completion_timestamp": completed_at,
                "created_at": completed_at,
                "completion_identity": completion_identity,
                "provider_request_id": self._safe_identifier(
                    response.provider_request_id
                ),
                "finish_reason": self._safe_identifier(response.finish_reason),
            }
        )
        try:
            persisted = self.store.transition(
                completed, expected_status=AgentCallStatus.DISPATCHING
            )
        except Exception:
            raise LiveMutationBridgeError(
                "model_call_completion_persistence_failed"
            ) from None
        if not persisted:
            raise LiveMutationBridgeError("model_call_completion_persistence_failed")
        return output, self._reservation(completed, mutation_reservation_id)

    def _reuse(
        self, record: AgentCallRecord, request: dict, mutation_reservation_id: str
    ) -> tuple[dict, ModelBridgeReservation]:
        assert self.approval is not None
        config = self.contract.model_config_contract
        parent = request["parent"]
        call_context = self.context.model_copy(
            update={
                "generation": int(parent["generation"]) + 1,
                "parent_candidate_id": str(
                    parent.get("authoritative_candidate_id", parent["id"])
                ),
            }
        )
        input_hash = mutation_input_hash(request, self.contract.prompt_version)
        output_hash = (
            payload_hash(record.structured_output)
            if record.structured_output is not None
            else None
        )
        expected_completion = payload_hash(
            {
                "domain": "auto-researcher-openevolve-model-call-completion",
                "call_id": record.call_id,
                "input_hash": input_hash,
                "output_hash": output_hash,
                "approval_hash": self.approval.approval_hash,
                "provider": config.provider,
                "model_id": config.model_id,
            }
        )
        calculated_cost = config.pricing.estimate(
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cache_creation_input_tokens=record.cache_creation_input_tokens,
            cache_read_input_tokens=record.cache_read_input_tokens,
        )
        if (
            record.role != AgentRole.OPENEVOLVE_MUTATION
            or record.run_id != self.context.run_id
            or record.approval_id != self.approval.approval_id
            or record.approval_hash != self.approval.approval_hash
            or record.provider != config.provider
            or record.model_id != config.model_id
            or record.prompt_name != self.contract.prompt_id
            or record.prompt_version != self.contract.prompt_version
            or record.prompt_hash != payload_hash(self.system_prompt)
            or record.context_hash != payload_hash(call_context)
            or record.response_schema_version != self.contract.response_schema_version
            or record.budget_identity != call_context.model_budget_identity
            or record.pricing != config.pricing
            or record.input_payload_hash != input_hash
            or record.structured_output is None
            or output_hash != record.response_hash
            or record.completion_identity != expected_completion
            or record.completion_timestamp is None
            or record.input_tokens <= 0
            or record.output_tokens <= 0
            or abs(calculated_cost - record.estimated_cost) > 1e-12
            or record.estimated_cost > config.maximum_cost_per_call
        ):
            raise LiveMutationBridgeError("model_call_completed_response_corrupt")
        envelope = UpstreamMutationEnvelope.model_validate(record.structured_output)
        self._validate_envelope(envelope, request)
        return record.structured_output, self._reservation(
            record, mutation_reservation_id
        )

    def _reservation(
        self, record: AgentCallRecord, mutation_reservation_id: str
    ) -> ModelBridgeReservation:
        return ModelBridgeReservation(
            reservation_id=record.call_id,
            mutation_reservation_id=mutation_reservation_id,
            provider=record.provider,
            model_id=record.model_id,
            prompt_version=record.prompt_version,
            maximum_output_bytes=self.contract.maximum_input_bytes,
            completed=True,
            response_hash=record.response_hash,
        )

    def _fail(
        self,
        current: AgentCallRecord,
        status: AgentCallStatus,
        code: ProviderErrorCode,
        *,
        ordinal: int,
        cost: float = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        failed = current.model_copy(
            update={
                "record_id": stable_record_id(current.call_id, status, ordinal),
                "status": status,
                "error_code": code,
                "created_at": self.now().astimezone(UTC),
                "estimated_cost": cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "provider_request_started": status
                != AgentCallStatus.FAILED_BEFORE_DISPATCH,
                "attempt_count": 0
                if status == AgentCallStatus.FAILED_BEFORE_DISPATCH
                else current.attempt_count,
            }
        )
        try:
            persisted = self.store.transition(
                failed, expected_status=AgentCallStatus.DISPATCHING
            )
        except Exception:
            raise LiveMutationBridgeError(
                "model_call_completion_persistence_failed"
            ) from None
        if not persisted:
            raise LiveMutationBridgeError("model_call_completion_persistence_failed")

    def _validate_request(self, request: dict, mutation_reservation_id: str) -> None:
        assert self.approval is not None
        try:
            encoded = json.dumps(
                request, sort_keys=True, separators=(",", ":")
            ).encode()
        except (TypeError, ValueError):
            raise LiveMutationBridgeError("model_call_input_invalid") from None
        if len(encoded) > self.contract.maximum_input_bytes:
            raise LiveMutationBridgeError("model_call_input_oversize")
        # Conservative UTF-8 preflight: never assume more than four bytes/token.
        if len(encoded) > self.approval.maximum_input_tokens * 4:
            raise LiveMutationBridgeError("model_call_input_oversize")
        if not mutation_reservation_id:
            raise LiveMutationBridgeError("model_call_identity_conflict")
        if self.contract.prompt_version == OPENEVOLVE_MUTATION_PROMPT_V1:
            if (
                request.get("protocol") != "upstream-adapter-mutation-request-v1"
                or "mutation_constraints" in request
            ):
                raise LiveMutationBridgeError("model_call_input_invalid")
        else:
            if request.get("protocol") != "upstream-adapter-mutation-request-v2":
                raise LiveMutationBridgeError("model_call_input_invalid")
            try:
                constraints = MutationConstraints.model_validate(
                    request.get("mutation_constraints")
                )
            except (TypeError, ValueError):
                raise LiveMutationBridgeError("model_call_input_invalid") from None
            if (
                constraints.mutable_file != self.context.mutable_file
                or request.get("mutable_file") != constraints.mutable_file
                or request.get("interface_contract")
                != constraints.immutable_interface_contract
                or request.get("maximum_source_bytes")
                != constraints.maximum_source_bytes
            ):
                raise LiveMutationBridgeError("model_call_input_invalid")
            if isinstance(self.approval, MetadataOnlyLiveMutationApproval):
                if not isinstance(self.context, MetadataOnlyOpenEvolveModelCallContext):
                    raise LiveMutationBridgeError("live_mutation_approval_mismatch")
                try:
                    validate_metadata_only_request(
                        request,
                        expected_exposure_identity=self.context.model_exposure_identity,
                    )
                except (TypeError, ValueError):
                    raise LiveMutationBridgeError(
                        "metadata_only_model_input_rejected"
                    ) from None
        parent = request.get("parent")
        if (
            not isinstance(parent, dict)
            or not isinstance(parent.get("id"), str)
            or not parent["id"]
            or not isinstance(parent.get("generation"), int)
            or parent["generation"] < 0
            or (
                "authoritative_candidate_id" in parent
                and (
                    not isinstance(parent["authoritative_candidate_id"], str)
                    or not parent["authoritative_candidate_id"]
                )
            )
        ):
            raise LiveMutationBridgeError("model_call_identity_conflict")
        forbidden = {
            "provider_configuration",
            "api_key",
            "credentials",
            "retry",
            "executor_digest",
            "shell_command",
            "additional_files",
        }

        def keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield str(key)
                    yield from keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)

        if forbidden.intersection(keys(request)):
            raise LiveMutationBridgeError("upstream_direct_provider_forbidden")

    def _validate_envelope(
        self, envelope: UpstreamMutationEnvelope, request: dict
    ) -> None:
        if (
            envelope.mutable_file != self.context.mutable_file
            or envelope.dependency_requests
            or envelope.provider_configuration
            or len(envelope.source.encode())
            > int(request.get("maximum_source_bytes", 0))
        ):
            raise ValueError("model_call_response_invalid")
        response_text = envelope.model_dump_json()
        if re.search(
            r"(?i)(api[_-]?key|bearer\s+|password|provider[_-]?secret|"
            r"/users/|/home/|/private/|subprocess\.|os\.system|shell=true)",
            response_text,
        ):
            raise ValueError("model_call_response_unsafe")
        if isinstance(self.approval, MetadataOnlyLiveMutationApproval):
            try:
                assert_no_prohibited_dynamic_content(
                    {
                        "source": envelope.source,
                        "description": envelope.description,
                        "upstream_program_id": envelope.upstream_program_id,
                    }
                )
            except ValueError:
                raise ValueError("model_call_response_unsafe") from None

    @staticmethod
    def _safe_identifier(value: str | None) -> str | None:
        if value is None or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value):
            return None
        return value

    def _validate_approval(
        self,
        context: OpenEvolveModelCallContext | MetadataOnlyOpenEvolveModelCallContext,
    ) -> None:
        assert self.approval is not None
        try:
            if isinstance(self.approval, MetadataOnlyLiveMutationApproval):
                if (
                    not isinstance(context, MetadataOnlyOpenEvolveModelCallContext)
                    or self.metadata_only_boundary is None
                    or self.approval.prompt_hash != payload_hash(self.system_prompt)
                ):
                    raise ValueError("live_mutation_approval_mismatch")
                validate_metadata_only_approval(
                    self.approval,
                    context,
                    self.contract,
                    self.metadata_only_boundary,
                    now=self.now(),
                )
            else:
                if not isinstance(context, OpenEvolveModelCallContext):
                    raise ValueError("live_mutation_approval_mismatch")
                validate_approval(self.approval, context, self.contract, now=self.now())
        except ValueError as exc:
            raise LiveMutationBridgeError(str(exc)) from None


def load_mutation_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("model_call_prompt_not_approved")
    return prompt
