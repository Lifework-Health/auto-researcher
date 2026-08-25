"""Shared replay-safe execution for one bounded structured model call."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from auto_researcher.agents.call_store import AgentCallStore, stable_record_id
from auto_researcher.agents.models import (
    AgentBudgetPolicy,
    AgentCallRecord,
    AgentCallTelemetry,
    ModelCallConfig,
    json_schema_version,
)
from auto_researcher.agents.prompts import PromptBundle
from auto_researcher.agents.reconciliation import ReconciliationError
from auto_researcher.contracts.enums import (
    AgentCallStatus,
    AgentRole,
    ProviderErrorCode,
)
from auto_researcher.providers.protocols import (
    ProviderCallError,
    StructuredModelClient,
)

ProposalT = TypeVar("ProposalT", bound=BaseModel)
ResultT = TypeVar("ResultT")


class LiveAgentExecutionError(RuntimeError):
    def __init__(
        self,
        code: str,
        telemetry: AgentCallTelemetry | None = None,
    ) -> None:
        self.code = code
        self.telemetry = telemetry
        super().__init__(code)


def deterministic_call_id(
    *,
    run_id: str,
    cycle: int,
    role: AgentRole,
    prompt_version: str,
    context_hash: str,
    schema_version: str,
    provider: str,
    model_id: str,
) -> str:
    digest = hashlib.sha256(
        "\x1f".join(
            (
                run_id,
                str(cycle),
                role.value,
                prompt_version,
                context_hash,
                schema_version,
                provider,
                model_id,
            )
        ).encode()
    ).hexdigest()[:24]
    return f"model-call-{digest}"


class BoundedStructuredCall:
    def __init__(
        self,
        *,
        client: StructuredModelClient,
        config: ModelCallConfig,
        budget_policy: AgentBudgetPolicy,
        store: AgentCallStore,
        clock: Callable[[], datetime],
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.config = config
        self.budget_policy = budget_policy
        self.store = store
        self.clock = clock
        self.sleeper = sleeper

    def run(
        self,
        *,
        run_id: str,
        cycle: int,
        role: AgentRole,
        context_hash: str,
        context_json: str,
        remaining_cost_budget: float,
        model_calls_used: int,
        prompt: PromptBundle,
        response_model: type[ProposalT],
        reconcile: Callable[[ProposalT, str], ResultT],
    ) -> tuple[ResultT, AgentCallTelemetry]:
        if prompt.version != self.config.prompt_version:
            raise LiveAgentExecutionError("prompt_version_configuration_mismatch")
        if len(context_json) > self.budget_policy.maximum_input_context_size:
            raise LiveAgentExecutionError("agent_context_too_large")
        output_limit = (
            self.budget_policy.maximum_research_director_output_tokens
            if role == AgentRole.RESEARCH_DIRECTOR
            else self.budget_policy.maximum_output_tokens
        )
        cost_limit = (
            self.budget_policy.maximum_research_director_cost_per_call
            if role == AgentRole.RESEARCH_DIRECTOR
            else self.budget_policy.maximum_cost_per_call
        )
        if self.config.maximum_output_tokens > output_limit:
            raise LiveAgentExecutionError("maximum_output_tokens_exceeds_agent_policy")
        if self.config.maximum_cost_per_call > cost_limit:
            raise LiveAgentExecutionError("maximum_call_cost_exceeds_agent_policy")
        if remaining_cost_budget < self.config.maximum_cost_per_call:
            raise LiveAgentExecutionError("insufficient_remaining_cost_budget")
        attempts_allowed = min(
            self.config.maximum_attempts,
            self.budget_policy.maximum_attempts_per_agent_call,
            self.budget_policy.maximum_total_model_calls - model_calls_used,
        )
        if attempts_allowed < 1:
            raise LiveAgentExecutionError("maximum_total_model_calls_reached")

        schema_version = json_schema_version(response_model)
        base_call_id = deterministic_call_id(
            run_id=run_id,
            cycle=cycle,
            role=role,
            prompt_version=prompt.version,
            context_hash=context_hash,
            schema_version=schema_version,
            provider=self.config.provider,
            model_id=self.config.model_id,
        )
        role_limit = {
            AgentRole.HYPOTHESIS: self.budget_policy.maximum_hypothesis_calls_per_cycle,
            AgentRole.PLANNER: self.budget_policy.maximum_planner_calls_per_cycle,
            AgentRole.RESEARCH_DIRECTOR: (
                self.budget_policy.maximum_research_director_calls_per_cycle
            ),
        }.get(role, self.budget_policy.maximum_planner_calls_per_cycle)
        existing_role_calls = {
            record.call_id
            for record in self.store.list_records(run_id)
            if record.cycle == cycle
            and record.role == role
            and record.retry_of_call_id is None
        }
        if role == AgentRole.RESEARCH_DIRECTOR:
            total_director_calls = {
                record.call_id
                for record in self.store.list_records(run_id)
                if record.role == role and record.retry_of_call_id is None
            }
            if (
                base_call_id not in total_director_calls
                and len(total_director_calls)
                >= self.budget_policy.maximum_research_director_calls_total
            ):
                raise LiveAgentExecutionError("maximum_research_director_calls_reached")
        if (
            base_call_id not in existing_role_calls
            and len(existing_role_calls) >= role_limit
        ):
            raise LiveAgentExecutionError("maximum_agent_calls_per_cycle_reached")
        completed = self._completed_record(base_call_id)
        selected_call_id = (
            base_call_id
            if completed is not None
            else self._select_call_id(base_call_id)
        )
        completed = completed or self._completed_record(selected_call_id)
        if completed is not None:
            assert completed.structured_output is not None
            try:
                proposal = response_model.model_validate(completed.structured_output)
                result = reconcile(proposal, selected_call_id)
            except (ValidationError, ReconciliationError) as exc:
                raise LiveAgentExecutionError(
                    "completed_call_reconciliation_conflict"
                ) from exc
            return result, _telemetry_from_record(
                completed,
                replayed=True,
                maximum_cost_per_call=self.config.maximum_cost_per_call,
            )

        reservation = self._record(
            call_id=selected_call_id,
            base_call_id=base_call_id,
            run_id=run_id,
            cycle=cycle,
            role=role,
            context_hash=context_hash,
            prompt=prompt,
            schema_version=schema_version,
            status=AgentCallStatus.RESERVED,
            provider_request_started=True,
        )
        self.store.append(reservation)
        correction = ""
        total_input = total_output = total_cache_create = total_cache_read = 0
        total_cost = 0.0
        total_latency = 0
        last_error = ProviderErrorCode.PERMANENT_PROVIDER_ERROR
        last_response = None
        attempts_made = 0
        for attempt in range(1, attempts_allowed + 1):
            attempts_made = attempt
            system_prompt, user_prompt = prompt.render(
                context_json=context_json,
                correction=correction,
            )
            try:
                response = self.client.generate_structured(
                    call_id=selected_call_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                    call_config=self.config,
                    context_hash=context_hash,
                )
                if (
                    response.call_id != selected_call_id
                    or response.provider != self.config.provider
                    or response.model_id != self.config.model_id
                    or response.prompt_version != self.config.prompt_version
                    or response.context_hash != context_hash
                    or response.attempts != 1
                ):
                    raise ProviderCallError(
                        ProviderErrorCode.PERMANENT_PROVIDER_ERROR,
                        retryable=False,
                    )
                last_response = response
                total_input += response.input_tokens
                total_output += response.output_tokens
                total_cache_create += response.cache_creation_input_tokens
                total_cache_read += response.cache_read_input_tokens
                total_cost += self.config.pricing.estimate(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cache_creation_input_tokens=(response.cache_creation_input_tokens),
                    cache_read_input_tokens=response.cache_read_input_tokens,
                )
                total_latency += response.latency_ms
                if response.output_tokens > self.config.maximum_output_tokens:
                    last_error = ProviderErrorCode.PERMANENT_PROVIDER_ERROR
                    break
                proposal = response_model.model_validate(response.structured_output)
                result = reconcile(proposal, selected_call_id)
            except (ValidationError, ReconciliationError) as exc:
                last_error = ProviderErrorCode.INVALID_STRUCTURED_OUTPUT
                safe_reason = (
                    exc.code
                    if isinstance(exc, ReconciliationError)
                    else "invalid_structured_output"
                )
                print(
                    "AUTO_RESEARCHER_AGENT_RETRY "
                    f"role={role.value} attempt={attempt} reason={safe_reason}",
                    flush=True,
                )
                correction = (
                    "Correction required: "
                    + safe_reason
                    + ". Return a corrected structured proposal."
                )
                if (
                    attempt < attempts_allowed
                    and total_cost < self.config.maximum_cost_per_call
                ):
                    self._backoff(attempt)
                    continue
                break
            except ProviderCallError as exc:
                last_error = exc.code
                total_input += exc.input_tokens
                total_output += exc.output_tokens
                total_cache_create += exc.cache_creation_input_tokens
                total_cache_read += exc.cache_read_input_tokens
                total_cost += exc.estimated_cost
                total_latency += exc.latency_ms
                if (
                    exc.retryable
                    and exc.code
                    in {
                        ProviderErrorCode.TIMEOUT,
                        ProviderErrorCode.TRANSIENT_PROVIDER_ERROR,
                        ProviderErrorCode.INVALID_STRUCTURED_OUTPUT,
                    }
                    and attempt < attempts_allowed
                    and total_cost < self.config.maximum_cost_per_call
                ):
                    print(
                        "AUTO_RESEARCHER_AGENT_RETRY "
                        f"role={role.value} attempt={attempt} "
                        f"reason={exc.code.value}",
                        flush=True,
                    )
                    correction = (
                        f"Retry after safe provider error {exc.code.value}; "
                        "return the requested structured proposal."
                    )
                    self._backoff(attempt)
                    continue
                break
            completed = self._record(
                call_id=selected_call_id,
                base_call_id=base_call_id,
                run_id=run_id,
                cycle=cycle,
                role=role,
                context_hash=context_hash,
                prompt=prompt,
                schema_version=schema_version,
                status=AgentCallStatus.COMPLETED,
                structured_output=proposal.model_dump(mode="json"),
                input_tokens=total_input,
                output_tokens=total_output,
                cache_creation_input_tokens=total_cache_create,
                cache_read_input_tokens=total_cache_read,
                estimated_cost=total_cost,
                latency_ms=total_latency,
                provider_request_id=response.provider_request_id,
                attempt_count=attempt,
                response_hash=response.response_hash,
                provider_request_started=True,
            )
            self.store.append(completed)
            return result, _telemetry_from_record(
                completed,
                maximum_cost_per_call=self.config.maximum_cost_per_call,
            )

        failed = self._record(
            call_id=selected_call_id,
            base_call_id=base_call_id,
            run_id=run_id,
            cycle=cycle,
            role=role,
            context_hash=context_hash,
            prompt=prompt,
            schema_version=schema_version,
            status=AgentCallStatus.FAILED,
            input_tokens=total_input,
            output_tokens=total_output,
            cache_creation_input_tokens=total_cache_create,
            cache_read_input_tokens=total_cache_read,
            estimated_cost=total_cost,
            latency_ms=total_latency,
            provider_request_id=(
                last_response.provider_request_id if last_response else None
            ),
            attempt_count=attempts_made,
            error_code=last_error,
            response_hash=last_response.response_hash if last_response else None,
            provider_request_started=True,
        )
        self.store.append(failed)
        telemetry = _telemetry_from_record(
            failed,
            maximum_cost_per_call=self.config.maximum_cost_per_call,
        )
        raise LiveAgentExecutionError(last_error.value, telemetry)

    def _backoff(self, failed_attempt: int) -> None:
        self.sleeper(min(0.25, 0.05 * (2 ** (failed_attempt - 1))))

    def _select_call_id(self, base_call_id: str) -> str:
        latest = self.store.latest(base_call_id)
        if latest is None:
            return base_call_id
        if latest.status == AgentCallStatus.COMPLETED:
            return base_call_id
        if (
            latest.status == AgentCallStatus.RESERVED
            and latest.provider_request_started
        ):
            indeterminate = latest.model_copy(
                update={
                    "record_id": stable_record_id(
                        base_call_id,
                        AgentCallStatus.INDETERMINATE,
                        len(self.store.records_for_call(base_call_id)) + 1,
                    ),
                    "status": AgentCallStatus.INDETERMINATE,
                    "created_at": self.clock(),
                    "error_code": None,
                }
            )
            self.store.append(indeterminate)
            raise LiveAgentExecutionError("model_call_indeterminate")
        if latest.status == AgentCallStatus.FAILED:
            raise LiveAgentExecutionError("model_call_previously_failed")
        if latest.status == AgentCallStatus.INDETERMINATE:
            return self._resolve_retry(base_call_id, latest.run_id)
        return base_call_id

    def _resolve_retry(self, parent_call_id: str, run_id: str) -> str:
        children = []
        seen_call_ids: set[str] = set()
        for item in self.store.list_records(run_id):
            if (
                item.retry_of_call_id == parent_call_id
                and item.call_id not in seen_call_ids
            ):
                child_latest = self.store.latest(item.call_id)
                if child_latest is not None:
                    children.append(child_latest)
                    seen_call_ids.add(item.call_id)
        if not children:
            raise LiveAgentExecutionError("model_call_indeterminate")
        completed_children = [
            child
            for child in children
            if self._completed_record(child.call_id) is not None
        ]
        if len(completed_children) > 1:
            raise LiveAgentExecutionError("conflicting_completed_model_calls")
        if completed_children:
            return completed_children[0].call_id
        started_children = [
            child
            for child in children
            if child.status == AgentCallStatus.RESERVED
            and child.provider_request_started
        ]
        if started_children:
            if len(started_children) > 1:
                raise LiveAgentExecutionError("multiple_started_model_call_retries")
            child = started_children[0]
            indeterminate = child.model_copy(
                update={
                    "record_id": stable_record_id(
                        child.call_id,
                        AgentCallStatus.INDETERMINATE,
                        len(self.store.records_for_call(child.call_id)) + 1,
                    ),
                    "status": AgentCallStatus.INDETERMINATE,
                    "created_at": self.clock(),
                }
            )
            self.store.append(indeterminate)
            raise LiveAgentExecutionError("model_call_indeterminate")
        unused_children = [
            child
            for child in children
            if child.status == AgentCallStatus.RESERVED
            and not child.provider_request_started
        ]
        if len(unused_children) > 1:
            raise LiveAgentExecutionError("multiple_authorized_model_call_retries")
        if unused_children:
            return unused_children[0].call_id
        indeterminate_children = [
            child for child in children if child.status == AgentCallStatus.INDETERMINATE
        ]
        if len(indeterminate_children) > 1:
            raise LiveAgentExecutionError("multiple_indeterminate_model_call_retries")
        if indeterminate_children:
            return self._resolve_retry(indeterminate_children[0].call_id, run_id)
        raise LiveAgentExecutionError("model_call_previously_failed")

    def _completed_record(self, call_id: str) -> AgentCallRecord | None:
        completed = [
            record
            for record in self.store.records_for_call(call_id)
            if record.status == AgentCallStatus.COMPLETED
        ]
        if not completed:
            return None
        fingerprints = {
            json.dumps(
                {
                    "structured_output": record.structured_output,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "cache_creation_input_tokens": (record.cache_creation_input_tokens),
                    "cache_read_input_tokens": record.cache_read_input_tokens,
                    "estimated_cost": record.estimated_cost,
                    "response_hash": record.response_hash,
                    "provider_request_id": record.provider_request_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in completed
        }
        if len(fingerprints) != 1:
            raise LiveAgentExecutionError("conflicting_completed_model_calls")
        return completed[0]

    def _record(
        self,
        *,
        call_id: str,
        base_call_id: str,
        run_id: str,
        cycle: int,
        role: AgentRole,
        context_hash: str,
        prompt: PromptBundle,
        schema_version: str,
        status: AgentCallStatus,
        structured_output: dict | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        estimated_cost: float = 0,
        latency_ms: int = 0,
        provider_request_id: str | None = None,
        attempt_count: int = 0,
        error_code: ProviderErrorCode | None = None,
        response_hash: str | None = None,
        provider_request_started: bool = False,
    ) -> AgentCallRecord:
        ordinal = len(self.store.records_for_call(call_id)) + 1
        retry_of = base_call_id if call_id != base_call_id else None
        return AgentCallRecord(
            record_id=stable_record_id(call_id, status, ordinal),
            call_id=call_id,
            run_id=run_id,
            cycle=cycle,
            role=role,
            provider=self.config.provider,
            model_id=self.config.model_id,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            prompt_hash=prompt.prompt_hash,
            context_hash=context_hash,
            response_schema_version=schema_version,
            status=status,
            structured_output=structured_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            estimated_cost=estimated_cost,
            pricing=self.config.pricing,
            pricing_version=self.config.pricing.version,
            pricing_currency=self.config.pricing.currency,
            latency_ms=latency_ms,
            provider_request_id=provider_request_id,
            attempt_count=attempt_count,
            created_at=self.clock(),
            error_code=error_code,
            response_hash=response_hash,
            retry_of_call_id=retry_of,
            provider_request_started=provider_request_started,
        )


def _telemetry_from_record(
    record: AgentCallRecord,
    *,
    replayed: bool = False,
    maximum_cost_per_call: float | None = None,
) -> AgentCallTelemetry:
    return AgentCallTelemetry(
        call_id=record.call_id,
        role=record.role,
        provider=record.provider,
        model_id=record.model_id,
        prompt_version=record.prompt_version,
        context_hash=record.context_hash,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cache_creation_input_tokens=record.cache_creation_input_tokens,
        cache_read_input_tokens=record.cache_read_input_tokens,
        estimated_cost=record.estimated_cost,
        provider_attempts=record.attempt_count,
        replayed=replayed,
        failed=record.status != AgentCallStatus.COMPLETED,
        cost_limit_exceeded=(
            maximum_cost_per_call is not None
            and record.estimated_cost > maximum_cost_per_call
        ),
        error_code=record.error_code,
    )
