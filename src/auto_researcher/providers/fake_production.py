"""Production-shaped, deterministic provider used only by offline tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from auto_researcher.agents.models import (
    ModelCallConfig,
    StructuredModelResponse,
)
from auto_researcher.contracts.enums import ProviderErrorCode
from auto_researcher.providers.protocols import ProviderCallError
from auto_researcher.runtime.identity import payload_hash


@dataclass
class FakeProductionProviderControls:
    mode: Literal[
        "success",
        "confirmed_failure",
        "timeout",
        "uncertain",
        "malformed",
        "oversized",
    ] = "success"
    input_tokens: int = 100
    output_tokens: int = 50
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class FakeProductionStructuredModelClient:
    """No network, credentials, environment reads, retries, or hidden state."""

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        response: dict,
        controls: FakeProductionProviderControls | None = None,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.response = response
        self.controls = controls or FakeProductionProviderControls()
        self.invocation_count = 0

    def generate_structured(
        self,
        *,
        call_id: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        call_config: ModelCallConfig,
        context_hash: str,
    ) -> StructuredModelResponse:
        del system_prompt, user_prompt
        self.invocation_count += 1
        mode = self.controls.mode
        if mode == "confirmed_failure":
            raise ProviderCallError(
                ProviderErrorCode.PERMANENT_PROVIDER_ERROR, retryable=False
            )
        if mode in {"timeout", "uncertain"}:
            raise ProviderCallError(
                ProviderErrorCode.TIMEOUT
                if mode == "timeout"
                else ProviderErrorCode.TRANSIENT_PROVIDER_ERROR,
                retryable=False,
            )
        payload = dict(self.response)
        if mode == "malformed":
            payload = {"unexpected": True}
        elif mode == "oversized":
            payload["source"] = "x" * (call_config.maximum_output_tokens * 100)
        parsed = response_model.model_validate(payload)
        output = parsed.model_dump(mode="json")
        cost = call_config.pricing.estimate(
            input_tokens=self.controls.input_tokens,
            output_tokens=self.controls.output_tokens,
            cache_creation_input_tokens=self.controls.cache_creation_input_tokens,
            cache_read_input_tokens=self.controls.cache_read_input_tokens,
        )
        return StructuredModelResponse(
            call_id=call_id,
            provider=self.provider,
            model_id=self.model_id,
            structured_output=output,
            input_tokens=self.controls.input_tokens,
            output_tokens=self.controls.output_tokens,
            cache_creation_input_tokens=self.controls.cache_creation_input_tokens,
            cache_read_input_tokens=self.controls.cache_read_input_tokens,
            estimated_cost=cost,
            latency_ms=1,
            attempts=1,
            finish_reason="fake_complete",
            provider_request_id="fake-request-" + call_id[-12:],
            prompt_version=call_config.prompt_version,
            context_hash=context_hash,
            response_hash=payload_hash(output),
        )
