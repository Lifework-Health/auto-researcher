"""Provider-neutral structured model boundary."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from auto_researcher.agents.models import ModelCallConfig, StructuredModelResponse
from auto_researcher.contracts.enums import ProviderErrorCode


class ProviderCallError(RuntimeError):
    def __init__(
        self,
        code: ProviderErrorCode,
        *,
        retryable: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        estimated_cost: float = 0,
        latency_ms: int = 0,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.estimated_cost = estimated_cost
        self.latency_ms = latency_ms
        super().__init__(code.value)


class StructuredModelClient(Protocol):
    provider: str
    model_id: str

    def generate_structured(
        self,
        *,
        call_id: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        call_config: ModelCallConfig,
        context_hash: str,
    ) -> StructuredModelResponse: ...
