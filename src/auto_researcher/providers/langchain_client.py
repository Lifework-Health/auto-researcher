"""LangChain BaseChatModel adapter using typed structured output."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from auto_researcher.agents.models import ModelCallConfig, StructuredModelResponse
from auto_researcher.contracts.enums import ProviderErrorCode
from auto_researcher.providers.protocols import ProviderCallError


def _usage_from_raw(raw: Any) -> tuple[int, int, int, int, dict[str, Any]]:
    usage = getattr(raw, "usage_metadata", None) or {}
    response_metadata = getattr(raw, "response_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    input_details = usage.get("input_token_details", {}) or {}
    cache_creation = int(
        input_details.get("cache_creation", 0)
        or response_metadata.get("cache_creation_input_tokens", 0)
        or 0
    )
    cache_read = int(
        input_details.get("cache_read", 0)
        or input_details.get("cache", 0)
        or response_metadata.get("cache_read_input_tokens", 0)
        or 0
    )
    return (
        input_tokens,
        output_tokens,
        cache_creation,
        cache_read,
        response_metadata,
    )


def _classify_exception(exc: Exception) -> ProviderCallError:
    name = type(exc).__name__.casefold()
    text = str(exc).casefold()
    if "auth" in name or "authentication" in text or "api key" in text:
        return ProviderCallError(ProviderErrorCode.AUTHENTICATION_ERROR, retryable=False)
    if "rate" in name or "rate limit" in text or "429" in text:
        return ProviderCallError(ProviderErrorCode.RATE_LIMITED, retryable=True)
    if "timeout" in name or "timed out" in text:
        return ProviderCallError(ProviderErrorCode.TIMEOUT, retryable=True)
    if "context" in text and ("long" in text or "large" in text):
        return ProviderCallError(ProviderErrorCode.CONTEXT_TOO_LARGE, retryable=False)
    if any(token in name for token in ("connection", "serviceunavailable", "internalserver")):
        return ProviderCallError(
            ProviderErrorCode.TRANSIENT_PROVIDER_ERROR,
            retryable=True,
        )
    return ProviderCallError(
        ProviderErrorCode.PERMANENT_PROVIDER_ERROR,
        retryable=False,
    )


class LangChainStructuredModelClient:
    def __init__(self, model: Any, *, provider: str, model_id: str) -> None:
        self._model = model
        self.provider = provider
        self.model_id = model_id

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
        if (call_config.provider, call_config.model_id) != (
            self.provider,
            self.model_id,
        ):
            raise ProviderCallError(
                ProviderErrorCode.PERMANENT_PROVIDER_ERROR,
                retryable=False,
            )
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            structured = self._model.with_structured_output(
                response_model,
                include_raw=True,
            )
            started = time.monotonic()
            result = structured.invoke(
                [SystemMessage(system_prompt), HumanMessage(user_prompt)],
                config={"run_name": call_id},
            )
            latency_ms = int((time.monotonic() - started) * 1000)
        except Exception as exc:
            raise _classify_exception(exc) from None
        parsed = result.get("parsed") if isinstance(result, dict) else None
        parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
        raw = result.get("raw") if isinstance(result, dict) else None
        (
            input_tokens,
            output_tokens,
            cache_creation,
            cache_read,
            response_metadata,
        ) = _usage_from_raw(raw)
        estimated_cost = call_config.pricing.estimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )
        if parsing_error is not None or parsed is None:
            raise ProviderCallError(
                ProviderErrorCode.INVALID_STRUCTURED_OUTPUT,
                retryable=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                estimated_cost=estimated_cost,
                latency_ms=latency_ms,
            )
        try:
            parsed_model = (
                parsed
                if isinstance(parsed, response_model)
                else response_model.model_validate(parsed)
            )
        except ValidationError:
            raise ProviderCallError(
                ProviderErrorCode.INVALID_STRUCTURED_OUTPUT,
                retryable=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                estimated_cost=estimated_cost,
                latency_ms=latency_ms,
            ) from None
        output = parsed_model.model_dump(mode="json")
        encoded = json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
        return StructuredModelResponse(
            call_id=call_id,
            provider=self.provider,
            model_id=self.model_id,
            structured_output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            estimated_cost=estimated_cost,
            latency_ms=latency_ms,
            attempts=1,
            finish_reason=response_metadata.get("stop_reason"),
            provider_request_id=(
                getattr(raw, "id", None) or response_metadata.get("id")
            ),
            prompt_version=call_config.prompt_version,
            context_hash=context_hash,
            response_hash=hashlib.sha256(encoded).hexdigest(),
        )
