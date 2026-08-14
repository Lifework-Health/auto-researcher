"""Optional Anthropic client factory with runtime-only secret resolution."""

from __future__ import annotations

from importlib import import_module

from pydantic import SecretStr

from auto_researcher.agents.models import ModelCallConfig
from auto_researcher.providers.langchain_client import LangChainStructuredModelClient
from auto_researcher.secrets import (
    ResolvedSecret,
    SecretProviderKind,
    SecretReference,
    SecretResolutionError,
    SecretResolutionErrorCode,
    provider_for_reference,
)


ANTHROPIC_ENVIRONMENT_SECRET = SecretReference(
    logical_name="anthropic_api_key",
    provider=SecretProviderKind.ENVIRONMENT,
    provider_identifier="ANTHROPIC_API_KEY",
)


def create_anthropic_client(
    config: ModelCallConfig,
    *,
    credential: ResolvedSecret | None = None,
) -> LangChainStructuredModelClient:
    if config.provider.casefold() != "anthropic":
        raise ValueError("Anthropic factory requires provider='anthropic'")
    if credential is None:
        resolver = provider_for_reference(ANTHROPIC_ENVIRONMENT_SECRET)
        credential = resolver.resolve(ANTHROPIC_ENVIRONMENT_SECRET)
        if credential is None:
            raise SecretResolutionError(
                SecretResolutionErrorCode.MISSING,
                ANTHROPIC_ENVIRONMENT_SECRET,
            ) from None
    try:
        ChatAnthropic = import_module("langchain_anthropic").ChatAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "Live Anthropic mode requires the agents-anthropic extra: "
            "pip install 'auto-researcher[agents-anthropic]'"
        ) from exc
    initialisation_failed = False
    try:
        model = ChatAnthropic(
            api_key=SecretStr(credential.reveal()),
            model=config.model_id,
            temperature=config.temperature,
            max_tokens=config.maximum_output_tokens,
            timeout=config.timeout_seconds,
            max_retries=0,
        )
    except Exception:
        initialisation_failed = True
        model = None
    if initialisation_failed:
        raise RuntimeError("Anthropic client initialisation failed") from None
    return LangChainStructuredModelClient(
        model,
        provider=config.provider,
        model_id=config.model_id,
    )
