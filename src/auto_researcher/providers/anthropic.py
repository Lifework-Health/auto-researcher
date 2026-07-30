"""Optional Anthropic client factory; credentials remain environment-only."""

from __future__ import annotations

import os

from auto_researcher.agents.models import ModelCallConfig
from auto_researcher.providers.langchain_client import LangChainStructuredModelClient


def create_anthropic_client(
    config: ModelCallConfig,
) -> LangChainStructuredModelClient:
    if config.provider.casefold() != "anthropic":
        raise ValueError("Anthropic factory requires provider='anthropic'")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required for live Anthropic mode; "
            "the key is read from the environment only"
        )
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "Live Anthropic mode requires the agents-anthropic extra: "
            "pip install 'auto-researcher[agents-anthropic]'"
        ) from exc
    model = ChatAnthropic(
        model=config.model_id,
        temperature=config.temperature,
        max_tokens=config.maximum_output_tokens,
        timeout=config.timeout_seconds,
        max_retries=0,
    )
    return LangChainStructuredModelClient(
        model,
        provider=config.provider,
        model_id=config.model_id,
    )
