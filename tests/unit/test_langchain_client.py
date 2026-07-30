from __future__ import annotations

from types import SimpleNamespace

from auto_researcher.agents.models import HypothesisProposal
from auto_researcher.providers.langchain_client import (
    LangChainStructuredModelClient,
)
from auto_researcher.providers.protocols import ProviderCallError
from auto_researcher.contracts.enums import ProviderErrorCode
import pytest
from tests.integration.test_live_agents import _call_config


class FakeRunnable:
    def __init__(self, response_model):
        self.response_model = response_model

    def invoke(self, messages, config):
        return {
            "parsed": self.response_model(
                statement="A bounded parameter may change the objective.",
                rationale="Contract test.",
                predicted_subspace={"complexity": [3, 6]},
                expected_observation="objective_score changes",
                falsification_condition="objective_score does not change",
                confidence=0.4,
            ),
            "parsing_error": None,
            "raw": SimpleNamespace(
                id="provider-request-1",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "input_token_details": {
                        "cache_creation": 10,
                        "cache_read": 20,
                    },
                },
                response_metadata={"stop_reason": "end_turn"},
            ),
        }


class FakeChatModel:
    def with_structured_output(self, response_model, *, include_raw):
        assert include_raw is True
        return FakeRunnable(response_model)


def test_langchain_client_uses_pydantic_structured_output_and_usage():
    config = _call_config()
    response = LangChainStructuredModelClient(
        FakeChatModel(),
        provider=config.provider,
        model_id=config.model_id,
    ).generate_structured(
        call_id="call-1",
        system_prompt="system",
        user_prompt="user",
        response_model=HypothesisProposal,
        call_config=config,
        context_hash="context-hash",
    )
    assert response.structured_output["confidence"] == 0.4
    assert response.input_tokens == 100
    assert response.cache_creation_input_tokens == 10
    assert response.cache_read_input_tokens == 20
    assert response.estimated_cost > 0
    assert response.provider_request_id == "provider-request-1"


def test_invalid_structured_output_preserves_billed_usage():
    class InvalidRunnable:
        def invoke(self, messages, config):
            return {
                "parsed": None,
                "parsing_error": ValueError("invalid structured output"),
                "raw": SimpleNamespace(
                    usage_metadata={
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "input_token_details": {
                            "cache_creation": 10,
                            "cache_read": 20,
                        },
                    },
                    response_metadata={},
                ),
            }

    class InvalidModel:
        def with_structured_output(self, response_model, *, include_raw):
            return InvalidRunnable()

    config = _call_config()
    client = LangChainStructuredModelClient(
        InvalidModel(),
        provider=config.provider,
        model_id=config.model_id,
    )
    with pytest.raises(ProviderCallError) as captured:
        client.generate_structured(
            call_id="call-invalid",
            system_prompt="system",
            user_prompt="user",
            response_model=HypothesisProposal,
            call_config=config,
            context_hash="context-hash",
        )
    error = captured.value
    assert error.code == ProviderErrorCode.INVALID_STRUCTURED_OUTPUT
    assert error.input_tokens == 100
    assert error.output_tokens == 50
    assert error.cache_creation_input_tokens == 10
    assert error.cache_read_input_tokens == 20
    assert error.estimated_cost == config.pricing.estimate(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=10,
        cache_read_input_tokens=20,
    )


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (TimeoutError("timed out"), ProviderErrorCode.TIMEOUT, True),
        (
            RuntimeError("429 rate limit"),
            ProviderErrorCode.RATE_LIMITED,
            True,
        ),
        (
            ValueError("authentication failed for API key"),
            ProviderErrorCode.AUTHENTICATION_ERROR,
            False,
        ),
    ],
)
def test_provider_errors_are_safely_classified(error, code, retryable):
    class FailingRunnable:
        def invoke(self, messages, config):
            raise error

    class FailingModel:
        def with_structured_output(self, response_model, *, include_raw):
            return FailingRunnable()

    config = _call_config()
    client = LangChainStructuredModelClient(
        FailingModel(),
        provider=config.provider,
        model_id=config.model_id,
    )
    with pytest.raises(ProviderCallError) as captured:
        client.generate_structured(
            call_id="call-fail",
            system_prompt="system",
            user_prompt="user",
            response_model=HypothesisProposal,
            call_config=config,
            context_hash="context-hash",
        )
    assert captured.value.code == code
    assert captured.value.retryable is retryable
    assert str(captured.value) == code.value
