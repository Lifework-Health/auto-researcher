from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from auto_researcher.agents.call_store import InMemoryAgentCallStore
from auto_researcher.agents.context import stable_context_hash
from auto_researcher.agents.models import (
    AgentCallRecord,
    HypothesisProposal,
    ModelCallConfig,
    ModelPricing,
    PlannerProposal,
)
from auto_researcher.contracts.enums import (
    AgentCallStatus,
    AgentRole,
    SearchType,
)


def test_pricing_and_call_config_are_explicit_and_reject_latest_alias():
    pricing = ModelPricing(
        version="test-v1",
        input_cost_per_million_tokens=2,
        output_cost_per_million_tokens=4,
        currency="USD",
    )
    assert pricing.estimate(input_tokens=1_000_000, output_tokens=500_000) == 4
    with pytest.raises(ValidationError, match="latest alias"):
        ModelCallConfig(
            provider="fake",
            model_id="model-latest",
            temperature=0,
            maximum_output_tokens=100,
            timeout_seconds=30,
            maximum_attempts=2,
            maximum_cost_per_call=1,
            pricing=pricing,
            prompt_version="1.0.0",
        )


def test_proposals_cannot_supply_trusted_platform_fields():
    with pytest.raises(ValidationError):
        HypothesisProposal(
            hypothesis_id="invented",
            statement="test",
            rationale="summary",
            predicted_subspace={"x": 1},
            expected_observation="metric changes",
            falsification_condition="metric does not change",
            confidence=0.5,
        )
    with pytest.raises(ValidationError):
        PlannerProposal(
            search_type=SearchType.DIRECT,
            target="metric",
            proposed_search_space={},
            requested_experiment_budget=1,
            rationale="summary",
            request_id="invented",
        )


def test_planner_proposal_supports_openevolve_without_platform_identities():
    proposal = PlannerProposal(
        search_type=SearchType.OPENEVOLVE,
        target="bounded scoring transformation",
        proposed_search_space={"openevolve": {"population_size": 1}},
        requested_experiment_budget=1,
        rationale="offline bounded search",
    )
    assert proposal.search_type == SearchType.OPENEVOLVE


def test_context_hash_is_canonical():
    assert stable_context_hash({"b": 2, "a": 1}) == stable_context_hash(
        {"a": 1, "b": 2}
    )


def test_call_store_is_append_only_and_retry_is_linked():
    now = datetime(2026, 7, 30, tzinfo=UTC)
    store = InMemoryAgentCallStore()
    reserved = AgentCallRecord(
        record_id="call-1:1:reserved",
        call_id="call-1",
        run_id="run-1",
        cycle=1,
        role=AgentRole.HYPOTHESIS,
        provider="fake",
        model_id="fake-model-1",
        prompt_name="hypothesis",
        prompt_version="1.0.0",
        prompt_hash="prompt-hash",
        context_hash="context-hash",
        response_schema_version="schema-hash",
        status=AgentCallStatus.RESERVED,
        pricing=ModelPricing(
            version="test-v1",
            input_cost_per_million_tokens=1,
            output_cost_per_million_tokens=2,
            currency="USD",
        ),
        pricing_version="test-v1",
        pricing_currency="USD",
        created_at=now,
        provider_request_started=True,
    )
    store.append(reserved)
    indeterminate = reserved.model_copy(
        update={
            "record_id": "call-1:2:indeterminate",
            "status": AgentCallStatus.INDETERMINATE,
        }
    )
    store.append(indeterminate)
    retry = store.create_retry("call-1", created_at=now)
    assert retry.call_id != reserved.call_id
    assert retry.retry_of_call_id == reserved.call_id
    assert store.records_for_call("call-1") == (reserved, indeterminate)
