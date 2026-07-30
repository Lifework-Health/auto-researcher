from __future__ import annotations

from datetime import UTC, datetime

import pytest

from auto_researcher.agents.call_store import InMemoryAgentCallStore
from auto_researcher.agents.live.base import LiveAgentExecutionError
from auto_researcher.agents.live.hypothesis import LiveHypothesisAgent
from auto_researcher.agents.models import AgentBudgetPolicy
from auto_researcher.contracts.enums import AgentCallStatus
from auto_researcher.contracts.enums import ProviderErrorCode
from auto_researcher.providers.protocols import ProviderCallError
from auto_researcher.contracts.models import BudgetState
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)
from tests.fakes_agents import FakeStructuredModelClient
from tests.integration.test_live_agents import _call_config

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


class CrashClient:
    provider = "fake"
    model_id = "fake-model-2026-07-30"

    def generate_structured(self, **kwargs):
        raise KeyboardInterrupt("simulated process loss after durable reservation")


class ErrorThenSuccessClient(FakeStructuredModelClient):
    def __init__(self, hypothesis, code, *, always=False):
        super().__init__(hypothesis, {})
        self.code = code
        self.always = always
        self.attempts = 0

    def generate_structured(self, **kwargs):
        self.attempts += 1
        if self.always or self.attempts == 1:
            raise ProviderCallError(
                self.code,
                retryable=True,
                input_tokens=10,
                estimated_cost=0.00001,
            )
        return super().generate_structured(**kwargs)


def _context(dependencies):
    contract = default_synthetic_contract()
    state = {
        "run_id": "replay-run",
        "thread_id": "replay-thread",
        "contract": contract,
        "status": "RUNNING",
        "cycle": 1,
        "budget": BudgetState(
            maximum_cycles=1,
            maximum_experiments=1,
            maximum_cost=1,
            cycles_used=1,
        ),
        "decision_event_ids": [],
        "errors": [],
        "executed_nodes": [],
    }
    return dependencies.agent_context_assembler.hypothesis_context(
        state,
        dependencies.task_agent_context,
    )


def _proposal(contract_id):
    return {
        "statement": "Complexity may change the bounded objective.",
        "rationale": "Contract-bounded test.",
        "predicted_subspace": {"complexity": [3, 6]},
        "expected_observation": "objective_score changes",
        "falsification_condition": "objective_score does not change",
        "evidence_references": [contract_id],
        "confidence": 0.5,
    }


def test_unfinished_reservation_becomes_indeterminate_and_requires_linked_retry():
    contract = default_synthetic_contract()
    store = InMemoryAgentCallStore()
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(manifest_created_at=NOW),
        contract,
        default_synthetic_configuration(),
        agent_call_store=store,
        model_client=CrashClient(),
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        clock=lambda: NOW,
    )
    context = _context(dependencies)
    with pytest.raises(KeyboardInterrupt):
        dependencies.hypothesis_agent.generate(context)
    original_call_id = store.list_records("replay-run")[0].call_id

    with pytest.raises(LiveAgentExecutionError, match="indeterminate"):
        dependencies.hypothesis_agent.generate(context)
    assert store.latest(original_call_id).status == AgentCallStatus.INDETERMINATE

    authorised = store.create_retry(original_call_id, created_at=NOW)
    working = FakeStructuredModelClient(_proposal(contract.contract_id), {})
    retrying_agent = LiveHypothesisAgent(
        client=working,
        call_config=_call_config(),
        budget_policy=AgentBudgetPolicy(),
        call_store=store,
        clock=lambda: NOW,
    )
    hypothesis = retrying_agent.generate(context)
    assert hypothesis.agent_call_id == authorised.call_id
    assert store.latest(authorised.call_id).status == AgentCallStatus.COMPLETED
    assert len(working.calls) == 1


def test_completed_call_reuses_structured_output_without_second_provider_request():
    contract = default_synthetic_contract()
    store = InMemoryAgentCallStore()
    client = FakeStructuredModelClient(_proposal(contract.contract_id), {})
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(manifest_created_at=NOW),
        contract,
        default_synthetic_configuration(),
        agent_call_store=store,
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        clock=lambda: NOW,
    )
    context = _context(dependencies)
    first = dependencies.hypothesis_agent.generate(context)
    second = dependencies.hypothesis_agent.generate(context)
    assert first == second
    assert len(client.calls) == 1
    assert dependencies.hypothesis_agent.consume_telemetry().replayed is True


def test_conflicting_completed_snapshots_fail_closed():
    contract = default_synthetic_contract()
    store = InMemoryAgentCallStore()
    client = FakeStructuredModelClient(_proposal(contract.contract_id), {})
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(manifest_created_at=NOW),
        contract,
        default_synthetic_configuration(),
        agent_call_store=store,
        model_client=client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        clock=lambda: NOW,
    )
    context = _context(dependencies)
    dependencies.hypothesis_agent.generate(context)
    completed = next(
        item
        for item in store.list_records("replay-run")
        if item.status == AgentCallStatus.COMPLETED
    )
    conflicting = completed.model_copy(
        update={
            "record_id": f"{completed.call_id}:3:completed",
            "structured_output": {
                **completed.structured_output,
                "confidence": 0.1,
            },
            "response_hash": "conflicting-hash",
        }
    )
    store.append(conflicting)
    with pytest.raises(LiveAgentExecutionError, match="conflicting_completed"):
        dependencies.hypothesis_agent.generate(context)


def test_timeout_retries_once_but_rate_limit_does_not_auto_retry():
    contract = default_synthetic_contract()
    timeout_client = ErrorThenSuccessClient(
        _proposal(contract.contract_id),
        ProviderErrorCode.TIMEOUT,
    )
    timeout_dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(manifest_created_at=NOW),
        contract,
        default_synthetic_configuration(),
        model_client=timeout_client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        clock=lambda: NOW,
    )
    timeout_dependencies.hypothesis_agent.generate(_context(timeout_dependencies))
    timeout_telemetry = timeout_dependencies.hypothesis_agent.consume_telemetry()
    assert timeout_client.attempts == 2
    assert timeout_telemetry.provider_attempts == 2

    rate_client = ErrorThenSuccessClient(
        _proposal(contract.contract_id),
        ProviderErrorCode.RATE_LIMITED,
        always=True,
    )
    rate_dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(manifest_created_at=NOW),
        contract,
        default_synthetic_configuration(),
        model_client=rate_client,
        hypothesis_call_config=_call_config(),
        planner_call_config=_call_config(),
        clock=lambda: NOW,
    )
    with pytest.raises(LiveAgentExecutionError, match="RATE_LIMITED"):
        rate_dependencies.hypothesis_agent.generate(_context(rate_dependencies))
    assert rate_client.attempts == 1
