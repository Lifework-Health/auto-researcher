"""Explicitly opt-in smoke test for the real optional Anthropic integration."""

from __future__ import annotations

import os

import pytest

from auto_researcher.agents.models import ModelCallConfig, ModelPricing
from auto_researcher.contracts.enums import RunStatus
from auto_researcher.graph.builder import build_graph
from auto_researcher.providers.anthropic import create_anthropic_client
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)

pytestmark = [
    pytest.mark.live_agent,
    pytest.mark.skipif(
        os.environ.get("AUTO_RESEARCHER_RUN_LIVE_ANTHROPIC") != "1"
        or not os.environ.get("ANTHROPIC_API_KEY")
        or not os.environ.get("AUTO_RESEARCHER_ANTHROPIC_MODEL")
        or not os.environ.get("AUTO_RESEARCHER_ANTHROPIC_INPUT_RATE")
        or not os.environ.get("AUTO_RESEARCHER_ANTHROPIC_OUTPUT_RATE"),
        reason=(
            "set AUTO_RESEARCHER_RUN_LIVE_ANTHROPIC=1, ANTHROPIC_API_KEY, "
            "AUTO_RESEARCHER_ANTHROPIC_MODEL, and explicit input/output rates "
            "to run the paid live smoke test"
        ),
    ),
]


def test_anthropic_structured_output_smoke(tmp_path):
    config = ModelCallConfig(
        provider="anthropic",
        model_id=os.environ["AUTO_RESEARCHER_ANTHROPIC_MODEL"],
        temperature=0,
        maximum_output_tokens=300,
        timeout_seconds=30,
        maximum_attempts=1,
        maximum_cost_per_call=float(
            os.environ.get("AUTO_RESEARCHER_ANTHROPIC_MAX_COST_PER_CALL", "0.25")
        ),
        pricing=ModelPricing(
            version="live-test-explicit",
            input_cost_per_million_tokens=float(
                os.environ["AUTO_RESEARCHER_ANTHROPIC_INPUT_RATE"]
            ),
            output_cost_per_million_tokens=float(
                os.environ["AUTO_RESEARCHER_ANTHROPIC_OUTPUT_RATE"]
            ),
            currency=os.environ.get("AUTO_RESEARCHER_ANTHROPIC_CURRENCY", "USD"),
        ),
        prompt_version="1.0.0",
    )
    contract = default_synthetic_contract()
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="anthropic-live-smoke", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
        model_client=create_anthropic_client(config),
        planner_model_client=create_anthropic_client(
            config.model_copy(update={"temperature": 0})
        ),
        hypothesis_call_config=config,
        planner_call_config=config.model_copy(update={"temperature": 0}),
    )
    final = build_graph(dependencies).invoke(
        {
            "run_id": "anthropic-live-smoke",
            "thread_id": "anthropic-live-smoke-thread",
            "contract": contract,
        },
        {"configurable": {"thread_id": "anthropic-live-smoke-thread"}},
    )
    assert final["status"] == RunStatus.COMPLETED
    assert final["active_hypothesis"].agent_call_id
    assert final["search_request"].agent_call_id
    assert final["verification_result"].verified is True
    assert final["budget"].model_calls_used == 2
    assert final["budget"].model_cost_used > 0
    assert os.environ["ANTHROPIC_API_KEY"] not in final["active_hypothesis"].model_dump_json()
