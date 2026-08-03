from __future__ import annotations

import sys
from pathlib import Path
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from auto_researcher.agents.call_store import SQLiteAgentCallStore
from auto_researcher.agents.models import AgentCallRecord, ModelPricing
from auto_researcher.cli import _load_live_agents, _load_task_configuration, app
from auto_researcher.contracts.enums import AgentCallStatus, AgentRole
from auto_researcher.runtime.dependencies import memory_dependencies


def test_task_configuration_identity_requires_matching_id_and_version(tmp_path):
    configuration = tmp_path / "task.yaml"
    configuration.write_text(
        """
task:
  id: synthetic
  version: "2.0"
experiment:
  model_family: tree
  complexity: 4
  learning_rate: 0.05
runtime: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version"):
        _load_task_configuration(configuration, "synthetic", "1.0")


def test_example_task_configurations_have_expected_sections():
    repository = Path(__file__).resolve().parents[2]
    for task_id in ("synthetic", "icca_nbs"):
        experiment, runtime = _load_task_configuration(
            repository / "examples" / "tasks" / task_id / "task.yaml",
            task_id,
            "1.0",
        )
        assert experiment
        assert "output_dir" in runtime


def test_optuna_task_configuration_uses_search_section():
    repository = Path(__file__).resolve().parents[2]
    search, runtime = _load_task_configuration(
        repository / "examples/tasks/synthetic/optuna.yaml",
        "synthetic",
        "1.0",
    )
    assert search["trial_budget"] == 8
    assert "type" not in search
    assert "output_dir" in runtime


def test_mock_agent_mode_is_default_and_live_requires_pricing():
    assert _load_live_agents({})[-1] == "mock"
    with pytest.raises(ValueError, match="pricing"):
        _load_live_agents(
            {
                "agents": {
                    "mode": "live",
                    "provider": "anthropic",
                    "model_id": "explicit-model-2026-07-30",
                }
            }
        )


def test_default_runtime_does_not_import_optional_provider_packages():
    memory_dependencies()
    assert "langchain_anthropic" not in sys.modules
    assert "anthropic" not in sys.modules
    assert "neo4j" not in sys.modules


def test_live_anthropic_missing_key_is_actionable_before_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    role = {
        "maximum_output_tokens": 100,
        "timeout_seconds": 10,
        "maximum_attempts": 1,
        "maximum_cost_per_call": 0.1,
    }
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        _load_live_agents(
            {
                "agents": {
                    "mode": "live",
                    "provider": "anthropic",
                    "model_id": "explicit-model-2026-07-30",
                    "pricing": {
                        "version": "test-v1",
                        "input_cost_per_million_tokens": 1,
                        "output_cost_per_million_tokens": 2,
                        "currency": "USD",
                    },
                    "hypothesis": role,
                    "planner": role,
                }
            }
        )


def test_agent_call_cli_lists_shows_and_authorises_retry(tmp_path):
    path = tmp_path / "agent-calls.sqlite"
    store = SQLiteAgentCallStore(path)
    now = datetime(2026, 7, 30, tzinfo=UTC)
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
    store.append(
        reserved.model_copy(
            update={
                "record_id": "call-1:2:indeterminate",
                "status": AgentCallStatus.INDETERMINATE,
            }
        )
    )
    store.close()
    runner = CliRunner()
    listed = runner.invoke(
        app,
        ["agent-calls", "list", "--run-id", "run-1", "--agent-calls-db", str(path)],
    )
    assert listed.exit_code == 0
    assert "INDETERMINATE" in listed.stdout
    shown = runner.invoke(
        app,
        ["agent-calls", "show", "--call-id", "call-1", "--agent-calls-db", str(path)],
    )
    assert shown.exit_code == 0
    assert "prompt_hash" in shown.stdout
    assert "system_prompt" not in shown.stdout
    retried = runner.invoke(
        app,
        ["agent-calls", "retry", "--call-id", "call-1", "--agent-calls-db", str(path)],
    )
    assert retried.exit_code == 0
    assert "linked to call-1" in retried.stdout
    reopened = SQLiteAgentCallStore(path)
    try:
        assert any(
            record.retry_of_call_id == "call-1"
            for record in reopened.list_records("run-1")
        )
    finally:
        reopened.close()


def test_run_start_inspect_and_terminal_resume_are_explicit_and_safe(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    checkpoint = tmp_path / "checkpoints.sqlite"
    provenance = tmp_path / "provenance.sqlite"
    agent_calls = tmp_path / "agent-calls.sqlite"
    knowledge = tmp_path / "knowledge.sqlite"
    stores = [
        "--checkpoint-db",
        str(checkpoint),
        "--provenance-db",
        str(provenance),
        "--agent-calls-db",
        str(agent_calls),
        "--knowledge-retrievals-db",
        str(knowledge),
    ]
    start = runner.invoke(
        app,
        [
            "run",
            "start",
            "--run-id",
            "cli-run",
            "--thread-id",
            "cli-thread",
            *stores,
        ],
    )
    assert start.exit_code == 0, start.stdout
    assert "Status: COMPLETED" in start.stdout

    duplicate = runner.invoke(
        app,
        [
            "run",
            "start",
            "--run-id",
            "cli-run",
            "--thread-id",
            "cli-thread",
            *stores,
        ],
    )
    assert duplicate.exit_code == 2
    assert "thread_already_exists_use_resume_or_inspect" in duplicate.stderr

    inspected = runner.invoke(
        app,
        [
            "run",
            "inspect",
            "--thread-id",
            "cli-thread",
            "--checkpoint-db",
            str(checkpoint),
        ],
    )
    assert inspected.exit_code == 0, inspected.stdout
    assert "Execution protocol: run-execution-v2" in inspected.stdout
    assert "Status: COMPLETED" in inspected.stdout

    resumed = runner.invoke(
        app,
        [
            "run",
            "resume",
            "--thread-id",
            "cli-thread",
            *stores,
        ],
    )
    assert resumed.exit_code == 2
    assert "thread_is_terminal_use_inspect" in resumed.stderr
