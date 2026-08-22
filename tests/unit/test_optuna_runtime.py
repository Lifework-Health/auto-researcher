from __future__ import annotations

import pytest

from auto_researcher.contracts.enums import SearchType
from auto_researcher.runtime.dependencies import (
    task_memory_dependencies,
    task_sqlite_dependencies,
)
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import SyntheticTask, default_synthetic_contract


def test_optuna_capability_is_actionable_when_package_is_absent(monkeypatch):
    monkeypatch.setattr(
        "auto_researcher.runtime.dependencies.importlib.util.find_spec",
        lambda name: None if name == "optuna" else None,
    )
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPTUNA}),
        maximum_experiments=2,
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        contract,
        {"trial_budget": 2},
        search_type=SearchType.OPTUNA,
    )
    capability = dependencies.search_capabilities[SearchType.OPTUNA]
    assert capability.available is False
    assert "hpo extra" in capability.message


def test_three_sqlite_stores_must_be_distinct(tmp_path):
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPTUNA}),
        maximum_experiments=2,
    )
    shared = tmp_path / "shared.sqlite"
    with pytest.raises(ValueError, match="separate"):
        with task_sqlite_dependencies(
            SyntheticTask(),
            TaskRuntimeContext(),
            contract,
            {"trial_budget": 2},
            shared,
            tmp_path / "provenance.sqlite",
            shared,
            search_type=SearchType.OPTUNA,
        ):
            pass


def test_agent_call_store_must_be_separate_from_checkpoint_and_provenance(tmp_path):
    contract = default_synthetic_contract()
    checkpoint = tmp_path / "checkpoint.sqlite"
    provenance = tmp_path / "provenance.sqlite"
    with pytest.raises(ValueError, match="agent-call stores must use separate"):
        with task_sqlite_dependencies(
            SyntheticTask(),
            TaskRuntimeContext(),
            contract,
            {
                "model_family": "tree",
                "complexity": 4,
                "learning_rate": 0.05,
            },
            checkpoint,
            provenance,
            agent_calls_path=provenance,
        ):
            pass


def test_sqlite_optuna_backend_is_available_to_adaptive_direct_campaign(tmp_path):
    pytest.importorskip("optuna")
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.DIRECT, SearchType.OPTUNA}),
        maximum_experiments=2,
    )
    with task_sqlite_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(),
        contract,
        {
            "model_family": "tree",
            "complexity": 4,
            "learning_rate": 0.05,
        },
        tmp_path / "checkpoint.sqlite",
        tmp_path / "provenance.sqlite",
        tmp_path / "optuna.sqlite",
        search_type=SearchType.DIRECT,
    ) as dependencies:
        capability = dependencies.search_capabilities[SearchType.OPTUNA]
        assert capability.available is True
        assert dependencies.optuna_backend is not None
