from __future__ import annotations

import importlib.util
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from auto_researcher.contracts.enums import RunStatus, SearchType
from auto_researcher.contracts.models import SearchRequest
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.execution import start_run
from auto_researcher.search.openevolve.artifacts import verify_search_artefacts
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.hardened_executor import (
    HardenedDockerExecutor,
    docker_policy,
)
from auto_researcher.search.openevolve.mutation import DeterministicMutationOperator
from auto_researcher.tasks.iris_knn import IrisKNNTask, default_iris_contract
from auto_researcher.tasks.iris_knn.openevolve import (
    default_iris_openevolve_configuration,
)
from auto_researcher.tasks.models import TaskRuntimeContext

FIXED_TIME = datetime(2026, 8, 7, 12, tzinfo=UTC)


def _invoke(dependencies, contract, run_id):
    return start_run(
        build_graph(dependencies),
        {"run_id": run_id, "thread_id": f"{run_id}-thread", "contract": contract},
        {"configurable": {"thread_id": f"{run_id}-thread"}},
    )


@pytest.mark.hpo
@pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None, reason="Optuna is not installed"
)
def test_iris_optuna_completes_bounded_real_data_study(tmp_path):
    task = IrisKNNTask()
    contract = default_iris_contract(
        search_types=frozenset({SearchType.OPTUNA}), maximum_experiments=20
    )
    context = TaskRuntimeContext(
        run_id="iris-optuna",
        output_dir=tmp_path,
        manifest_created_at=FIXED_TIME,
    )
    dependencies = task_memory_dependencies(
        task,
        context,
        contract,
        {"trial_budget": 20, "seed": 20260807},
        search_type=SearchType.OPTUNA,
        clock=lambda: FIXED_TIME,
    )
    final = _invoke(dependencies, contract, "iris-optuna")
    result = final["optuna_study_result"]
    assert final["status"] == RunStatus.COMPLETED
    assert result.trials_completed == 20
    assert result.best_feasible_score is not None
    assert result.best_feasible_score >= 0.94
    assert final["executed_nodes"].count("evaluate_experiment") == 20
    assert final["executed_nodes"].count("verify_evidence") == 20


def test_iris_offline_openevolve_evaluates_seed_and_known_regression_fixtures(tmp_path):
    task = IrisKNNTask()
    contract = default_iris_contract(
        search_types=frozenset({SearchType.OPENEVOLVE}), maximum_experiments=3
    )
    context = TaskRuntimeContext(
        run_id="iris-openevolve",
        output_dir=tmp_path / "artefacts",
        workspace_dir=tmp_path / "workspace",
        manifest_created_at=FIXED_TIME,
    )
    configuration = default_iris_openevolve_configuration()
    dependencies = task_memory_dependencies(
        task,
        context,
        contract,
        configuration,
        search_type=SearchType.OPENEVOLVE,
        clock=lambda: FIXED_TIME,
    )
    final = _invoke(dependencies, contract, "iris-openevolve")
    population = final["openevolve_population_state"]
    result = final["openevolve_search_result"]
    scores = [item.objective_value for item in population.outcomes]
    assert final["status"] == RunStatus.COMPLETED
    direct_context = TaskRuntimeContext(
        run_id="iris-direct-comparison",
        output_dir=tmp_path / "direct-artefacts",
        manifest_created_at=FIXED_TIME,
    )
    direct = _invoke(
        task_memory_dependencies(
            task,
            direct_context,
            default_iris_contract(),
            {
                "feature_weights": [1.0, 1.0, 1.0, 1.0],
                "k": 3,
                "distance_power": 2,
            },
            clock=lambda: FIXED_TIME,
        ),
        default_iris_contract(),
        "iris-direct-comparison",
    )
    assert scores[0] == direct["evaluation_result"].primary_score
    assert scores == [0.94, 0.953333333333, 0.96]
    assert result.generations_completed == 2
    assert result.stop_reason == "maximum_candidate_evaluations_reached"
    assert population.budget.model_calls == 0
    assert population.budget.candidate_evaluations == 3
    assert population.budget.verifier_calls == 3
    assert all(
        item.verified and item.constraint_compliant for item in population.outcomes
    )
    assert verify_search_artefacts(context, result.search_request_id)[0] is True


@pytest.mark.hardened_executor
def test_iris_seed_prepares_in_retained_hardened_image(tmp_path):
    image = os.getenv("AUTO_RESEARCHER_HARDENED_IMAGE")
    digest = os.getenv("AUTO_RESEARCHER_HARDENED_IMAGE_DIGEST")
    if not image or not digest:
        pytest.skip("retained hardened image and digest were not explicitly selected")
    root = Path(__file__).parents[2]
    docker_version = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    executor = HardenedDockerExecutor(
        docker_policy(
            image,
            digest,
            root / "docker/openevolve-executor/Dockerfile",
            root / "docker/openevolve-executor/worker.py",
            docker_version,
        ),
        tmp_path / "hardened-workspace",
    )
    task = IrisKNNTask()
    contract = default_iris_contract(
        search_types=frozenset({SearchType.OPENEVOLVE}), maximum_experiments=3
    )
    context = TaskRuntimeContext(run_id="iris-hardened", output_dir=tmp_path)
    dependencies = task_memory_dependencies(
        task,
        context,
        contract,
        default_iris_openevolve_configuration(),
        search_type=SearchType.OPENEVOLVE,
    )
    assert dependencies.openevolve_backend is not None
    existing = dependencies.openevolve_backend
    backend = OpenEvolveBackend(
        existing.component,
        existing.metadata,
        existing.verifier_identity,
        DeterministicMutationOperator(),
        executor,
    )
    raw = default_iris_openevolve_configuration()["openevolve"]
    request = SearchRequest(
        request_id="iris-hardened-request",
        hypothesis_id="iris-hardened-hypothesis",
        search_type=SearchType.OPENEVOLVE,
        target="bounded Iris weighted k-NN configuration",
        search_space={
            "openevolve": {
                **raw,
                "sandbox_policy_id": "openevolve-hardened-executor-v2",
            }
        },
        experiment_budget=3,
        rationale="retained-image offline smoke",
    )
    search = backend.create_search_contract(request, contract)
    isolation = executor.verify_isolation()
    result = backend.prepare(backend.seed_candidate(search), search)
    assert isolation.network_isolation_verified is True
    assert isolation.mount_isolation_verified is True
    assert result.execution_status.value == "COMPLETED"
    assert result.generated_configuration["feature_weights"] == [1.0] * 4
