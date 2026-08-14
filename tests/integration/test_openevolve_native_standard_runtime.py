from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from auto_researcher.contracts.enums import RunStatus, SearchType
from auto_researcher.graph.builder import build_graph
from auto_researcher.resources import (
    CourtesyResourceAdmissionPolicy,
    InMemoryResourceLeaseStore,
    ResourceBroker,
)
from auto_researcher.runtime.dependencies import task_sqlite_dependencies
from auto_researcher.runtime.execution import resume_run, start_run
from auto_researcher.search.openevolve.native_engine import ApprovedModel
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_contract,
    default_synthetic_openevolve_configuration,
)
from tests.integration.test_openevolve_full_strength import (
    ScriptedModel,
    SimulatedGPUProvider,
)

pytestmark = pytest.mark.upstream_openevolve


TREE = """def evolve(configuration):
    return {"model_family": "tree", "complexity": 4, "learning_rate": 0.05}
"""
NEURAL = """def evolve(configuration):
    return {"model_family": "neural", "complexity": 4, "learning_rate": 0.05}
"""
LINEAR_FIVE = """def evolve(configuration):
    return {"model_family": "linear", "complexity": 5, "learning_rate": 0.05}
"""
TREE_DUPLICATE = """def evolve(configuration):
 return {"learning_rate": .05, "complexity": 4, "model_family": "tree"}
"""
NEURAL_DUPLICATE = """def evolve(configuration):
 return {"learning_rate": .05, "complexity": 4, "model_family": "neural"}
"""


class DelayedEvaluator:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.evaluator_id = delegate.evaluator_id
        self.version = delegate.version
        self.cost_per_experiment = delegate.cost_per_experiment
        self.active = 0
        self.maximum_parallel = 0
        self._lock = threading.Lock()

    def evaluate(self, experiment, contract):
        with self._lock:
            self.active += 1
            self.maximum_parallel = max(self.maximum_parallel, self.active)
        try:
            time.sleep(0.05)
            return self.delegate.evaluate(experiment, contract)
        finally:
            with self._lock:
                self.active -= 1


def _native_configuration() -> dict:
    configuration = default_synthetic_openevolve_configuration()
    configuration["openevolve"].update(
        {
            "native_controller": True,
            "population_size": 6,
            "archive_size": 6,
            "num_islands": 3,
            "migration_interval": 1,
            "migration_rate": 0.5,
            "feature_dimensions": ["primary_score", "runtime"],
            "feature_bins": 4,
            "parallel_evaluations": 3,
            "checkpoint_interval": 1,
            "maximum_generations": 5,
            "maximum_candidate_evaluations": 6,
            "maximum_model_calls": 5,
            "standard_runtime_iterations_per_invocation": 3,
            "objective_threshold": None,
        }
    )
    configuration["resources"] = {
        "mode": "equivalent_pool",
        "resource_type": "gpu",
        "quantity_per_candidate": 1,
        "maximum_wait_seconds": 5,
        "equivalence_requirements": ["equivalent-a4-gpu"],
    }
    return configuration


def _manager(
    tmp_path: Path,
    *,
    evaluator: DelayedEvaluator,
    model: ScriptedModel,
    broker: ResourceBroker,
    configuration: dict,
    contract,
):
    return task_sqlite_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(
            run_id="native-standard-a4",
            output_dir=tmp_path / "outputs",
            workspace_dir=tmp_path / "workspace",
        ),
        contract,
        configuration,
        tmp_path / "checkpoints.sqlite",
        tmp_path / "provenance.sqlite",
        agent_calls_path=tmp_path / "agent-calls.sqlite",
        knowledge_retrievals_path=tmp_path / "knowledge.sqlite",
        evaluator=evaluator,
        search_type=SearchType.OPENEVOLVE,
        native_openevolve_models=(
            ApprovedModel(name=model.model, weight=1.0, adapter=model),
        ),
        native_openevolve_resource_broker=broker,
    )


def test_standard_runtime_native_a4_like_start_resume_and_reuse_v2(
    tmp_path: Path,
) -> None:
    pytest.importorskip("openevolve")
    configuration = _native_configuration()
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPENEVOLVE}),
        maximum_experiments=6,
    )
    context = TaskRuntimeContext(
        run_id="native-standard-a4",
        output_dir=tmp_path / "outputs",
        workspace_dir=tmp_path / "workspace",
    )
    evaluator = DelayedEvaluator(SyntheticTask().create_evaluator(context))
    model = ScriptedModel((TREE, NEURAL, LINEAR_FIVE, TREE_DUPLICATE, NEURAL_DUPLICATE))
    broker = ResourceBroker(
        SimulatedGPUProvider(3),
        CourtesyResourceAdmissionPolicy(maximum_utilization_percent=80),
        lease_store=InMemoryResourceLeaseStore(),
        poll_seconds=0.005,
    )
    graph_config = {"configurable": {"thread_id": "native-standard-a4-thread"}}
    initial = {
        "run_id": "native-standard-a4",
        "thread_id": "native-standard-a4-thread",
        "contract": contract,
    }

    with _manager(
        tmp_path,
        evaluator=evaluator,
        model=model,
        broker=broker,
        configuration=configuration,
        contract=contract,
    ) as dependencies:
        assert dependencies.native_openevolve_runtime is not None
        paused = start_run(
            build_graph(
                dependencies,
                interrupt_after=["run_native_openevolve"],
            ),
            initial,
            graph_config,
        )
        first = paused["openevolve_native_result"]

    assert paused["status"] == RunStatus.RUNNING
    assert paused["executed_nodes"][-1] == "run_native_openevolve"
    assert first.completed_iterations == 3
    assert not first.finished
    assert {resource for _, resource in first.resource_placements} == {
        "gpu:0",
        "gpu:1",
        "gpu:2",
    }
    assert evaluator.maximum_parallel == 3

    with _manager(
        tmp_path,
        evaluator=evaluator,
        model=model,
        broker=broker,
        configuration=configuration,
        contract=contract,
    ) as dependencies:
        final = resume_run(build_graph(dependencies), graph_config)
        result = final["openevolve_native_result"]
        records = tuple(
            dependencies.provenance_store.get_evaluation_reuse(
                "native-standard-a4",
                feedback.evaluation_reuse_experiment_id,
            )
            for feedback in result.feedback
            if feedback.evaluation_reuse_experiment_id is not None
        )

    assert final["status"] == RunStatus.COMPLETED
    assert result.finished
    assert result.resumed_from_iteration == 3
    assert result.completed_iterations == 5
    assert set(first.programme_ids).intersection(result.programme_ids)
    assert result.expensive_evaluations == 4
    assert result.reused_evaluations == 2
    assert model.calls == 5
    assert records and all(record is not None for record in records)
    assert all(record.protocol_version == "evaluation-reuse-v2" for record in records)
    assert all(
        feedback.evaluation_reuse_identity_hash is not None
        for feedback in result.feedback
    )


@pytest.mark.parametrize("native_value", [False, None])
def test_explicit_false_or_omitted_native_controller_keeps_legacy_backend(
    tmp_path: Path,
    native_value: bool | None,
) -> None:
    configuration = default_synthetic_openevolve_configuration()
    if native_value is not None:
        configuration["openevolve"]["native_controller"] = native_value
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPENEVOLVE}),
        maximum_experiments=4,
    )
    with task_sqlite_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(
            run_id=f"legacy-{native_value}",
            output_dir=tmp_path / "outputs",
            workspace_dir=tmp_path / "workspace",
        ),
        contract,
        configuration,
        tmp_path / f"checkpoints-{native_value}.sqlite",
        tmp_path / f"provenance-{native_value}.sqlite",
        search_type=SearchType.OPENEVOLVE,
    ) as dependencies:
        assert dependencies.openevolve_backend is not None
        assert dependencies.native_openevolve_runtime is None
