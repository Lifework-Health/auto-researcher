from __future__ import annotations

from dataclasses import replace

import pytest

from auto_researcher.graph.builder import build_graph
from auto_researcher.graph.nodes.evaluate import evaluate_experiment
from auto_researcher.graph.nodes.verify import verify_evidence
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.execution import start_run
from auto_researcher.tasks import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)


class CountingEvaluator:
    def __init__(self, inner):
        self.inner = inner
        self.evaluator_id = inner.evaluator_id
        self.version = inner.version
        self.cost_per_experiment = inner.cost_per_experiment
        self.calls = 0

    def evaluate(self, experiment, contract):
        self.calls += 1
        return self.inner.evaluate(experiment, contract)


class CountingVerifier:
    def __init__(self, inner):
        self.inner = inner
        self.verifier_id = inner.verifier_id
        self.version = inner.version
        self.calls = 0

    def verify(self, *args, **kwargs):
        self.calls += 1
        return self.inner.verify(*args, **kwargs)


def _runtime(tmp_path):
    contract = default_synthetic_contract()
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="reuse-run", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
    )
    evaluator = CountingEvaluator(dependencies.evaluator)
    verifier = CountingVerifier(dependencies.verifier)
    dependencies = replace(
        dependencies,
        evaluator=evaluator,
        verifier=verifier,
    )
    graph = build_graph(dependencies)
    initial = {
        "run_id": "reuse-run",
        "thread_id": "reuse-thread",
        "contract": contract,
    }
    config = {"configurable": {"thread_id": "reuse-thread"}}
    return dependencies, evaluator, verifier, graph, initial, config


def test_direct_duplicate_execution_reuses_result_and_verification_without_writes(
    tmp_path,
):
    dependencies, evaluator, verifier, graph, initial, config = _runtime(tmp_path)
    first = start_run(graph, initial, config)
    paths = [tmp_path / item for item in first["evaluation_result"].artefact_references]
    snapshot = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    events = dependencies.provenance_store.list_events("reuse-run")

    second = graph.invoke(initial, config)  # Deliberate bypass of run-execution-v2.

    assert (evaluator.calls, verifier.calls) == (1, 1)
    assert second["evaluation_result"] == first["evaluation_result"]
    assert second["verification_result"] == first["verification_result"]
    assert dependencies.provenance_store.list_events("reuse-run") == events
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths
    } == snapshot
    assert "evaluate_experiment_reused" in second["executed_nodes"]
    assert "verify_evidence_reused" in second["executed_nodes"]


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_missing_or_tampered_artefact_prevents_evaluation_reuse(tmp_path, damage):
    _, evaluator, verifier, graph, initial, config = _runtime(tmp_path)
    first = start_run(graph, initial, config)
    evaluation_path = next(
        tmp_path / item
        for item in first["evaluation_result"].artefact_references
        if item.endswith("evaluation_result.json")
    )
    if damage == "missing":
        evaluation_path.unlink()
    else:
        evaluation_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artefact_bundle_(missing|tampered)"):
        graph.invoke(initial, config)
    assert (evaluator.calls, verifier.calls) == (1, 1)


def test_conflicting_experiment_spec_prevents_result_reuse(tmp_path):
    dependencies, evaluator, _, graph, initial, config = _runtime(tmp_path)
    final = start_run(graph, initial, config)
    conflicting = final["experiment_spec"].model_copy(
        update={"code_version": "different-code-version"}
    )
    state = {**final, "experiment_spec": conflicting}

    with pytest.raises(RuntimeError, match="conflicting_completed_evaluation_identity"):
        evaluate_experiment(state, dependencies)
    assert evaluator.calls == 1


def test_changed_verifier_policy_prevents_verification_reuse(tmp_path):
    dependencies, _, verifier, graph, initial, config = _runtime(tmp_path)
    final = start_run(graph, initial, config)

    class ChangedPolicy:
        policy_id = "synthetic-policy-v2"
        required_metrics = dependencies.verification_policy.required_metrics

        def evaluate_constraints(self, evaluation, contract):
            return dependencies.verification_policy.evaluate_constraints(
                evaluation,
                contract,
            )

    changed = replace(dependencies, verification_policy=ChangedPolicy())
    with pytest.raises(
        RuntimeError,
        match="conflicting_completed_verification_identity",
    ):
        verify_evidence(final, changed)
    assert verifier.calls == 1
