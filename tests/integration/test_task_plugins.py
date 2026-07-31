from __future__ import annotations

import inspect
import subprocess
import sys
from datetime import UTC, datetime

from auto_researcher.contracts.enums import (
    EvidenceStatus,
    EventType,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.tasks.icca_nbs import ICCANBSTask
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.registry import default_task_registry
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)
from auto_researcher.tasks.synthetic.verification import SyntheticVerificationPolicy
from tests.fakes_icca import make_fake_icca_bindings


FIXED_TIME = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


def _invoke(graph, contract, run_id, thread_id):
    return graph.invoke(
        {"run_id": run_id, "thread_id": thread_id, "contract": contract},
        {"configurable": {"thread_id": thread_id}},
    )


def _icca_contract() -> ResearchContract:
    return ResearchContract(
        contract_id="icca-fake-contract",
        schema_version="1.0",
        task_id="icca_nbs",
        task_version="1.0",
        objective_version="0.9",
        primary_metric="stability_objective",
        task_constraints_version="1.0",
        question="Does this bounded iCCA configuration meet its eligibility gates?",
        objective="maximise the imported v2 stability objective",
        constraints={},
        allowed_search_types=frozenset({SearchType.DIRECT}),
        evaluator_id="icca-nbs-v2-evaluator",
        verifier_id="deterministic-verifier",
        maximum_cycles=1,
        maximum_experiments=1,
        maximum_cost=1.0,
        provenance=ProvenanceKind.REAL,
    )


def _fake_icca_context(tmp_path, run_id="icca-fake"):
    (tmp_path / "Combined_binary_matrix.csv").write_text(
        "PatID,G1\nprivate-patient,1\n",
        encoding="utf-8",
    )
    (tmp_path / "Combined_clinical.csv").write_text(
        "PatID,OS_MONTHS\nprivate-patient,12\n",
        encoding="utf-8",
    )
    return TaskRuntimeContext(
        run_id=run_id,
        data_dir=tmp_path,
        workspace_dir=tmp_path,
        output_dir=tmp_path / "outputs",
        manifest_created_at=FIXED_TIME,
    )


def test_synthetic_plugin_completes_with_provenance_and_safe_artefacts(tmp_path):
    registry = default_task_registry()
    task = registry.get("synthetic", "1.0")
    context = TaskRuntimeContext(
        run_id="synthetic-e2e",
        output_dir=tmp_path,
        manifest_created_at=FIXED_TIME,
    )
    dependencies = task_memory_dependencies(
        task,
        context,
        default_synthetic_contract(),
        default_synthetic_configuration(),
    )

    final = _invoke(
        build_graph(dependencies),
        default_synthetic_contract(),
        "synthetic-e2e",
        "synthetic-e2e-thread",
    )

    assert final["evaluation_result"].success is True
    assert final["verification_result"].verified is True
    assert final["verification_result"].evidence_status == EvidenceStatus.INCONCLUSIVE
    assert final["executed_nodes"].count("evaluate_experiment") == 1
    assert final["executed_nodes"].count("verify_evidence") == 1
    events = dependencies.provenance_store.list_events("synthetic-e2e")
    assert [event.event_type for event in events] == [
        EventType.HYPOTHESIS_PROPOSED,
        EventType.SEARCH_PLANNED,
        EventType.EXPERIMENT_PREPARED,
        EventType.EVALUATION_OBSERVED,
        EventType.EVIDENCE_VERIFIED,
    ]
    evaluation_event = next(
        event
        for event in events
        if event.event_type == EventType.EVALUATION_OBSERVED
    )
    assert set(final["evaluation_result"].artefact_references).issubset(
        evaluation_event.output_references
    )
    artefact_dir = (
        tmp_path
        / "runs"
        / "synthetic-e2e"
        / final["evaluation_result"].experiment_id
    )
    assert {path.name for path in artefact_dir.iterdir()} == {
        "experiment_spec.json",
        "evaluation_result.json",
        "dataset_manifest.json",
        "evaluator_manifest.json",
    }
    assert all(
        not reference.startswith("/")
        for reference in final["evaluation_result"].artefact_references
    )


def test_synthetic_policy_covers_supported_refuted_and_inconclusive():
    task = SyntheticTask()
    contract = default_synthetic_contract()
    evaluator = task.create_evaluator(TaskRuntimeContext())
    metadata = task.experiment_metadata(TaskRuntimeContext())
    policy = SyntheticVerificationPolicy()

    configurations = (
        (default_synthetic_configuration(), EvidenceStatus.SUPPORTED),
        (
            {"model_family": "linear", "complexity": 10, "learning_rate": 1.0},
            EvidenceStatus.REFUTED,
        ),
        (
            {"model_family": "linear", "complexity": 8, "learning_rate": 0.4},
            EvidenceStatus.INCONCLUSIVE,
        ),
    )
    from auto_researcher.contracts.models import ExperimentSpec

    for index, (configuration, expected) in enumerate(configurations):
        experiment = ExperimentSpec(
            experiment_id=f"synthetic-outcome-{index}",
            hypothesis_id="hypothesis",
            search_request_id="request",
            configuration=task.normalise_configuration(configuration),
            evaluator_id=metadata.evaluator_id,
            code_version=metadata.code_version,
            dataset_version=metadata.dataset_version,
            provenance=metadata.provenance,
        )
        evaluation = evaluator.evaluate(experiment, contract)
        assert policy.evaluate_constraints(evaluation, contract).evidence_status == expected


def test_synthetic_path_never_imports_optional_harness():
    script = """
import sys
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)
contract = default_synthetic_contract()
deps = task_memory_dependencies(
    SyntheticTask(), TaskRuntimeContext(), contract,
    default_synthetic_configuration(),
)
build_graph(deps).invoke(
    {"run_id": "offline", "thread_id": "offline-thread", "contract": contract},
    {"configurable": {"thread_id": "offline-thread"}},
)
assert not any(name == "harness" or name.startswith("harness.") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_same_graph_topology_and_node_sequence_across_tasks(tmp_path):
    synthetic_contract = default_synthetic_contract()
    synthetic_dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(manifest_created_at=FIXED_TIME),
        synthetic_contract,
        default_synthetic_configuration(),
    )
    bindings, calls = make_fake_icca_bindings()
    icca_task = ICCANBSTask(bindings)
    icca_contract = _icca_contract()
    icca_dependencies = task_memory_dependencies(
        icca_task,
        _fake_icca_context(tmp_path),
        icca_contract,
        {
            "network": "ideker",
            "alignment": "intersect",
            "alpha": 0.7,
            "K": 5,
            "r": 10,
        },
    )
    synthetic_graph = build_graph(synthetic_dependencies)
    icca_graph = build_graph(icca_dependencies)

    assert (
        synthetic_graph.get_graph().draw_mermaid()
        == icca_graph.get_graph().draw_mermaid()
    )
    synthetic_final = _invoke(
        synthetic_graph,
        synthetic_contract,
        "cross-synthetic",
        "cross-synthetic-thread",
    )
    icca_final = _invoke(
        icca_graph,
        icca_contract,
        "cross-icca",
        "cross-icca-thread",
    )
    assert synthetic_final["executed_nodes"] == icca_final["executed_nodes"]
    assert synthetic_final["experiment_spec"].configuration != (
        icca_final["experiment_spec"].configuration
    )
    assert synthetic_dependencies.evaluator.evaluator_id != (
        icca_dependencies.evaluator.evaluator_id
    )
    assert synthetic_dependencies.verification_policy.policy_id != (
        icca_dependencies.verification_policy.policy_id
    )
    assert calls["evaluate"] == 1


def test_graph_builder_has_no_task_specific_branch():
    source = inspect.getsource(build_graph).casefold()
    assert "task_id" not in source
    assert "synthetic" not in source
    assert "icca" not in source
    assert "segmentation" not in source
