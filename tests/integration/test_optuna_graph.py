from __future__ import annotations

from datetime import UTC, datetime
import importlib.util
import inspect

import pytest

from auto_researcher.contracts.enums import (
    EvidenceStatus,
    EventType,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import EvaluationResult, ResearchContract
from auto_researcher.graph.builder import build_graph
from auto_researcher.graph.nodes import optuna as optuna_nodes
from auto_researcher.runtime.dependencies import (
    task_memory_dependencies,
    task_sqlite_dependencies,
)
from auto_researcher.tasks.icca_nbs import ICCANBSTask
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import SyntheticTask, default_synthetic_contract
from tests.fakes_icca import make_fake_icca_bindings

FIXED_TIME = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
pytestmark = [
    pytest.mark.hpo,
    pytest.mark.skipif(
        importlib.util.find_spec("optuna") is None,
        reason="install the hpo extra to run Optuna integration tests",
    ),
]


def invoke(graph, contract, run_id, thread_id, value=None):
    payload = (
        {"run_id": run_id, "thread_id": thread_id, "contract": contract}
        if value is None
        else value
    )
    return graph.invoke(payload, {"configurable": {"thread_id": thread_id}})


def test_synthetic_optuna_study_is_generic_verified_and_writes_artefacts(tmp_path):
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPTUNA}),
        maximum_experiments=8,
    )
    context = TaskRuntimeContext(
        run_id="synthetic-optuna",
        output_dir=tmp_path,
        manifest_created_at=FIXED_TIME,
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        context,
        contract,
        {"trial_budget": 8, "seed": 123},
        search_type=SearchType.OPTUNA,
        clock=lambda: FIXED_TIME,
    )
    final = invoke(
        build_graph(dependencies),
        contract,
        "synthetic-optuna",
        "synthetic-optuna-thread",
    )
    study = final["optuna_study_result"]
    assert study.trials_asked == 8
    assert study.trials_completed == 8
    assert study.best_feasible_trial_number is not None
    assert final["executed_nodes"].count("evaluate_experiment") == 8
    assert final["executed_nodes"].count("verify_evidence") == 8
    assert final["verification_result"].evidence_status == EvidenceStatus.INCONCLUSIVE
    assert len(study.artefact_references) == 4
    assert all((tmp_path / reference).is_file() for reference in study.artefact_references)

    events = dependencies.provenance_store.list_events("synthetic-optuna")
    assert sum(event.event_type == EventType.HYPOTHESIS_PROPOSED for event in events) == 1
    assert sum(event.event_type == EventType.SEARCH_PLANNED for event in events) == 1
    assert sum(event.event_type == EventType.OPTUNA_TRIAL_PROPOSED for event in events) == 8
    assert sum(event.event_type == EventType.EXPERIMENT_PREPARED for event in events) == 8
    assert sum(event.event_type == EventType.EVALUATION_OBSERVED for event in events) == 8
    assert sum(event.event_type == EventType.EVIDENCE_VERIFIED for event in events) == 8
    assert sum(event.event_type == EventType.OPTUNA_TRIAL_REPORTED for event in events) == 8
    assert events[-1].event_type == EventType.OPTUNA_STUDY_COMPLETED


@pytest.mark.parametrize(
    "interrupt_node",
    [
        "optuna_prepare_study",
        "optuna_ask_trial",
        "evaluate_experiment",
        "optuna_tell_trial",
    ],
)
def test_sqlite_process_reconstruction_is_replay_safe(tmp_path, interrupt_node):
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPTUNA}),
        maximum_experiments=3,
    )
    context = TaskRuntimeContext(
        run_id="resume-run",
        output_dir=tmp_path / "output",
        manifest_created_at=FIXED_TIME,
    )
    paths = (
        tmp_path / "checkpoints.sqlite",
        tmp_path / "provenance.sqlite",
        tmp_path / "optuna.sqlite",
    )
    with task_sqlite_dependencies(
        SyntheticTask(),
        context,
        contract,
        {"trial_budget": 3, "seed": 99},
        *paths,
        search_type=SearchType.OPTUNA,
        clock=lambda: FIXED_TIME,
    ) as dependencies:
        partial = invoke(
            build_graph(dependencies, interrupt_after=[interrupt_node]),
            contract,
            "resume-run",
            "resume-thread",
        )
        first_reference = partial["optuna_study_state"].current_trial
        first_number = first_reference.trial_number if first_reference else None
        if interrupt_node in {"optuna_prepare_study", "optuna_ask_trial"}:
            assert partial["budget"].experiments_used == 0

    with task_sqlite_dependencies(
        SyntheticTask(),
        context,
        contract,
        {"trial_budget": 3, "seed": 99},
        *paths,
        search_type=SearchType.OPTUNA,
        clock=lambda: FIXED_TIME,
    ) as dependencies:
        final = build_graph(dependencies).invoke(
            None,
            {"configurable": {"thread_id": "resume-thread"}},
        )
        assert final["optuna_study_result"].trials_asked == 3
        outcomes = dependencies.optuna_backend.trial_outcomes(
            final["optuna_study_result"].study_name
        )
        if first_number is not None:
            assert outcomes[0].trial_number == first_number
        events = dependencies.provenance_store.list_events("resume-run")
        assert len({event.event_id for event in events}) == len(events)


def icca_contract() -> ResearchContract:
    return ResearchContract(
        contract_id="icca-optuna",
        schema_version="1.0",
        task_id="icca_nbs",
        task_version="1.0",
        objective_version="0.9",
        primary_metric="stability_objective",
        task_constraints_version="0.9",
        question="Which bounded configuration is eligible?",
        objective="maximise imported stability objective",
        constraints={},
        allowed_search_types=frozenset({SearchType.OPTUNA}),
        evaluator_id="icca-nbs-v2-evaluator",
        verifier_id="deterministic-verifier",
        maximum_cycles=1,
        maximum_experiments=3,
        maximum_cost=1.0,
        provenance=ProvenanceKind.REAL,
    )


def test_fake_icca_uses_same_optuna_nodes_and_imported_bounds(tmp_path):
    for filename in ("Combined_binary_matrix.csv", "Combined_clinical.csv"):
        (tmp_path / filename).write_text("fake\n", encoding="utf-8")
    bindings, calls = make_fake_icca_bindings()
    task = ICCANBSTask(bindings)
    contract = icca_contract()
    context = TaskRuntimeContext(
        run_id="icca-optuna",
        data_dir=tmp_path,
        workspace_dir=tmp_path,
        manifest_created_at=FIXED_TIME,
    )
    proposal = {
        "trial_budget": 3,
        "seed": 5,
        "fixed": {"network": "ideker", "alignment": "intersect", "r": 10},
    }
    dependencies = task_memory_dependencies(
        task,
        context,
        contract,
        proposal,
        search_type=SearchType.OPTUNA,
        clock=lambda: FIXED_TIME,
    )
    final = invoke(
        build_graph(dependencies),
        contract,
        "icca-optuna",
        "icca-optuna-thread",
    )
    spec = final["optuna_study_spec"]
    assert [(item.name, item.low, item.high) for item in spec.parameters] == [
        ("alpha", 0.3, 0.9),
        ("K", 4, 8),
    ]
    assert spec.fixed_configuration == {
        "network": "Ideker",
        "alignment": "Intersect",
        "r": 10,
    }
    assert calls["evaluate"] == 3
    assert calls["objective"] == 3
    assert final["executed_nodes"].count("optuna_ask_trial") == 3
    synthetic_contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPTUNA}),
        maximum_experiments=3,
    )
    synthetic_dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(manifest_created_at=FIXED_TIME),
        synthetic_contract,
        {"trial_budget": 3, "seed": 5},
        search_type=SearchType.OPTUNA,
        clock=lambda: FIXED_TIME,
    )
    synthetic_final = invoke(
        build_graph(synthetic_dependencies),
        synthetic_contract,
        "synthetic-topology",
        "synthetic-topology-thread",
    )
    def lifecycle(result):
        return [
            node
            for node in result["executed_nodes"]
            if node.startswith("optuna_")
            or node in {"evaluate_experiment", "verify_evidence"}
        ]
    assert lifecycle(final) == lifecycle(synthetic_final)
    assert set(build_graph(dependencies).get_graph().nodes) == set(
        build_graph(synthetic_dependencies).get_graph().nodes
    )


def test_generic_optuna_nodes_contain_no_scientific_domain_conditionals():
    source = inspect.getsource(optuna_nodes).casefold()
    for domain_term in (
        "icca",
        "network based stratification",
        "survival",
        "gene",
        "mri",
        "dice",
        "pytorch",
    ):
        assert domain_term not in source


def test_all_failed_study_clears_primary_and_diagnostic_results():
    class AlwaysFailEvaluator:
        evaluator_id = "synthetic-evaluator"
        cost_per_experiment = 0.0

        def evaluate(self, experiment, contract):
            return EvaluationResult(
                experiment_id=experiment.experiment_id,
                success=False,
                primary_score=None,
                metrics={},
                constraint_results={},
                evaluator_version="always-fail-v1",
                provenance=ProvenanceKind.SIMULATED,
                error="intentional_test_failure",
            )

    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPTUNA}),
        maximum_experiments=3,
    )
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(manifest_created_at=FIXED_TIME),
        contract,
        {"trial_budget": 3, "seed": 19},
        search_type=SearchType.OPTUNA,
        evaluator=AlwaysFailEvaluator(),
        clock=lambda: FIXED_TIME,
    )
    final = invoke(
        build_graph(dependencies),
        contract,
        "all-failed",
        "all-failed-thread",
    )
    assert final["optuna_study_result"].trials_completed == 0
    assert final["optuna_study_result"].trials_failed == 3
    for field in (
        "experiment_spec",
        "evaluation_result",
        "verification_result",
        "diagnostic_experiment_spec",
        "diagnostic_evaluation_result",
        "diagnostic_verification_result",
    ):
        assert final[field] is None
