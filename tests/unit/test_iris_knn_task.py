from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import pytest

from auto_researcher.contracts.enums import EvidenceStatus, SearchType
from auto_researcher.contracts.models import ExperimentSpec, SearchRequest
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.execution import start_run
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.mutation import DeterministicMutationOperator
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
from auto_researcher.search.openevolve.upstream import mutation_constraints
from auto_researcher.search.openevolve.validation import validate_candidate
from auto_researcher.tasks.artifacts import verify_artefact_bundle
from auto_researcher.tasks.iris_knn import (
    IrisKNNConfiguration,
    IrisKNNTask,
    default_iris_configuration,
    default_iris_contract,
)
from auto_researcher.tasks.iris_knn.configuration import configuration_schema
from auto_researcher.tasks.iris_knn.evaluator import (
    balanced_accuracy,
    choose_label,
    evaluate_configuration,
    fit_standardisation,
)
from auto_researcher.tasks.iris_knn.manifests import (
    CLASS_NAMES,
    DATA_SHA256,
    FOLD_SHA256,
    FOLD_VERSION,
    IrisRow,
    load_fold_assignments,
    load_iris_rows,
)
from auto_researcher.tasks.iris_knn.openevolve import (
    IrisKNNEvolvableComponent,
    default_iris_openevolve_configuration,
)
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.registry import default_task_registry

FIXED_TIME = datetime(2026, 8, 7, 12, tzinfo=UTC)


def _context(tmp_path, run_id="iris-unit"):
    return TaskRuntimeContext(
        run_id=run_id,
        output_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
        manifest_created_at=FIXED_TIME,
    )


def _experiment(task, context, configuration, experiment_id="iris-experiment"):
    metadata = task.experiment_metadata(context)
    return ExperimentSpec(
        experiment_id=experiment_id,
        hypothesis_id="iris-hypothesis",
        search_request_id="iris-request",
        configuration=task.normalise_configuration(configuration),
        evaluator_id=metadata.evaluator_id,
        code_version=metadata.code_version,
        dataset_version=metadata.dataset_version,
        provenance=metadata.provenance,
    )


def _openevolve_backend(task, context):
    contract = default_iris_contract(
        search_types=frozenset({SearchType.OPENEVOLVE}), maximum_experiments=3
    )
    dependencies = task_memory_dependencies(
        task,
        context,
        contract,
        default_iris_openevolve_configuration(),
        search_type=SearchType.OPENEVOLVE,
    )
    assert dependencies.openevolve_backend is not None
    existing = dependencies.openevolve_backend
    return (
        OpenEvolveBackend(
            existing.component,
            existing.metadata,
            existing.verifier_identity,
            DeterministicMutationOperator(),
            LocalSandboxRunner(context.workspace_dir),
        ),
        contract,
    )


def _openevolve_request():
    return SearchRequest(
        request_id="iris-openevolve-request",
        hypothesis_id="iris-openevolve-hypothesis",
        search_type=SearchType.OPENEVOLVE,
        target="bounded Iris weighted k-NN configuration",
        search_space=default_iris_openevolve_configuration(),
        experiment_budget=3,
        rationale="offline deterministic benchmark",
    )


def test_manifest_hashes_shape_and_fixed_stratified_folds(tmp_path):
    task = IrisKNNTask()
    manifest = task.dataset_manifest(_context(tmp_path))
    rows = load_iris_rows()
    assignments = load_fold_assignments()

    assert manifest.hashes["bezdekIris.data"] == DATA_SHA256
    assert manifest.hashes["folds-v1.json"] == FOLD_SHA256
    assert manifest.metadata["fold_version"] == FOLD_VERSION
    assert len(rows) == len(assignments) == 150
    assert {name: sum(row.label == name for row in rows) for name in CLASS_NAMES} == {
        name: 50 for name in CLASS_NAMES
    }
    for fold in range(5):
        indices = [
            index for index, assigned in enumerate(assignments) if assigned == fold
        ]
        assert len(indices) == len(set(indices)) == 30
        assert {
            name: sum(rows[index].label == name for index in indices)
            for name in CLASS_NAMES
        } == {name: 10 for name in CLASS_NAMES}


@pytest.mark.parametrize(
    "configuration",
    [
        {"feature_weights": [1, 1, 1], "k": 3, "distance_power": 2},
        {"feature_weights": [0, 1, 1, 1], "k": 3, "distance_power": 2},
        {"feature_weights": [1, 1, 1, math.inf], "k": 3, "distance_power": 2},
        {"feature_weights": [1, 1, 1, 1], "k": 2, "distance_power": 2},
        {"feature_weights": [1, 1, 1, 1], "k": 3, "distance_power": 3},
    ],
)
def test_configuration_rejects_invalid_or_out_of_range_values(configuration):
    with pytest.raises(ValueError):
        IrisKNNConfiguration.model_validate(configuration)


def test_evaluator_is_deterministic_plausible_and_publishes_valid_bundle(tmp_path):
    task = IrisKNNTask()
    contract = default_iris_contract()
    context = _context(tmp_path)
    experiment = _experiment(task, context, default_iris_configuration())
    evaluator = task.create_evaluator(context)

    first = evaluator.evaluate(experiment, contract)
    second = evaluator.evaluate(experiment, contract)

    assert first == second
    assert first.success is True
    assert first.primary_score == 0.94
    assert 0.8 < first.primary_score <= 1.0
    assert len(first.metrics["per_fold_balanced_accuracy"]) == 5
    integrity = verify_artefact_bundle(context, experiment.experiment_id)
    assert integrity.complete is True
    assert integrity.untampered is True


def test_metric_arithmetic_label_permutation_and_pathological_weights():
    assert (
        balanced_accuracy(
            ["a", "a", "b", "b"],
            ["a", "b", "b", "b"],
            ("a", "b"),
        )
        == 0.75
    )
    rows = load_iris_rows()
    assignments = load_fold_assignments()
    baseline = evaluate_configuration(
        IrisKNNConfiguration.model_validate(default_iris_configuration()),
        rows,
        assignments,
    )["mean_balanced_accuracy"]
    pathological = evaluate_configuration(
        IrisKNNConfiguration.model_validate(
            {"feature_weights": [4.0, 4.0, 0.1, 0.1], "k": 3, "distance_power": 2}
        ),
        rows,
        assignments,
    )["mean_balanced_accuracy"]
    actual = [row.label for row in rows]
    permuted = [CLASS_NAMES[(CLASS_NAMES.index(label) + 1) % 3] for label in actual]
    assert balanced_accuracy(actual, permuted) == 0.0
    assert pathological == 0.833333333333
    assert pathological != baseline


def test_preprocessing_fits_training_only_and_zero_variance_is_safe():
    training = (
        IrisRow(0, (1.0, 2.0, 5.0, 8.0), CLASS_NAMES[0]),
        IrisRow(1, (3.0, 4.0, 5.0, 10.0), CLASS_NAMES[1]),
    )
    validation = IrisRow(2, (1000.0, -500.0, 5.0, 20.0), CLASS_NAMES[2])
    before = fit_standardisation(training)
    after = fit_standardisation(training)
    assert before == after
    assert before[0] == (2.0, 3.0, 5.0, 9.0)
    assert before[1][2] == 1.0
    assert validation.features not in tuple(row.features for row in training)


def test_knn_tie_break_is_vote_then_distance_then_canonical_class():
    assert (
        choose_label(((0.4, 1, CLASS_NAMES[1]), (0.2, 2, CLASS_NAMES[2])))
        == CLASS_NAMES[2]
    )
    assert (
        choose_label(((0.2, 1, CLASS_NAMES[1]), (0.2, 2, CLASS_NAMES[2])))
        == CLASS_NAMES[1]
    )


def test_verification_checks_registered_evidence_identity(tmp_path):
    task = IrisKNNTask()
    contract = default_iris_contract()
    context = _context(tmp_path)
    evaluation = task.create_evaluator(context).evaluate(
        _experiment(task, context, default_iris_configuration()), contract
    )
    decision = task.create_verification_policy(contract).evaluate_constraints(
        evaluation, contract
    )
    assert decision.constraint_compliant is True
    assert decision.evidence_status == EvidenceStatus.SUPPORTED
    changed = evaluation.model_copy(
        update={"metrics": {**evaluation.metrics, "fold_version": "changed"}}
    )
    assert (
        task.create_verification_policy(contract)
        .evaluate_constraints(changed, contract)
        .constraint_compliant
        is False
    )


def test_registry_and_three_search_modes_share_scientific_identity(tmp_path):
    task = default_task_registry().get("iris_knn", "1.0")
    descriptor = task.descriptor()
    assert descriptor.supported_search_types == {
        SearchType.DIRECT,
        SearchType.OPTUNA,
        SearchType.OPENEVOLVE,
    }
    contracts = [
        default_iris_contract(search_types=frozenset({search_type}))
        for search_type in SearchType
        if search_type in descriptor.supported_search_types
    ]
    assert len({item.task_version for item in contracts}) == 1
    assert len({item.primary_metric for item in contracts}) == 1
    assert len({item.evaluator_id for item in contracts}) == 1
    assert len({item.task_constraints_version for item in contracts}) == 1
    assert (
        len({task.create_verification_policy(item).policy_id for item in contracts})
        == 1
    )
    manifests = [
        task.dataset_manifest(_context(tmp_path, f"iris-fairness-{index}"))
        for index, _ in enumerate(contracts)
    ]
    assert len({manifest.dataset_version for manifest in manifests}) == 1
    assert len({tuple(manifest.hashes.items()) for manifest in manifests}) == 1
    assert len({manifest.metadata["fold_version"] for manifest in manifests}) == 1
    assert (
        len(
            {
                task.experiment_metadata(
                    _context(tmp_path, f"iris-meta-{index}")
                ).code_version
                for index, _ in enumerate(contracts)
            }
        )
        == 1
    )

    optuna_contract = default_iris_contract(
        search_types=frozenset({SearchType.OPTUNA}), maximum_experiments=20
    )
    optuna_request = SearchRequest(
        request_id="iris-fairness-optuna",
        hypothesis_id="iris-fairness-hypothesis",
        search_type=SearchType.OPTUNA,
        target="bounded Iris weighted k-NN configuration",
        search_space={},
        experiment_budget=20,
        rationale="verify registered search-space fairness",
    )
    study = task.create_optuna_study_spec(optuna_contract, optuna_request)
    parameters = {parameter.name: parameter for parameter in study.parameters}
    schema = configuration_schema()
    for index in range(4):
        parameter = parameters[f"feature_weight_{index}"]
        assert (parameter.low, parameter.high) == (
            schema["feature_weights"]["items"]["minimum"],
            schema["feature_weights"]["items"]["maximum"],
        )
    assert list(parameters["k"].choices) == schema["k"]["enum"]
    assert (
        list(parameters["distance_power"].choices) == schema["distance_power"]["enum"]
    )
    assert IrisKNNEvolvableComponent().component_spec().parameter_schema == schema


def test_mutation_context_contains_bounds_but_no_rows_labels_folds_or_results():
    component = IrisKNNEvolvableComponent().component_spec()
    constraints = mutation_constraints(component).model_dump(mode="json")
    rendered = json.dumps(
        {"constraints": constraints, "context": component.task_mutation_context},
        sort_keys=True,
    )
    assert constraints["mutable_file"] == "candidate.py"
    assert constraints["allowed_files"] == ["candidate.py"]
    assert constraints["allowed_imports_display"] == "NONE"
    assert constraints["allowed_dependencies_display"] == "NONE"
    assert "sepal_length_cm" in rendered
    for forbidden in (
        "5.1,3.5,1.4",
        "folds-v1",
        "aggregate_confusion_counts",
        "Iris-setosa\n",
    ):
        assert forbidden not in rendered


def test_iris_candidate_static_validation_sandbox_and_experiment_conversion(tmp_path):
    task = IrisKNNTask()
    context = _context(tmp_path, "iris-candidate")
    backend, contract = _openevolve_backend(task, context)
    request = _openevolve_request()
    search = backend.create_search_contract(request, contract)
    candidate = backend.seed_candidate(search)

    validation = validate_candidate(candidate, backend.component_spec)
    assert validation.status.value == "VALID"
    preparation = backend.prepare(candidate, search)
    assert preparation.execution_status.value == "COMPLETED"
    assert preparation.generated_configuration == {
        "feature_weights": [1.0, 1.0, 1.0, 1.0],
        "k": 3,
        "distance_power": 2,
    }
    experiment = backend.component.candidate_to_experiment(
        candidate,
        preparation,
        request,
        contract,
        task.experiment_metadata(context),
        run_id=context.run_id,
    )
    assert experiment.configuration == task.normalise_configuration(
        default_iris_configuration()
    )
    assert experiment.evaluator_id == task.descriptor().evaluator_id
    assert list(context.workspace_dir.iterdir()) == []

    hostile = candidate.model_copy(
        update={"source_payload": "import os\ndef evolve(configuration):\n return {}\n"}
    )
    assert validate_candidate(hostile, backend.component_spec).status.value == "INVALID"


def test_direct_graph_reuse_and_generation_zero_match(tmp_path):
    task = IrisKNNTask()
    direct_contract = default_iris_contract()
    context = _context(tmp_path, "iris-direct")
    dependencies = task_memory_dependencies(
        task, context, direct_contract, default_iris_configuration()
    )
    initial = {
        "run_id": "iris-direct",
        "thread_id": "iris-direct-thread",
        "contract": direct_contract,
    }
    graph = build_graph(dependencies)
    first = start_run(
        graph, initial, {"configurable": {"thread_id": "iris-direct-thread"}}
    )
    second = graph.invoke(
        initial, {"configurable": {"thread_id": "iris-direct-thread"}}
    )
    assert first["evaluation_result"].primary_score == 0.94
    assert second["evaluation_result"] == first["evaluation_result"]
    assert "evaluate_experiment_reused" in second["executed_nodes"]
    record = dependencies.provenance_store.get_evaluation_reuse(
        "iris-direct", first["experiment_spec"].experiment_id
    )
    assert record is not None and record.protocol_version == "evaluation-reuse-v2"

    for index, update in enumerate(
        (
            {"dataset_version": "changed-dataset"},
            {"code_version": "changed-evaluator"},
            {"evaluator_id": "changed-evaluator"},
        )
    ):
        incompatible = first["experiment_spec"].model_copy(
            update={"experiment_id": f"incompatible-{index}", **update}
        )
        rejected = dependencies.evaluator.evaluate(incompatible, direct_contract)
        assert rejected.success is False
        assert rejected.error == "experiment_metadata_mismatch"

    oe_context = _context(tmp_path / "oe", "iris-oe-seed")
    backend, _ = _openevolve_backend(task, oe_context)
    assert backend.component.seed_configuration() == default_iris_configuration()
