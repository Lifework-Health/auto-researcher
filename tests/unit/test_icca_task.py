from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import (
    ExperimentSpec,
    ResearchContract,
    SearchRequest,
)
from auto_researcher.search.protocols import SearchCapability
from auto_researcher.tasks.icca_nbs.configuration import (
    ICCA_DEFAULT_RESAMPLING_ITERATIONS,
    ICCA_MINIMUM_RESAMPLING_ITERATIONS,
    ICCADirectConfiguration,
)
from auto_researcher.tasks.icca_nbs.diagnostics import (
    ICCAEvaluationFailureStage,
    classify_scientific_failure,
)
from auto_researcher.tasks.icca_nbs.manifests import build_icca_dataset_manifest
from auto_researcher.tasks.icca_nbs.task import ICCANBSTask
from auto_researcher.tasks.artifacts import verify_artefact_bundle
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.verification.verifier import DeterministicVerifier
from tests.fakes_icca import make_fake_icca_bindings


def icca_contract() -> ResearchContract:
    return ResearchContract(
        contract_id="icca-test",
        schema_version="1.0",
        task_id="icca_nbs",
        task_version="1.0",
        objective_version="0.9",
        primary_metric="stability_objective",
        task_constraints_version="1.0",
        question="Can the requested iCCA configuration satisfy eligibility?",
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


@pytest.fixture
def icca_context(tmp_path):
    (tmp_path / "Combined_binary_matrix.csv").write_text(
        "PatID,G1\nsecret-patient,1\n", encoding="utf-8"
    )
    (tmp_path / "Combined_clinical.csv").write_text(
        "PatID,OS_MONTHS\nsecret-patient,12\n", encoding="utf-8"
    )
    return TaskRuntimeContext(
        run_id="icca-run",
        data_dir=tmp_path,
        workspace_dir=tmp_path,
        output_dir=tmp_path / "outputs",
        task_options={"objective_version": "0.9"},
        manifest_created_at=datetime(2026, 7, 30, tzinfo=UTC),
    )


def valid_configuration():
    return {
        "network": "ideker",
        "alignment": "intersect",
        "alpha": 0.7,
        "K": 5,
        "r": 10,
    }


def test_valid_configuration_and_aliases_normalise():
    bindings, _ = make_fake_icca_bindings()
    config = ICCADirectConfiguration.normalise(valid_configuration(), bindings)
    assert config.network == "Ideker"
    assert config.alignment == "Intersect"


@pytest.mark.parametrize("r", [1, 9])
def test_consensus_resampling_below_minimum_is_rejected(r):
    bindings, _ = make_fake_icca_bindings()
    with pytest.raises(
        ValueError,
        match="r must be at least 10 consensus resampling iterations",
    ):
        ICCADirectConfiguration.normalise(
            {**valid_configuration(), "r": r},
            bindings,
        )


@pytest.mark.parametrize("r", [10, 100])
def test_consensus_resampling_minimum_and_recommended_values_are_accepted(r):
    bindings, _ = make_fake_icca_bindings()
    configuration = ICCADirectConfiguration.normalise(
        {**valid_configuration(), "r": r},
        bindings,
    )
    assert configuration.r == r


def test_consensus_resampling_defaults_to_recommended_value():
    bindings, _ = make_fake_icca_bindings()
    proposed = valid_configuration()
    proposed.pop("r")
    configuration = ICCADirectConfiguration.normalise(proposed, bindings)
    assert configuration.r == ICCA_DEFAULT_RESAMPLING_ITERATIONS == 100


def test_agent_context_advertises_one_authoritative_resampling_policy(icca_context):
    bindings, _ = make_fake_icca_bindings()
    task = ICCANBSTask(bindings)
    context = task.create_agent_context(
        icca_contract(),
        icca_context,
        {
            SearchType.DIRECT: SearchCapability(
                search_type=SearchType.DIRECT,
                available=True,
                code="available",
                message="DIRECT is available.",
            )
        },
    )
    schema = context.direct_configuration_schema["r"]
    assert schema == {
        "type": "integer",
        "minimum": ICCA_MINIMUM_RESAMPLING_ITERATIONS,
        "default": ICCA_DEFAULT_RESAMPLING_ITERATIONS,
        "recommended": ICCA_DEFAULT_RESAMPLING_ITERATIONS,
    }
    assert any(
        "at least 10" in limitation and "100 is the recommended" in limitation
        for limitation in context.task_limitations
    )


def test_additional_registered_aliases_normalise():
    bindings, _ = make_fake_icca_bindings()
    config = ICCADirectConfiguration.normalise(
        {**valid_configuration(), "network": "OMNI", "alignment": "full"},
        bindings,
    )
    assert config.network == "Omni"
    assert config.alignment == "NetworkZeroPad"


def test_extra_fields_are_forbidden():
    bindings, _ = make_fake_icca_bindings()
    with pytest.raises(ValidationError):
        ICCADirectConfiguration.normalise(
            {**valid_configuration(), "data_path": "/secret"},
            bindings,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("alpha", 0.2, "outside v2 bounds"), ("K", 9, "outside v2 bounds")],
)
def test_registered_bounds_are_enforced(field, value, message):
    bindings, _ = make_fake_icca_bindings()
    configuration = {**valid_configuration(), field: value}
    with pytest.raises(ValueError, match=message):
        ICCADirectConfiguration.normalise(configuration, bindings)


def test_v2_imports_are_lazy():
    script = (
        "import sys; "
        "import auto_researcher.tasks.icca_nbs; "
        "assert 'harness' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_unavailable_v2_readiness_is_actionable(monkeypatch):
    monkeypatch.setattr(
        "auto_researcher.tasks.icca_nbs.task.importlib.util.find_spec",
        lambda name: None,
    )
    result = ICCANBSTask().readiness(TaskRuntimeContext())
    assert result.ready is False
    assert any("pip install -e ../auto_agent_v2" in error for error in result.errors)


def _experiment(task, context, configuration=None):
    metadata = task.experiment_metadata(context)
    return ExperimentSpec(
        experiment_id="experiment-1",
        hypothesis_id="hypothesis-1",
        search_request_id="request-1",
        configuration=configuration
        or task.normalise_configuration(valid_configuration()),
        evaluator_id=metadata.evaluator_id,
        code_version=metadata.code_version,
        dataset_version=metadata.dataset_version,
        provenance=metadata.provenance,
    )


def test_metadata_mismatch_returns_structured_failure(icca_context):
    bindings, calls = make_fake_icca_bindings()
    task = ICCANBSTask(bindings)
    evaluator = task.create_evaluator(icca_context)
    experiment = _experiment(task, icca_context).model_copy(
        update={"dataset_version": "wrong"}
    )
    result = evaluator.evaluate(experiment, icca_contract())
    assert result.success is False
    assert result.error == "experiment_metadata_mismatch"
    assert calls["evaluate"] == 0


def test_invalid_resampling_never_reaches_data_or_scientific_evaluator(icca_context):
    bindings, calls = make_fake_icca_bindings()
    task = ICCANBSTask(bindings)
    experiment = _experiment(task, icca_context).model_copy(
        update={"configuration": {**valid_configuration(), "r": 1}}
    )

    result = task.create_evaluator(icca_context).evaluate(
        experiment,
        icca_contract(),
    )

    assert result.success is False
    assert result.error == (
        "icca_evaluation_failed: CONFIGURATION_VALIDATION: ValidationError"
    )
    assert calls["load_cohort"] == 0
    assert calls["cache_get"] == 0
    assert calls["evaluate"] == 0


@pytest.mark.parametrize("eligible", [True, False])
def test_v2_result_mapping_objective_metrics_and_constraints(
    eligible,
    icca_context,
):
    bindings, calls = make_fake_icca_bindings(eligible=eligible)
    task = ICCANBSTask(bindings)
    result = task.create_evaluator(icca_context).evaluate(
        _experiment(task, icca_context),
        icca_contract(),
    )
    assert result.success is True
    assert result.primary_score == (0.8 if eligible else -0.2)
    assert result.metrics["scientific"]["c_index"]["cv"] == 0.71
    assert result.metrics["scientific"]["status"] == "complete"
    assert result.metrics["selection_inputs"]["pac"] == 0.2
    assert result.metrics["configuration"]["K"] == 5
    assert result.metrics["evaluation_settings"] == {
        "r": 10,
        "objective_version": "0.9",
    }
    assert result.constraint_results == {
        "logrank_pass": eligible,
        "clinical_pass": eligible,
        "floors_pass": eligible,
    }
    assert calls["objective"] == 1
    assert calls["k_values"] == [5]
    payload = result.model_dump_json()
    assert "internal-patient" not in payload
    json.loads(payload)


def test_genuine_shaped_result_normalises_unavailable_c_indexes(icca_context):
    bindings, _ = make_fake_icca_bindings()
    base_evaluate = bindings.evaluate

    def evaluate(*args, **kwargs):
        result = base_evaluate(*args, **kwargs)
        row = result.per_k[kwargs["k_values"][0]]
        row.metrics = {
            "pac": 0.2,
            "logrank": {"chisq": 12.5, "p": 0.01, "passed": True},
            "pairwise_logrank": {
                "fraction_separated": 0.5,
                "n_separated": 2,
                "n_pairs": 4,
            },
            "c_index": {
                "apparent": 0.72,
                "cv": float("nan"),
                "incremental": float("nan"),
            },
            "clinical": {
                "fraction_clusters_associated": 1.0,
                "n_core_associations": 3,
                "n_distinct_core_drivers": 2,
                "promise": {"promising_distinction": True},
            },
            "cluster_sizes": {1: 55, 2: 61},
            "cluster_events": {1: 20, 2: 24},
        }
        row.per_cluster = {
            1: {"size": 55, "events": 20, "has_credit_association": True},
            2: {"size": 61, "events": 24, "has_credit_association": True},
        }
        return result

    task = ICCANBSTask(replace(bindings, evaluate=evaluate))
    experiment = _experiment(task, icca_context)
    result = task.create_evaluator(icca_context).evaluate(
        experiment,
        icca_contract(),
    )

    assert result.success is True
    assert result.primary_score == 0.8
    assert result.metrics["scientific"]["c_index"] == {
        "apparent": 0.72,
        "cv": None,
        "incremental": None,
    }
    assert result.metrics["metric_availability"] == {
        "unavailable_paths": [
            "scientific.c_index.cv",
            "scientific.c_index.incremental",
        ],
        "unavailable_count": 2,
        "encoding": "null_for_unavailable_non_finite_v1",
        "result_encoding_version": "scientific-json-v1",
    }
    assert all(result.constraint_results.values())
    verification = DeterministicVerifier(
        task.create_verification_policy(icca_contract())
    ).verify(experiment, result, icca_contract())
    assert verification.verified is True
    persisted = (icca_context.output_dir / result.artefact_references[1]).read_text(
        encoding="utf-8"
    )
    assert "NaN" not in persisted
    assert "Infinity" not in persisted
    json.loads(persisted, parse_constant=lambda token: pytest.fail(token))
    integrity = verify_artefact_bundle(icca_context, experiment.experiment_id)
    assert integrity.complete and integrity.untampered


def test_unexpected_non_finite_metric_fails_closed_at_normalisation(icca_context):
    bindings, _ = make_fake_icca_bindings()
    base_evaluate = bindings.evaluate

    def evaluate(*args, **kwargs):
        result = base_evaluate(*args, **kwargs)
        result.per_k[kwargs["k_values"][0]].metrics["unexpected"] = float("nan")
        return result

    task = ICCANBSTask(replace(bindings, evaluate=evaluate))
    result = task.create_evaluator(icca_context).evaluate(
        _experiment(task, icca_context),
        icca_contract(),
    )

    assert result.success is False
    diagnostics = result.metrics["failure_diagnostics"]
    assert diagnostics["failure_stage"] == "RESULT_NORMALISATION"
    assert diagnostics["normalisation_reason_code"] == (
        "unexpected_non_finite_scientific_value"
    )
    assert diagnostics["rejected_paths"] == ["scientific.unexpected"]


def test_non_finite_objective_fails_at_objective_calculation(icca_context):
    bindings, _ = make_fake_icca_bindings()
    task = ICCANBSTask(
        replace(bindings, stability_objective=lambda result: float("nan"))
    )
    result = task.create_evaluator(icca_context).evaluate(
        _experiment(task, icca_context),
        icca_contract(),
    )
    assert result.success is False
    diagnostics = result.metrics["failure_diagnostics"]
    assert diagnostics["failure_stage"] == "OBJECTIVE_CALCULATION"
    assert diagnostics["normalisation_reason_code"] == "non_finite_primary_score"


def test_non_finite_constraint_source_cannot_silently_pass(icca_context):
    bindings, _ = make_fake_icca_bindings()
    base_evaluate = bindings.evaluate

    def evaluate(*args, **kwargs):
        result = base_evaluate(*args, **kwargs)
        row = result.per_k[kwargs["k_values"][0]]
        row.eligibility["logrank_pass"] = float("nan")
        return result

    task = ICCANBSTask(replace(bindings, evaluate=evaluate))
    result = task.create_evaluator(icca_context).evaluate(
        _experiment(task, icca_context),
        icca_contract(),
    )
    assert result.success is False
    assert result.constraint_results == {}
    diagnostics = result.metrics["failure_diagnostics"]
    assert diagnostics["failure_stage"] == "RESULT_NORMALISATION"
    assert diagnostics["rejected_paths"] == ["eligibility.logrank_pass"]


def test_artefact_persistence_failure_returns_no_dangling_references(
    icca_context,
    monkeypatch,
):
    bindings, _ = make_fake_icca_bindings()
    task = ICCANBSTask(bindings)

    def fail_bundle(*args, **kwargs):
        raise OSError("private path must not persist")

    monkeypatch.setattr(
        "auto_researcher.tasks.icca_nbs.evaluator_adapter.write_artefact_bundle",
        fail_bundle,
    )
    result = task.create_evaluator(icca_context).evaluate(
        _experiment(task, icca_context),
        icca_contract(),
    )

    assert result.success is False
    assert result.artefact_references == ()
    assert "private path" not in result.model_dump_json()
    diagnostics = result.metrics["failure_diagnostics"]
    assert diagnostics["failure_stage"] == "ARTEFACT_WRITING"
    assert diagnostics["artefact_persistence_failure_code"] == (
        "bundle_publication_failed"
    )


def test_objective_score_is_taken_from_binding_not_recomputed(icca_context):
    bindings, calls = make_fake_icca_bindings()

    def registered_objective(result):
        calls["objective"] += 1
        return 0.123456

    task = ICCANBSTask(replace(bindings, stability_objective=registered_objective))
    result = task.create_evaluator(icca_context).evaluate(
        _experiment(task, icca_context),
        icca_contract(),
    )
    assert result.primary_score == 0.123456
    assert calls["objective"] == 1


def test_structured_scientific_failure(icca_context):
    bindings, _ = make_fake_icca_bindings(fail_evaluation=True)
    task = ICCANBSTask(bindings)
    result = task.create_evaluator(icca_context).evaluate(
        _experiment(task, icca_context),
        icca_contract(),
    )
    assert result.success is False
    assert result.error == "icca_evaluation_failed: UNKNOWN: RuntimeError"
    diagnostics = result.metrics["failure_diagnostics"]
    assert diagnostics == {
        "safe_exception_class": "RuntimeError",
        "failure_stage": "UNKNOWN",
        "evaluator_id": "icca-nbs-v2-evaluator",
        "evaluator_version": "icca-adapter-v1.2",
        "experiment_id": "experiment-1",
        "canonical_configuration": {
            "network": "Ideker",
            "alignment": "Intersect",
            "alpha": 0.7,
            "K": 5,
            "r": 10,
        },
        "dataset_fingerprint": task.dataset_manifest(icca_context).metadata[
            "combined_dataset_fingerprint"
        ],
        "dataset_loading_completed": True,
        "propagation_completed": True,
        "clustering_completed": False,
        "eligibility_evaluation_completed": False,
    }
    persisted = result.model_dump_json()
    assert "fake scientific failure" not in persisted
    assert "Traceback" not in persisted
    assert str(icca_context.data_dir) not in persisted
    assert "secret-patient" not in persisted
    assert "internal-patient" not in persisted
    persisted_artefact = (
        icca_context.output_dir / result.artefact_references[1]
    ).read_text(encoding="utf-8")
    assert "fake scientific failure" not in persisted_artefact
    assert "Traceback" not in persisted_artefact
    assert str(icca_context.data_dir) not in persisted_artefact
    assert "secret-patient" not in persisted_artefact
    assert "internal-patient" not in persisted_artefact


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("harness.evaluator.pac", ICCAEvaluationFailureStage.CONSENSUS_CLUSTERING),
        (
            "harness.evaluator.survival",
            ICCAEvaluationFailureStage.ELIGIBILITY_EVALUATION,
        ),
        ("third_party.unknown", ICCAEvaluationFailureStage.UNKNOWN),
    ],
)
def test_scientific_tracebacks_reduce_to_closed_safe_stages(module, expected):
    namespace = {"__name__": module}
    exec("def fail(): raise ValueError('private patient /absolute/path')", namespace)
    try:
        namespace["fail"]()
    except ValueError as exc:
        assert classify_scientific_failure(exc) == expected
    else:  # pragma: no cover - defensive assertion for the dynamic fixture
        raise AssertionError("failure fixture did not raise")


@pytest.mark.parametrize("r", [9, 100])
def test_optuna_fixed_resampling_uses_direct_configuration_policy(r):
    bindings, _ = make_fake_icca_bindings()
    task = ICCANBSTask(bindings)
    contract = icca_contract().model_copy(
        update={
            "allowed_search_types": frozenset({SearchType.OPTUNA}),
            "maximum_experiments": 2,
        }
    )
    request = SearchRequest(
        request_id=f"optuna-r-{r}",
        hypothesis_id="hypothesis",
        search_type=SearchType.OPTUNA,
        target="stability_objective",
        search_space={
            "trial_budget": 2,
            "fixed": {
                "network": "Ideker",
                "alignment": "Intersect",
                "r": r,
            },
        },
        experiment_budget=2,
        rationale="Validate fixed resampling context.",
    )
    if r < ICCA_MINIMUM_RESAMPLING_ITERATIONS:
        with pytest.raises(ValueError, match="r must be at least 10"):
            task.create_optuna_study_spec(contract, request)
    else:
        specification = task.create_optuna_study_spec(contract, request)
        assert specification.fixed_configuration["r"] == 100
        assert "r" not in {item.name for item in specification.parameters}


def test_dataset_fingerprint_is_deterministic_and_safe(icca_context):
    first = build_icca_dataset_manifest(icca_context, loader_version="fake")
    second = build_icca_dataset_manifest(icca_context, loader_version="fake")
    assert first == second
    payload = first.model_dump_json()
    assert first.dataset_version.startswith("icca-nbs:")
    assert "secret-patient" not in payload
    assert str(icca_context.data_dir) not in payload
    assert first.files == (
        "Combined_binary_matrix.csv",
        "Combined_clinical.csv",
    )
