from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import ExperimentSpec, ResearchContract
from auto_researcher.tasks.icca_nbs.configuration import ICCADirectConfiguration
from auto_researcher.tasks.icca_nbs.evaluator_adapter import ICCANBSEvaluatorAdapter
from auto_researcher.tasks.icca_nbs.manifests import build_icca_dataset_manifest
from auto_researcher.tasks.icca_nbs.task import ICCANBSTask
from auto_researcher.tasks.models import TaskRuntimeContext
from tests.fakes_icca import make_fake_icca_bindings


def icca_contract() -> ResearchContract:
    return ResearchContract(
        contract_id="icca-test",
        schema_version="1.0",
        task_id="icca_nbs",
        task_version="1.0",
        objective_version="0.9",
        primary_metric="stability_objective",
        task_constraints_version="0.9",
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
        configuration=configuration or task.normalise_configuration(valid_configuration()),
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


def test_objective_score_is_taken_from_binding_not_recomputed(icca_context):
    bindings, calls = make_fake_icca_bindings()

    def registered_objective(result):
        calls["objective"] += 1
        return 0.123456

    task = ICCANBSTask(
        replace(bindings, stability_objective=registered_objective)
    )
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
    assert result.error == "icca_evaluation_failed: RuntimeError"


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
