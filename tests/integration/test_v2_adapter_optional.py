from __future__ import annotations

import importlib.util
from datetime import UTC, datetime

import pytest

from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import ExperimentSpec, ResearchContract
from auto_researcher.tasks.icca_nbs.bindings import load_installed_icca_bindings
from auto_researcher.tasks.icca_nbs.configuration import ICCADirectConfiguration
from auto_researcher.tasks.icca_nbs.evaluator_adapter import ICCANBSEvaluatorAdapter
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)


@pytest.mark.v2
def test_v2_adapter_boundary_when_reference_package_is_available():
    if importlib.util.find_spec("harness") is None:
        pytest.skip("auto_agent_v2 is not installed; adapter remains an optional boundary")
    import numpy as np
    from harness.evaluator.evaluator import EvaluationResult as V2EvaluationResult
    from harness.v2.search import stability_objective

    bindings = load_installed_icca_bindings()
    manifest = DatasetManifest(
        task_id="icca_nbs",
        dataset_version="icca-nbs:installed-contract",
        files=("Combined_binary_matrix.csv", "Combined_clinical.csv"),
        hashes={"Combined_binary_matrix.csv": "a", "Combined_clinical.csv": "b"},
        loader_version=f"harness-{bindings.package_version}",
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
        metadata={"objective_version": "0.9"},
    )
    metadata = ExperimentMetadata(
        evaluator_id="icca-nbs-v2-evaluator",
        code_version=bindings.code_version,
        dataset_version=manifest.dataset_version,
        provenance=ProvenanceKind.REAL,
    )
    adapter = ICCANBSEvaluatorAdapter(
        bindings,
        TaskRuntimeContext(),
        metadata,
        manifest,
    )
    configuration = ICCADirectConfiguration(
        network="Ideker",
        alignment="Intersect",
        alpha=0.7,
        K=5,
        r=10,
    )
    experiment = ExperimentSpec(
        experiment_id="installed-v2-contract",
        hypothesis_id="hypothesis",
        search_request_id="request",
        configuration=configuration.model_dump(mode="json"),
        evaluator_id=metadata.evaluator_id,
        code_version=metadata.code_version,
        dataset_version=metadata.dataset_version,
        provenance=metadata.provenance,
    )
    contract = ResearchContract(
        contract_id="installed-v2-contract",
        schema_version="1.0",
        task_id="icca_nbs",
        task_version="1.0",
        objective_version="0.9",
        primary_metric="stability_objective",
        task_constraints_version="0.9",
        question="Does the mapped result retain v2 semantics?",
        objective="maximise the imported v2 stability objective",
        constraints={},
        allowed_search_types=frozenset({SearchType.DIRECT}),
        evaluator_id=metadata.evaluator_id,
        verifier_id="deterministic-verifier",
        maximum_cycles=1,
        maximum_experiments=1,
        maximum_cost=1.0,
        provenance=ProvenanceKind.REAL,
    )
    v2_result = V2EvaluationResult(
        selected_k=5,
        eligible=True,
        pac_curve={5: np.float64(0.2)},
        eligibility={
            "logrank_pass": np.bool_(True),
            "clinical_pass": np.bool_(True),
            "floors_pass": np.bool_(True),
            "eligible": np.bool_(True),
            "diagnostic_only": np.bool_(False),
        },
        metrics={"pac": np.float64(0.2), "c_index": {"cv": np.float64(0.71)}},
        per_cluster={1: {"size": np.int64(55), "events": np.int64(20)}},
        selection_inputs={"pac": np.float64(0.2), "promising": np.bool_(True)},
        provenance="REAL",
    )

    result = adapter.map_evaluation(experiment, configuration, v2_result, contract)

    assert bindings.stability_objective is stability_objective
    assert result.primary_score == pytest.approx(stability_objective(v2_result))
    assert result.metrics["selection_inputs"]["pac"] == pytest.approx(0.2)
    assert result.metrics["scientific"]["c_index"]["cv"] == pytest.approx(0.71)
    assert result.constraint_results == {
        "logrank_pass": True,
        "clinical_pass": True,
        "floors_pass": True,
    }
    assert result.model_dump_json()
