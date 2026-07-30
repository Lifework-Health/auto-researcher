from __future__ import annotations

import os
from pathlib import Path

import pytest

from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import ExperimentSpec, ResearchContract
from auto_researcher.tasks.icca_nbs.bindings import load_installed_icca_bindings
from auto_researcher.tasks.icca_nbs.configuration import resolve_enum_alias
from auto_researcher.tasks.icca_nbs.task import ICCANBSTask
from auto_researcher.tasks.models import TaskRuntimeContext


@pytest.mark.real_data
def test_icca_plugin_matches_direct_v2_evaluation_when_explicitly_configured():
    data_value = os.environ.get("AUTO_RESEARCHER_ICCA_DATA_DIR")
    workspace_value = os.environ.get("AUTO_RESEARCHER_ICCA_WORKSPACE_DIR")
    if not data_value or not workspace_value:
        pytest.skip(
            "set AUTO_RESEARCHER_ICCA_DATA_DIR and "
            "AUTO_RESEARCHER_ICCA_WORKSPACE_DIR to run the real-data gate"
        )

    data_dir = Path(data_value)
    workspace_dir = Path(workspace_value)
    configuration = {
        "network": os.environ.get("AUTO_RESEARCHER_ICCA_NETWORK", "Ideker"),
        "alignment": os.environ.get("AUTO_RESEARCHER_ICCA_ALIGNMENT", "Intersect"),
        "alpha": float(os.environ.get("AUTO_RESEARCHER_ICCA_ALPHA", "0.7")),
        "K": int(os.environ.get("AUTO_RESEARCHER_ICCA_K", "5")),
        "r": int(os.environ.get("AUTO_RESEARCHER_ICCA_R", "10")),
    }
    context = TaskRuntimeContext(
        run_id="real-data-compatibility",
        data_dir=data_dir,
        workspace_dir=workspace_dir,
    )
    bindings = load_installed_icca_bindings()
    task = ICCANBSTask(bindings)
    readiness = task.readiness(context)
    assert readiness.ready, "; ".join(readiness.errors)
    normalised = task.normalise_configuration(configuration)
    metadata = task.experiment_metadata(context)
    contract = ResearchContract(
        contract_id="icca-real-data-compatibility",
        schema_version="1.0",
        task_id="icca_nbs",
        task_version="1.0",
        objective_version="0.9",
        primary_metric="stability_objective",
        task_constraints_version="0.9",
        question="Does the task adapter preserve direct v2 evaluation semantics?",
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
    experiment = ExperimentSpec(
        experiment_id="real-data-compatibility",
        hypothesis_id="hypothesis",
        search_request_id="request",
        configuration=normalised,
        evaluator_id=metadata.evaluator_id,
        code_version=metadata.code_version,
        dataset_version=metadata.dataset_version,
        provenance=metadata.provenance,
    )

    cohort = bindings.load_cohort(data_dir, verbose=False)
    paths = bindings.harness_paths_factory(workspace_dir)
    cache = bindings.propagation_cache_factory(paths)
    network = resolve_enum_alias(bindings.network_type, normalised["network"])
    alignment = resolve_enum_alias(bindings.alignment_type, normalised["alignment"])
    propagated = cache.get(
        cohort.mutations,
        network,
        alignment,
        normalised["alpha"],
    )
    direct_config_evaluation = bindings.evaluate(
        propagated.matrix,
        propagated.patient_ids,
        cohort,
        k_values=[normalised["K"]],
        r=normalised["r"],
        config={
            "network": normalised["network"],
            "alignment": normalised["alignment"],
            "alpha": normalised["alpha"],
        },
    )
    direct = direct_config_evaluation.per_k[normalised["K"]]
    adapted = task.create_evaluator(context).evaluate(experiment, contract)

    assert adapted.success is True
    assert direct.selected_k == normalised["K"]
    assert adapted.primary_score == pytest.approx(
        bindings.stability_objective(direct),
        abs=1e-12,
    )
    assert adapted.metrics["selection_inputs"]["pac"] == pytest.approx(
        float(direct.selection_inputs["pac"]),
        abs=1e-12,
    )
    assert adapted.metrics["eligibility"] == direct.eligibility
    assert adapted.constraint_results == {
        gate: bool(direct.eligibility[gate])
        for gate in ("logrank_pass", "clinical_pass", "floors_pass")
    }
    assert adapted.metrics["scientific"] == direct.metrics
