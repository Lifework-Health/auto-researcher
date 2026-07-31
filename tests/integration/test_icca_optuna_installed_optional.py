from __future__ import annotations

import importlib.util

import pytest

from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import ResearchContract, SearchRequest
from auto_researcher.tasks.icca_nbs import ICCANBSTask

pytestmark = pytest.mark.v2


@pytest.mark.skipif(
    importlib.util.find_spec("harness") is None,
    reason="auto_agent_v2 is not installed",
)
def test_installed_v2_exposes_reference_optuna_bounds_without_patient_data():
    task = ICCANBSTask()
    contract = ResearchContract(
        contract_id="installed-v2-optuna",
        schema_version="1.0",
        task_id="icca_nbs",
        task_version="1.0",
        objective_version="0.9",
        primary_metric="stability_objective",
        task_constraints_version="1.0",
        question="Validate installed search semantics.",
        objective="maximise stability objective",
        constraints={},
        allowed_search_types=frozenset({SearchType.OPTUNA}),
        evaluator_id="icca-nbs-v2-evaluator",
        verifier_id="deterministic-verifier",
        maximum_cycles=1,
        maximum_experiments=2,
        maximum_cost=1,
        provenance=ProvenanceKind.REAL,
    )
    request = SearchRequest(
        request_id="installed-v2-request",
        hypothesis_id="installed-v2-hypothesis",
        search_type=SearchType.OPTUNA,
        target="validate",
        search_space={
            "trial_budget": 2,
            "fixed": {
                "network": "Ideker",
                "alignment": "Intersect",
                "r": 10,
            },
        },
        experiment_budget=2,
        rationale="contract test",
    )
    spec = task.create_optuna_study_spec(contract, request)
    assert [(item.name, item.low, item.high) for item in spec.parameters] == [
        ("alpha", 0.3, 0.9),
        ("K", 4, 8),
    ]
