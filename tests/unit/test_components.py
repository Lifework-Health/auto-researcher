from __future__ import annotations

from auto_researcher.agents.mock import MockHypothesisAgent, MockPlannerAgent
from auto_researcher.contracts.enums import EvidenceStatus, ProvenanceKind, SearchType
from auto_researcher.contracts.models import ExperimentSpec, SearchRequest
from auto_researcher.evaluation.mock import MockEvaluator
from auto_researcher.graph.nodes.search_router import search_router
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic.task import SyntheticTask
from auto_researcher.tasks.synthetic.verification import SyntheticVerificationPolicy
from auto_researcher.verification.verifier import DeterministicVerifier


def _experiment(configuration: dict) -> ExperimentSpec:
    metadata = SyntheticTask().experiment_metadata(TaskRuntimeContext())
    return ExperimentSpec(
        experiment_id="exp-1",
        hypothesis_id="hyp-1",
        search_request_id="request-1",
        configuration=configuration,
        evaluator_id=metadata.evaluator_id,
        code_version=metadata.code_version,
        dataset_version=metadata.dataset_version,
        provenance=metadata.provenance,
    )


def _request(search_type: SearchType) -> SearchRequest:
    return SearchRequest(
        request_id="request-1",
        hypothesis_id="hyp-1",
        search_type=search_type,
        target="score",
        search_space={"model_depth": [3]},
        experiment_budget=1,
        rationale="test",
    )


def test_mock_agents_are_deterministic(contract_factory):
    contract = contract_factory()
    hypothesis_agent = MockHypothesisAgent()
    first = hypothesis_agent.generate(contract, cycle=1)
    second = hypothesis_agent.generate(contract, cycle=1)
    assert first == second
    planner = MockPlannerAgent()
    assert planner.plan(contract, first, cycle=1) == planner.plan(contract, first, cycle=1)


def test_search_router_accepts_direct(contract_factory):
    result = search_router(
        {"contract": contract_factory(), "search_request": _request(SearchType.DIRECT)}
    )
    assert result["search_backend_result"].available is True
    assert result["search_backend_result"].requested_type == SearchType.DIRECT


def test_search_router_rejects_unavailable_optuna(contract_factory):
    contract = contract_factory(allowed=frozenset({SearchType.OPTUNA}))
    result = search_router({"contract": contract, "search_request": _request(SearchType.OPTUNA)})
    assert result["search_backend_result"].available is False
    assert result["search_backend_result"].code == "BACKEND_UNAVAILABLE"


def test_search_router_rejects_unavailable_openevolve(contract_factory):
    contract = contract_factory(allowed=frozenset({SearchType.OPENEVOLVE}))
    result = search_router(
        {"contract": contract, "search_request": _request(SearchType.OPENEVOLVE)}
    )
    assert result["search_backend_result"].available is False
    assert result["search_backend_result"].code == "BACKEND_UNAVAILABLE"


def test_search_router_rejects_type_not_allowed(contract_factory):
    result = search_router(
        {
            "contract": contract_factory(),
            "search_request": _request(SearchType.OPTUNA),
        }
    )
    assert result["search_backend_result"].code == "SEARCH_TYPE_NOT_ALLOWED"


def test_evaluator_stamps_mock_and_has_known_good_configuration(contract_factory):
    result = MockEvaluator().evaluate(
        _experiment(
            {"model_family": "tree", "complexity": 4, "learning_rate": 0.05}
        ),
        contract_factory(),
    )
    assert result.provenance == ProvenanceKind.SIMULATED
    assert result.primary_score == 0.84
    assert all(result.constraint_results.values())


def test_evaluator_has_constraint_violating_configuration(contract_factory):
    result = MockEvaluator().evaluate(
        _experiment(
            {"model_family": "tree", "complexity": 9, "learning_rate": 0.05}
        ),
        contract_factory(),
    )
    assert result.constraint_results["complexity_within_task_limit"] is False


def test_verifier_blocks_supported_from_mock(contract_factory):
    contract = contract_factory()
    experiment = _experiment(
        {"model_family": "tree", "complexity": 4, "learning_rate": 0.05}
    )
    evaluation = MockEvaluator().evaluate(experiment, contract)
    result = DeterministicVerifier(SyntheticVerificationPolicy()).verify(
        experiment, evaluation, contract
    )
    assert result.verified is True
    assert result.evidence_status == EvidenceStatus.INCONCLUSIVE
    assert "synthetic_evidence_cannot_support" in result.reasons


def test_verifier_catches_score_mismatch(contract_factory):
    contract = contract_factory()
    experiment = _experiment(
        {"model_family": "tree", "complexity": 4, "learning_rate": 0.05}
    )
    evaluation = MockEvaluator().evaluate(experiment, contract)
    result = DeterministicVerifier(SyntheticVerificationPolicy()).verify(
        experiment,
        evaluation,
        contract,
        claimed_score=0.12,
    )
    assert result.verified is False
    assert "score_mismatch" in result.reasons
