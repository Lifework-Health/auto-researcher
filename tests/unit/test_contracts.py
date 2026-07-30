from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from auto_researcher.contracts.enums import (
    EvidenceStatus,
    EventType,
    HypothesisStatus,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import (
    ApprovalRequest,
    BudgetState,
    DecisionEvent,
    EvaluationResult,
    ExperimentSpec,
    Hypothesis,
    SearchBackendResult,
    SearchRequest,
    VerificationResult,
)


def test_all_contracts_validate_and_json_serialise(contract_factory):
    contract = contract_factory()
    hypothesis = Hypothesis(
        hypothesis_id="hyp-1",
        statement="statement",
        rationale="rationale",
        predicted_subspace={"depth": [2, 3]},
        expected_observation="score increases",
        falsification_condition="score does not increase",
        prior_weight=0.5,
        status=HypothesisStatus.OPEN,
        provenance=ProvenanceKind.MOCK,
    )
    request = SearchRequest(
        request_id="req-1",
        hypothesis_id="hyp-1",
        search_type=SearchType.DIRECT,
        target="primary_score",
        search_space={"depth": [3]},
        experiment_budget=1,
        rationale="bounded direct check",
        requires_human_approval=False,
    )
    experiment = ExperimentSpec(
        experiment_id="exp-1",
        hypothesis_id="hyp-1",
        search_request_id="req-1",
        configuration={"depth": 3},
        evaluator_id="mock-evaluator",
        code_version="test",
        dataset_version="mock-v1",
        provenance=ProvenanceKind.MOCK,
    )
    evaluation = EvaluationResult(
        experiment_id="exp-1",
        success=True,
        primary_score=0.8,
        metrics={"primary_score": 0.8, "stability": 0.9},
        constraint_results={"valid": True},
        evaluator_version="mock-v1",
        provenance=ProvenanceKind.MOCK,
    )
    verification = VerificationResult(
        experiment_id="exp-1",
        verified=True,
        claimed_score=0.8,
        measured_score=0.8,
        constraint_compliant=True,
        evidence_status=EvidenceStatus.INCONCLUSIVE,
        reasons=("synthetic_evidence_cannot_support",),
        provenance=ProvenanceKind.MOCK,
    )
    event = DecisionEvent(
        event_id="event-1",
        run_id="run-1",
        cycle=1,
        event_type=EventType.EVIDENCE_VERIFIED,
        actor="verifier",
        input_references=("exp-1",),
        rationale="verified",
        timestamp=datetime(2026, 7, 30, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.MOCK,
    )
    budget = BudgetState(maximum_cycles=1, maximum_experiments=1, maximum_cost=1)
    approval = ApprovalRequest(
        request_id="approval-1",
        run_id="run-1",
        cycle=1,
        search_request_id="req-1",
        search_type=SearchType.DIRECT,
        target="primary_score",
        rationale="required by contract",
    )
    backend = SearchBackendResult(
        requested_type=SearchType.DIRECT,
        available=True,
        code="BACKEND_AVAILABLE",
        message="installed",
    )
    for model in (
        contract,
        hypothesis,
        request,
        experiment,
        evaluation,
        verification,
        event,
        budget,
        approval,
        backend,
    ):
        assert json.loads(model.model_dump_json())
        assert type(model).model_validate_json(model.model_dump_json()) == model


def test_research_contract_is_deeply_immutable(contract_factory):
    contract = contract_factory()
    with pytest.raises(ValidationError):
        contract.maximum_cycles = 9
    with pytest.raises(TypeError):
        contract.constraints["new"] = True
    with pytest.raises(TypeError):
        contract.constraints["nested"]["values"].append(3)


def test_synthetic_verification_cannot_claim_supported():
    with pytest.raises(ValidationError, match="cannot be SUPPORTED"):
        VerificationResult(
            experiment_id="exp-1",
            verified=True,
            claimed_score=0.8,
            measured_score=0.8,
            constraint_compliant=True,
            evidence_status=EvidenceStatus.SUPPORTED,
            reasons=(),
            provenance=ProvenanceKind.MOCK,
        )


def test_approval_payload_is_json_serialisable():
    payload = ApprovalRequest(
        request_id="approval-1",
        run_id="run-1",
        cycle=1,
        search_request_id="req-1",
        search_type=SearchType.DIRECT,
        target="score",
        rationale="contract requires it",
    ).model_dump(mode="json")
    assert json.loads(json.dumps(payload))["search_type"] == "DIRECT"
