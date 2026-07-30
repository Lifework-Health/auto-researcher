"""Structural reconciliation between experiment, measurement, and evidence claim."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from auto_researcher.contracts.enums import EvidenceStatus, ProvenanceKind
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
    VerificationResult,
)


@runtime_checkable
class Verifier(Protocol):
    verifier_id: str

    def verify(
        self,
        experiment: ExperimentSpec,
        evaluation: EvaluationResult,
        contract: ResearchContract,
        *,
        claimed_score: float | None = None,
    ) -> VerificationResult: ...


class DeterministicVerifier:
    verifier_id = "deterministic-verifier"
    required_metrics = frozenset({"primary_score", "stability"})

    def __init__(self, score_tolerance: float = 1e-9) -> None:
        self.score_tolerance = score_tolerance

    def verify(
        self,
        experiment: ExperimentSpec,
        evaluation: EvaluationResult,
        contract: ResearchContract,
        *,
        claimed_score: float | None = None,
    ) -> VerificationResult:
        reasons: list[str] = []
        if experiment.experiment_id != evaluation.experiment_id:
            reasons.append("experiment_result_mismatch")
        if experiment.evaluator_id != contract.evaluator_id:
            reasons.append("unregistered_evaluator")
        if contract.verifier_id != self.verifier_id:
            reasons.append("unregistered_verifier")
        if not evaluation.success:
            reasons.append("evaluation_failed")
        missing = sorted(self.required_metrics - evaluation.metrics.keys())
        if missing:
            reasons.append(f"missing_metrics:{','.join(missing)}")
        if not evaluation.constraint_results:
            reasons.append("constraints_not_evaluated")
        measured = evaluation.primary_score
        claim = measured if claimed_score is None else claimed_score
        if (
            claim is not None
            and measured is not None
            and abs(claim - measured) > self.score_tolerance
        ):
            reasons.append("score_mismatch")

        constraint_compliant = bool(evaluation.constraint_results) and all(
            evaluation.constraint_results.values()
        )
        if not constraint_compliant:
            reasons.append("constraint_violation")
        verified = not reasons

        if evaluation.provenance in {ProvenanceKind.MOCK, ProvenanceKind.SIMULATED}:
            evidence_status = EvidenceStatus.INCONCLUSIVE
            reasons.append("synthetic_evidence_cannot_support")
        elif verified and constraint_compliant:
            evidence_status = EvidenceStatus.SUPPORTED
        elif "constraint_violation" in reasons:
            evidence_status = EvidenceStatus.REFUTED
        else:
            evidence_status = EvidenceStatus.INCONCLUSIVE

        return VerificationResult(
            experiment_id=evaluation.experiment_id,
            verified=verified,
            claimed_score=claim,
            measured_score=measured,
            constraint_compliant=constraint_compliant,
            evidence_status=evidence_status,
            reasons=tuple(reasons),
            provenance=evaluation.provenance,
        )
