"""Structural reconciliation between experiment, measurement, and evidence claim."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from auto_researcher.contracts.enums import EvidenceStatus, ProvenanceKind
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
    VerificationResult,
)
from auto_researcher.tasks.protocols import VerificationPolicy


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

    def __init__(
        self,
        policy: VerificationPolicy,
        score_tolerance: float = 1e-9,
    ) -> None:
        self.policy = policy
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
        if experiment.provenance != evaluation.provenance:
            reasons.append("experiment_evaluation_provenance_mismatch")
        if not evaluation.success:
            reasons.append("evaluation_failed")
        missing = sorted(self.policy.required_metrics - evaluation.metrics.keys())
        if missing:
            reasons.append(f"missing_metrics:{','.join(missing)}")
        measured = evaluation.primary_score
        claim = measured if claimed_score is None else claimed_score
        if measured is not None and not math.isfinite(measured):
            reasons.append("non_finite_measured_score")
            measured = None
        if claim is not None and not math.isfinite(claim):
            reasons.append("non_finite_claimed_score")
            claim = None
        if (
            claim is not None
            and measured is not None
            and math.isfinite(claim)
            and math.isfinite(measured)
            and abs(claim - measured) > self.score_tolerance
        ):
            reasons.append("score_mismatch")

        structural_ok = not reasons
        if structural_ok:
            decision = self.policy.evaluate_constraints(evaluation, contract)
            constraint_compliant = decision.constraint_compliant
            reasons.extend(decision.reasons)
            evidence_status = decision.evidence_status
        else:
            constraint_compliant = False
            evidence_status = EvidenceStatus.INCONCLUSIVE

        if (
            evaluation.provenance in {ProvenanceKind.MOCK, ProvenanceKind.SIMULATED}
            and evidence_status == EvidenceStatus.SUPPORTED
        ):
            evidence_status = EvidenceStatus.INCONCLUSIVE
            reasons.append("synthetic_evidence_cannot_support")

        return VerificationResult(
            experiment_id=evaluation.experiment_id,
            verified=structural_ok,
            claimed_score=claim,
            measured_score=measured,
            constraint_compliant=constraint_compliant,
            evidence_status=evidence_status,
            reasons=tuple(reasons),
            provenance=evaluation.provenance,
        )
