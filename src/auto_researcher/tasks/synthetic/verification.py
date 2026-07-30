"""Synthetic task metric and constraint policy."""

from auto_researcher.contracts.enums import EvidenceStatus
from auto_researcher.contracts.models import EvaluationResult, ResearchContract
from auto_researcher.tasks.models import PolicyDecision


class SyntheticVerificationPolicy:
    policy_id = "synthetic-policy-v1"
    required_metrics = frozenset({"objective_score", "stability", "runtime"})

    def evaluate_constraints(
        self,
        evaluation: EvaluationResult,
        contract: ResearchContract,
    ) -> PolicyDecision:
        compliant = bool(evaluation.constraint_results) and all(
            evaluation.constraint_results.values()
        )
        score = float(evaluation.primary_score or 0.0)
        support_threshold = float(contract.constraints.get("support_threshold", 0.75))
        refute_threshold = float(contract.constraints.get("refute_threshold", 0.4))
        if not compliant:
            return PolicyDecision(
                constraint_compliant=False,
                evidence_status=EvidenceStatus.REFUTED,
                reasons=("synthetic_constraint_violation",),
            )
        if score >= support_threshold:
            return PolicyDecision(
                constraint_compliant=True,
                evidence_status=EvidenceStatus.SUPPORTED,
                reasons=("synthetic_support_threshold_met",),
            )
        if score < refute_threshold:
            return PolicyDecision(
                constraint_compliant=True,
                evidence_status=EvidenceStatus.REFUTED,
                reasons=("synthetic_refute_threshold_met",),
            )
        return PolicyDecision(
            constraint_compliant=True,
            evidence_status=EvidenceStatus.INCONCLUSIVE,
            reasons=("synthetic_thresholds_inconclusive",),
        )
