"""iCCA eligibility policy subordinate to structural verification."""

from auto_researcher.contracts.enums import EvidenceStatus
from auto_researcher.contracts.models import EvaluationResult, ResearchContract
from auto_researcher.tasks.models import PolicyDecision


class ICCANBSVerificationPolicy:
    policy_id = "icca-nbs-policy-v1"
    required_metrics = frozenset(
        {"primary_score", "stability", "scientific", "eligibility"}
    )

    def evaluate_constraints(
        self,
        evaluation: EvaluationResult,
        contract: ResearchContract,
    ) -> PolicyDecision:
        compliant = bool(evaluation.constraint_results) and all(
            evaluation.constraint_results.values()
        )
        return PolicyDecision(
            constraint_compliant=compliant,
            evidence_status=(
                EvidenceStatus.SUPPORTED if compliant else EvidenceStatus.REFUTED
            ),
            reasons=(
                ("icca_eligibility_gates_passed",)
                if compliant
                else ("icca_eligibility_gate_failed",)
            ),
        )
