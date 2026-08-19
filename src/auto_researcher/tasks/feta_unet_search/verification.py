"""Evidence verification for bounded U-Net search candidates."""

from auto_researcher.contracts.enums import EvidenceStatus
from auto_researcher.contracts.models import EvaluationResult, ResearchContract
from auto_researcher.tasks.feta_unet_direct.verification import (
    FeTAUNetDirectVerificationPolicy,
)
from auto_researcher.tasks.feta_unet_search.evaluator import (
    AUGMENTATION_ID,
    EVALUATOR_VERSION,
    LOSS_ID,
    OPTIMISER_ID,
    RESULT_ID,
)
from auto_researcher.tasks.models import PolicyDecision


class FeTAUNetSearchVerificationPolicy(FeTAUNetDirectVerificationPolicy):
    policy_id = "feta-basic-unet-search-evidence-policy-v1"

    def __init__(self) -> None:
        super().__init__(
            evaluator_version=EVALUATOR_VERSION,
            result_identity=RESULT_ID,
        )

    def evaluate_constraints(
        self, evaluation: EvaluationResult, contract: ResearchContract
    ) -> PolicyDecision:
        parent = super().evaluate_constraints(evaluation, contract)
        reasons = list(parent.reasons if not parent.constraint_compliant else ())
        expected_identities = {
            "augmentation_version": AUGMENTATION_ID,
            "loss_identity": LOSS_ID,
            "optimiser_identity": OPTIMISER_ID,
        }
        if any(
            evaluation.metrics.get(name) != expected
            for name, expected in expected_identities.items()
        ):
            reasons.append("feta_unet_search_training_policy_identity_mismatch")
        return PolicyDecision(
            constraint_compliant=not reasons,
            evidence_status=(
                EvidenceStatus.SUPPORTED if not reasons else EvidenceStatus.REFUTED
            ),
            reasons=("feta_unet_search_evidence_integrity_verified",)
            if not reasons
            else tuple(reasons),
        )
