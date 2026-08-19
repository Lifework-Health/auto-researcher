"""Evidence integrity policy for the frozen FeTA BasicUNet DIRECT task."""

from __future__ import annotations

import math

from auto_researcher.contracts.enums import EvidenceStatus
from auto_researcher.contracts.models import EvaluationResult, ResearchContract
from auto_researcher.tasks.feta_seg.manifests import EXPECTED_MANIFEST_HASH
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    FOLD_ID,
    SPLIT_ID,
)
from auto_researcher.tasks.feta_unet_direct.evaluator import (
    EVALUATOR_VERSION,
    RESULT_ID,
)
from auto_researcher.tasks.feta_unet_direct.identities import (
    AMP_POLICY_ID,
    DATA_LOADER_ID,
)
from auto_researcher.tasks.feta_unet_direct.model import (
    ARCHITECTURE_ID,
    TRAINABLE_PARAMETER_COUNT,
)
from auto_researcher.tasks.models import PolicyDecision


class FeTAUNetDirectVerificationPolicy:
    policy_id = "feta-basic-unet-direct-evidence-policy-v1"
    architecture_identity = ARCHITECTURE_ID
    architecture_trainable_parameters = TRAINABLE_PARAMETER_COUNT
    data_loader_identity = DATA_LOADER_ID
    required_metrics = frozenset(
        {
            "mean_subject_macro_dice",
            "mean_subject_macro_hd95_mm",
            "mean_subject_macro_volume_similarity",
            "mean_subject_macro_euler_distance",
            "per_class_summary",
            "per_tissue_dice",
            "reconstruction_macro_dice",
            "reconstruction_gap",
            "empty_prediction_count",
            "dataset_manifest_hash",
            "split_identity",
            "split_hash",
            "fold_identity",
            "fold_hash",
            "architecture_identity",
            "architecture_trainable_parameters",
            "evaluator_version",
            "evaluator_code_version",
            "runner_id",
            "data_loader_id",
            "result_identity",
            "folds_completed",
            "oof_subject_count",
            "holdout_subjects_evaluated",
            "failed_training_folds",
            "valid_prediction_labels",
            "scientific_baseline",
            "development_baseline",
            "validation_scope",
            "contains_subject_identifiers",
            "amp_policy_identity",
        }
    )

    def __init__(
        self,
        *,
        evaluator_version: str = EVALUATOR_VERSION,
        result_identity: str = RESULT_ID,
    ) -> None:
        self.evaluator_version = evaluator_version
        self.result_identity = result_identity

    def architecture_is_valid(self, evaluation: EvaluationResult) -> bool:
        return bool(
            evaluation.metrics.get("architecture_identity")
            == self.architecture_identity
            and evaluation.metrics.get("architecture_trainable_parameters")
            == self.architecture_trainable_parameters
        )

    def evaluate_constraints(
        self, evaluation: EvaluationResult, contract: ResearchContract
    ) -> PolicyDecision:
        reasons: list[str] = []
        score = evaluation.primary_score
        if not evaluation.success or evaluation.error:
            reasons.append("feta_unet_evaluator_unsuccessful")
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            reasons.append("feta_unet_primary_score_invalid")
        if not self.required_metrics.issubset(evaluation.metrics):
            reasons.append("feta_unet_required_metrics_missing")
        if evaluation.metrics.get("dataset_manifest_hash") != EXPECTED_MANIFEST_HASH:
            reasons.append("feta_unet_dataset_identity_mismatch")
        if (
            evaluation.metrics.get("split_identity") != SPLIT_ID
            or evaluation.metrics.get("split_hash") != EXPECTED_SPLIT_HASH
        ):
            reasons.append("feta_unet_split_identity_mismatch")
        if (
            evaluation.metrics.get("fold_identity") != FOLD_ID
            or evaluation.metrics.get("fold_hash") != EXPECTED_FOLD_HASH
        ):
            reasons.append("feta_unet_fold_identity_mismatch")
        if not self.architecture_is_valid(evaluation):
            reasons.append("feta_unet_architecture_identity_mismatch")
        if evaluation.metrics.get("evaluator_version") != self.evaluator_version:
            reasons.append("feta_unet_evaluator_identity_mismatch")
        if evaluation.metrics.get("result_identity") != self.result_identity:
            reasons.append("feta_unet_result_identity_mismatch")
        if evaluation.metrics.get("data_loader_id") != self.data_loader_identity:
            reasons.append("feta_unet_data_loader_identity_mismatch")
        if evaluation.metrics.get("amp_policy_identity") != AMP_POLICY_ID:
            reasons.append("feta_unet_amp_policy_identity_mismatch")
        if evaluation.metrics.get("holdout_subjects_evaluated") != 0:
            reasons.append("feta_unet_holdout_accessed")
        if evaluation.metrics.get("failed_training_folds") != 0:
            reasons.append("feta_unet_training_fold_failed")
        profile = evaluation.metrics.get("profile")
        expected = (
            {
                "engineering_smoke": (1, 1),
                "development_baseline": (1, 14),
                "frozen_baseline": (5, 68),
            }.get(profile)
            if isinstance(profile, str)
            else None
        )
        if expected is None:
            reasons.append("feta_unet_profile_invalid")
            expected = (-1, -1)
        if (
            evaluation.metrics.get("folds_completed") != expected[0]
            or evaluation.metrics.get("oof_subject_count") != expected[1]
        ):
            reasons.append("feta_unet_fold_evaluation_incomplete")
        if (
            evaluation.metrics.get("contains_subject_identifiers") is not False
            or "subject_metrics" in evaluation.metrics
        ):
            reasons.append("feta_unet_shareable_identifiers_present")
        if contract.primary_metric != "mean_subject_macro_dice":
            reasons.append("feta_unet_contract_metric_mismatch")
        if not evaluation.constraint_results or not all(
            evaluation.constraint_results.values()
        ):
            reasons.append("feta_unet_evaluator_constraint_failure")
        return PolicyDecision(
            constraint_compliant=not reasons,
            evidence_status=(
                EvidenceStatus.SUPPORTED if not reasons else EvidenceStatus.REFUTED
            ),
            reasons=("feta_unet_evidence_integrity_verified",)
            if not reasons
            else tuple(reasons),
        )
