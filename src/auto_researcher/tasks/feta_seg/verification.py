"""Scientific-integrity verification for FeTA development-only evidence."""

import math

from auto_researcher.contracts.enums import EvidenceStatus
from auto_researcher.contracts.models import EvaluationResult, ResearchContract
from auto_researcher.tasks.feta_seg.evaluator import EVALUATOR_VERSION
from auto_researcher.tasks.feta_seg.splits import FOLD_ID, SPLIT_ID
from auto_researcher.tasks.models import PolicyDecision


class FeTASegVerificationPolicy:
    policy_id = "feta-seg-evidence-policy-v1"
    required_metrics = frozenset(
        {
            "mean_subject_macro_dice",
            "per_tissue_dice",
            "reconstruction_macro_dice",
            "reconstruction_gap",
            "split_identity",
            "fold_identity",
            "evaluator_version",
            "folds_completed",
            "oof_subject_count",
            "holdout_subjects_evaluated",
            "failed_training_folds",
            "valid_prediction_labels",
            "scientific_baseline",
        }
    )

    def evaluate_constraints(
        self, evaluation: EvaluationResult, contract: ResearchContract
    ) -> PolicyDecision:
        reasons: list[str] = []
        score = evaluation.primary_score
        if not evaluation.success or evaluation.error:
            reasons.append("feta_evaluator_unsuccessful")
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            reasons.append("feta_primary_score_invalid")
        if not self.required_metrics.issubset(evaluation.metrics):
            reasons.append("feta_required_metrics_missing")
        if (
            evaluation.metrics.get("split_identity") != SPLIT_ID
            or evaluation.metrics.get("fold_identity") != FOLD_ID
        ):
            reasons.append("feta_partition_identity_mismatch")
        if evaluation.metrics.get("evaluator_version") != EVALUATOR_VERSION:
            reasons.append("feta_evaluator_identity_mismatch")
        if evaluation.metrics.get("holdout_subjects_evaluated") != 0:
            reasons.append("feta_holdout_accessed")
        if evaluation.metrics.get("failed_training_folds") != 0:
            reasons.append("feta_training_fold_failed")
        mode = evaluation.metrics.get("mode")
        expected_folds, expected_subjects = (1, 4) if mode == "smoke" else (5, 68)
        if (
            evaluation.metrics.get("folds_completed") != expected_folds
            or evaluation.metrics.get("oof_subject_count") != expected_subjects
        ):
            reasons.append("feta_fold_evaluation_incomplete")
        if (
            mode == "smoke"
            and evaluation.metrics.get("scientific_baseline") is not False
        ):
            reasons.append("feta_smoke_identity_invalid")
        if not evaluation.constraint_results or not all(
            evaluation.constraint_results.values()
        ):
            reasons.append("feta_evaluator_constraint_failure")
        return PolicyDecision(
            constraint_compliant=not reasons,
            evidence_status=EvidenceStatus.SUPPORTED
            if not reasons
            else EvidenceStatus.REFUTED,
            reasons=("feta_evidence_integrity_verified",)
            if not reasons
            else tuple(reasons),
        )
