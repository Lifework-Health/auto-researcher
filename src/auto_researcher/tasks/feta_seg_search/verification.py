"""Scientific-integrity verification for FeTA fold-0 search evidence."""

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
from auto_researcher.tasks.feta_seg_search.evaluator import EVALUATOR_VERSION
from auto_researcher.tasks.feta_seg_search.metric_tiers import (
    FULL_PANEL_METRIC_NAMES,
)
from auto_researcher.tasks.models import PolicyDecision


class FeTASegSearchVerificationPolicy:
    policy_id = "feta-seg-search-evidence-policy-v2"
    required_metrics = frozenset(
        {
            "mean_subject_macro_dice",
            "subject_metrics",
            "per_tissue_dice",
            "reconstruction_macro_dice",
            "reconstruction_gap",
            "empty_prediction_count",
            "best_epoch",
            "validation_score",
            "cache_prepare_seconds",
            "validation_prepare_seconds",
            "training_seconds",
            "training_duration_seconds",
            "validation_inference_seconds",
            "endpoint_metric_seconds",
            "total_duration_seconds",
            "peak_gpu_memory_bytes",
            "duplicate_endpoint_inference_avoided",
            "validation_epochs",
            "checkpoint_reference",
            "last_checkpoint_reference",
            "environment",
            "environment_identity",
            "configuration_identity",
            "trajectory_identity",
            "cache_identity",
            "cache_identity_version",
            "cache_reused",
            "metric_tier",
            "metric_tier_policy_version",
            "resumed",
            "resumed_from_epoch",
            "source_checkpoint_sha256",
            "continuation_version",
            "continuation_semantics",
            "dataset_manifest_hash",
            "split_identity",
            "split_hash",
            "fold_identity",
            "fold_hash",
            "fold",
            "training_subject_count",
            "validation_subject_count",
            "holdout_subjects_evaluated",
            "valid_prediction_labels",
            "evaluator_version",
            "evaluator_code_version",
            "search_scope",
        }
    )

    def evaluate_constraints(
        self, evaluation: EvaluationResult, contract: ResearchContract
    ) -> PolicyDecision:
        reasons: list[str] = []
        score = evaluation.primary_score
        if not evaluation.success or evaluation.error:
            reasons.append("feta_search_evaluator_unsuccessful")
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            reasons.append("feta_search_primary_score_invalid")
        if not self.required_metrics.issubset(evaluation.metrics):
            reasons.append("feta_search_required_metrics_missing")
        tier = evaluation.metrics.get("metric_tier")
        if tier not in {"screen", "full"}:
            reasons.append("feta_search_metric_tier_invalid")
        elif tier == "full" and not FULL_PANEL_METRIC_NAMES.issubset(
            evaluation.metrics
        ):
            reasons.append("feta_search_full_metrics_missing")
        elif tier == "screen" and FULL_PANEL_METRIC_NAMES & set(
            evaluation.metrics
        ):
            reasons.append("feta_search_screen_metrics_masquerade_as_full")
        if evaluation.metrics.get("dataset_manifest_hash") != EXPECTED_MANIFEST_HASH:
            reasons.append("feta_search_dataset_identity_mismatch")
        if (
            evaluation.metrics.get("split_identity") != SPLIT_ID
            or evaluation.metrics.get("split_hash") != EXPECTED_SPLIT_HASH
        ):
            reasons.append("feta_search_split_identity_mismatch")
        if (
            evaluation.metrics.get("fold_identity") != FOLD_ID
            or evaluation.metrics.get("fold_hash") != EXPECTED_FOLD_HASH
            or evaluation.metrics.get("fold") != 0
        ):
            reasons.append("feta_search_fold_identity_mismatch")
        if (
            evaluation.metrics.get("training_subject_count") != 54
            or evaluation.metrics.get("validation_subject_count") != 14
        ):
            reasons.append("feta_search_fold_membership_incomplete")
        if evaluation.metrics.get("holdout_subjects_evaluated") != 0:
            reasons.append("feta_search_holdout_accessed")
        if evaluation.metrics.get("evaluator_version") != EVALUATOR_VERSION:
            reasons.append("feta_search_evaluator_identity_mismatch")
        if contract.primary_metric != "mean_subject_macro_dice":
            reasons.append("feta_search_contract_metric_mismatch")
        if not evaluation.constraint_results or not all(
            evaluation.constraint_results.values()
        ):
            reasons.append("feta_search_evaluator_constraint_failure")
        return PolicyDecision(
            constraint_compliant=not reasons,
            evidence_status=(
                EvidenceStatus.SUPPORTED if not reasons else EvidenceStatus.REFUTED
            ),
            reasons=("feta_search_evidence_integrity_verified",)
            if not reasons
            else tuple(reasons),
        )
