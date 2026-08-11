"""Evidence verification for FeTA TrainingPolicy evolution."""

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
from auto_researcher.tasks.feta_seg_evolve.feasibility import (
    scientific_feasibility_reasons,
)
from auto_researcher.tasks.models import PolicyDecision


class FeTASegEvolveVerificationPolicy:
    policy_id = "feta-seg-evolve-evidence-policy-v2"
    required_metrics = frozenset(
        {
            "mean_subject_macro_dice",
            "subject_metrics",
            "per_tissue_dice",
            "empty_prediction_count",
            "reconstruction_macro_dice",
            "reconstruction_gap",
            "training_policy_identity",
            "base_configuration_identity",
            "candidate_provenance",
            "policy_trace",
            "dataset_manifest_hash",
            "split_identity",
            "split_hash",
            "fold_identity",
            "fold_hash",
            "fold",
            "training_subject_count",
            "validation_subject_count",
            "holdout_subjects_evaluated",
        }
    )

    def evaluate_constraints(
        self, evaluation: EvaluationResult, contract: ResearchContract
    ) -> PolicyDecision:
        reasons: list[str] = []
        score = evaluation.primary_score
        if not evaluation.success or evaluation.error:
            reasons.append("feta_evolve_evaluator_unsuccessful")
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            reasons.append("feta_evolve_primary_score_invalid")
        if not self.required_metrics.issubset(evaluation.metrics):
            reasons.append("feta_evolve_required_metrics_missing")
        if evaluation.metrics.get("dataset_manifest_hash") != EXPECTED_MANIFEST_HASH:
            reasons.append("feta_evolve_dataset_identity_mismatch")
        if (
            evaluation.metrics.get("split_identity") != SPLIT_ID
            or evaluation.metrics.get("split_hash") != EXPECTED_SPLIT_HASH
            or evaluation.metrics.get("fold_identity") != FOLD_ID
            or evaluation.metrics.get("fold_hash") != EXPECTED_FOLD_HASH
            or evaluation.metrics.get("fold") != 0
        ):
            reasons.append("feta_evolve_split_identity_mismatch")
        if evaluation.metrics.get("holdout_subjects_evaluated") != 0:
            reasons.append("feta_evolve_holdout_accessed")
        if contract.primary_metric != "mean_subject_macro_dice":
            reasons.append("feta_evolve_contract_metric_mismatch")
        if not evaluation.constraint_results or not all(
            evaluation.constraint_results.values()
        ):
            reasons.append("feta_evolve_evaluator_constraint_failure")
        reasons.extend(scientific_feasibility_reasons(evaluation, contract))
        return PolicyDecision(
            constraint_compliant=not reasons,
            evidence_status=EvidenceStatus.SUPPORTED
            if not reasons
            else EvidenceStatus.REFUTED,
            reasons=("feta_evolve_evidence_integrity_verified",)
            if not reasons
            else tuple(reasons),
        )
