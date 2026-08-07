"""Evidence-integrity policy for the fixed Iris benchmark."""

from __future__ import annotations

import math

from auto_researcher.contracts.enums import EvidenceStatus
from auto_researcher.contracts.models import EvaluationResult, ResearchContract
from auto_researcher.tasks.iris_knn.configuration import IrisKNNConfiguration
from auto_researcher.tasks.iris_knn.evaluator import EVALUATOR_VERSION
from auto_researcher.tasks.iris_knn.manifests import (
    CLASS_NAMES,
    DATASET_VERSION,
    FOLD_VERSION,
)
from auto_researcher.tasks.models import PolicyDecision


class IrisKNNVerificationPolicy:
    policy_id = "iris-knn-evidence-policy-v1"
    required_metrics = frozenset(
        {
            "mean_balanced_accuracy",
            "per_fold_balanced_accuracy",
            "overall_accuracy",
            "aggregate_confusion_counts",
            "per_species_recall",
            "configuration",
            "dataset_version",
            "fold_version",
            "evaluator_version",
        }
    )

    def evaluate_constraints(
        self, evaluation: EvaluationResult, contract: ResearchContract
    ) -> PolicyDecision:
        reasons: list[str] = []
        score = evaluation.primary_score
        if not evaluation.success or evaluation.error is not None:
            reasons.append("iris_evaluator_unsuccessful")
        if score is None or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            reasons.append("iris_primary_score_invalid")
        if not self.required_metrics.issubset(evaluation.metrics):
            reasons.append("iris_required_metrics_missing")
        if evaluation.metrics.get("dataset_version") != DATASET_VERSION:
            reasons.append("iris_dataset_identity_mismatch")
        if evaluation.metrics.get("fold_version") != FOLD_VERSION:
            reasons.append("iris_fold_identity_mismatch")
        if evaluation.metrics.get("evaluator_version") != EVALUATOR_VERSION:
            reasons.append("iris_evaluator_identity_mismatch")
        folds = evaluation.metrics.get("per_fold_balanced_accuracy")
        if not isinstance(folds, (list, tuple)) or len(folds) != 5:
            reasons.append("iris_fold_results_incomplete")
        recalls = evaluation.metrics.get("per_species_recall")
        if not isinstance(recalls, dict) or set(recalls) != set(CLASS_NAMES):
            reasons.append("iris_class_results_incomplete")
        try:
            IrisKNNConfiguration.model_validate(evaluation.metrics.get("configuration"))
        except Exception:
            reasons.append("iris_configuration_invalid")
        if not evaluation.constraint_results or not all(
            evaluation.constraint_results.values()
        ):
            reasons.append("iris_evaluator_constraint_failure")
        if contract.primary_metric != "mean_balanced_accuracy":
            reasons.append("iris_contract_metric_mismatch")
        if reasons:
            return PolicyDecision(
                constraint_compliant=False,
                evidence_status=EvidenceStatus.REFUTED,
                reasons=tuple(reasons),
            )
        return PolicyDecision(
            constraint_compliant=True,
            evidence_status=EvidenceStatus.SUPPORTED,
            reasons=("iris_evidence_integrity_verified",),
        )
