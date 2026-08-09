"""Trusted evaluator boundary for FeTA fold-0 search candidates."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from auto_researcher.contracts.enums import ProvenanceKind
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
)
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.artifacts import (
    ARTEFACT_BUNDLE_SCHEMA_VERSION,
    artefact_references,
    write_artefact_bundle,
)
from auto_researcher.tasks.feta_seg.manifests import EXPECTED_MANIFEST_HASH
from auto_researcher.tasks.feta_seg.metrics import (
    EMPTY_PREDICTION_VERSION,
    HD95_VERSION,
    LABELS,
    METRIC_PANEL_VERSION,
    TOPOLOGY_VERSION,
)
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    FOLD_ID,
    SPLIT_ID,
)
from auto_researcher.tasks.feta_seg_search.configuration import (
    CONFIGURATION_SCHEMA_VERSION,
    FeTASegSearchConfiguration,
)
from auto_researcher.tasks.feta_seg_search.cache import CACHE_IDENTITY_VERSION
from auto_researcher.tasks.feta_seg_search.continuation import CONTINUATION_VERSION
from auto_researcher.tasks.feta_seg_search.metric_tiers import (
    FULL_PANEL_METRIC_NAMES,
    METRIC_TIER_POLICY_VERSION,
    SCREEN_METRIC_VERSION,
    metric_tier_for_fidelity,
)
from auto_researcher.tasks.feta_seg_search.runner import (
    RUNNER_VERSION,
    run_search_candidate,
)
from auto_researcher.tasks.feta_seg_search.trainer import (
    ARCHITECTURE_VERSION,
    INFERENCE_VERSION,
    LOSS_VERSION,
    OPTIMISER_VERSION,
)
from auto_researcher.tasks.feta_seg_search.transforms import (
    AUGMENTATION_POLICY_VERSION,
    PREPROCESSING_VERSION,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)
from auto_researcher.tasks.scientific_json import (
    SCIENTIFIC_JSON_ENCODING_VERSION,
    ScientificJsonNormalisationError,
    ScientificJsonPolicy,
    normalise_scientific_json,
    require_valid_scientific_json,
)

EVALUATOR_ID = "feta-segresnet-search-evaluator"
EVALUATOR_VERSION = "feta-segresnet-search-evaluator-v2"
FETA_SEARCH_SCIENTIFIC_JSON_POLICY = ScientificJsonPolicy(
    permitted_nan_paths=frozenset()
)

SearchRunner = Callable[
    [TaskRuntimeContext, FeTASegSearchConfiguration, str], dict[str, Any]
]


def evaluator_code_version(dataset_version: str) -> str:
    return "+".join(
        (
            "feta-seg-search-task-1.0",
            dataset_version,
            SPLIT_ID,
            EXPECTED_SPLIT_HASH,
            FOLD_ID,
            EXPECTED_FOLD_HASH,
            CONFIGURATION_SCHEMA_VERSION,
            PREPROCESSING_VERSION,
            AUGMENTATION_POLICY_VERSION,
            ARCHITECTURE_VERSION,
            LOSS_VERSION,
            OPTIMISER_VERSION,
            INFERENCE_VERSION,
            METRIC_PANEL_VERSION,
            SCREEN_METRIC_VERSION,
            HD95_VERSION,
            EMPTY_PREDICTION_VERSION,
            TOPOLOGY_VERSION,
            METRIC_TIER_POLICY_VERSION,
            CACHE_IDENTITY_VERSION,
            CONTINUATION_VERSION,
            RUNNER_VERSION,
            SCIENTIFIC_JSON_ENCODING_VERSION,
            ARTEFACT_BUNDLE_SCHEMA_VERSION,
        )
    )


class FeTASegSearchEvaluator:
    evaluator_id = EVALUATOR_ID
    version = EVALUATOR_VERSION
    cost_per_experiment = 0.0

    def __init__(
        self,
        context: TaskRuntimeContext,
        metadata: ExperimentMetadata,
        manifest: DatasetManifest,
        *,
        search_runner: SearchRunner = run_search_candidate,
    ) -> None:
        self.context = context
        self.metadata = metadata
        self.manifest = manifest
        self.search_runner = search_runner

    def _evaluator_manifest(self) -> dict[str, Any]:
        return {
            "task_id": "feta_seg_search",
            "task_version": "1.0",
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.version,
            "code_version": self.metadata.code_version,
            "dataset_version": self.metadata.dataset_version,
            "dataset_manifest_hash": self.manifest.metadata.get("manifest_hash"),
            "split_identity": SPLIT_ID,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_identity": FOLD_ID,
            "fold_hash": EXPECTED_FOLD_HASH,
            "search_scope": "development-fold-0-only",
            "metric_panel_version": METRIC_PANEL_VERSION,
            "metric_tier_policy_version": METRIC_TIER_POLICY_VERSION,
            "cache_identity_version": CACHE_IDENTITY_VERSION,
            "continuation_version": CONTINUATION_VERSION,
            "result_encoding_version": SCIENTIFIC_JSON_ENCODING_VERSION,
            "artefact_bundle_schema_version": ARTEFACT_BUNDLE_SCHEMA_VERSION,
            "holdout_evaluator_calls": 0,
        }

    def _persist(
        self, experiment: ExperimentSpec, result: EvaluationResult
    ) -> EvaluationResult:
        try:
            write_artefact_bundle(
                self.context,
                experiment,
                result,
                self.manifest,
                self._evaluator_manifest(),
            )
            return result
        except Exception as exc:
            return EvaluationResult(
                experiment_id=experiment.experiment_id,
                success=False,
                primary_score=None,
                metrics={
                    "failure_diagnostics": {
                        "failure_stage": "ARTEFACT_WRITING",
                        "safe_exception_class": type(exc).__name__,
                    }
                },
                constraint_results={},
                artefact_references=(),
                evaluator_version=self.version,
                provenance=ProvenanceKind.REAL,
                error=f"artefact_bundle_publication_failed:{type(exc).__name__}",
            )

    def _failure(self, experiment: ExperimentSpec, code: str) -> EvaluationResult:
        return self._persist(
            experiment,
            EvaluationResult(
                experiment_id=experiment.experiment_id,
                success=False,
                primary_score=None,
                metrics={},
                constraint_results={},
                artefact_references=artefact_references(
                    self.context, experiment.experiment_id
                ),
                evaluator_version=self.version,
                provenance=ProvenanceKind.REAL,
                error=code,
            ),
        )

    def evaluate(
        self, experiment: ExperimentSpec, contract: ResearchContract
    ) -> EvaluationResult:
        if (
            experiment.evaluator_id != self.metadata.evaluator_id
            or experiment.code_version != self.metadata.code_version
            or experiment.dataset_version != self.metadata.dataset_version
            or experiment.provenance != self.metadata.provenance
            or contract.primary_metric != "mean_subject_macro_dice"
        ):
            return self._failure(
                experiment, "feta_search_experiment_metadata_mismatch"
            )
        try:
            configuration = FeTASegSearchConfiguration.model_validate(
                experiment.configuration
            )
        except Exception:
            return self._failure(experiment, "feta_search_configuration_invalid")
        expected_metric_tier = metric_tier_for_fidelity(
            configuration.maximum_epochs
        )

        try:
            metrics = self.search_runner(
                self.context, configuration, experiment.experiment_id
            )
        except Exception as exc:
            safe_reason = str(exc)
            safe_codes = {
                "feta_search_cuda_unavailable",
                "feta_ml_dependencies_unavailable",
                "feta_search_runner_paths_missing",
                "feta_search_dataset_identity_mismatch",
                "feta_search_split_subject_identity_mismatch",
                "feta_search_holdout_accessed",
                "feta_search_fold_zero_size_mismatch",
                "feta_search_fold_zero_membership_invalid",
                "feta_search_training_loss_non_finite",
                "feta_search_best_checkpoint_missing",
                "feta_search_best_prediction_identity_mismatch",
                "feta_search_validation_prediction_count_mismatch",
                "feta_search_validation_preparation_incomplete",
                "feta_search_endpoint_prediction_count_mismatch",
                "feta_search_shared_cache_manifest_invalid",
                "feta_search_shared_cache_identity_mismatch",
                "feta_search_shared_cache_manifest_missing",
                "feta_search_shared_cache_completion_invalid",
                "feta_search_shared_cache_completion_mismatch",
                "feta_search_shared_cache_population_mismatch",
                "feta_search_shared_cache_population_incomplete",
                "feta_search_shared_cache_record_invalid",
                "feta_search_resume_candidate_root_invalid",
                "feta_search_resume_checkpoint_missing",
                "feta_search_resume_best_checkpoint_missing",
                "feta_search_resume_checkpoint_unreadable",
                "feta_search_resume_checkpoint_invalid",
                "feta_search_resume_checkpoint_identity_mismatch",
                "feta_search_resume_continuation_identity_mismatch",
                "feta_search_resume_runtime_identity_mismatch",
                "feta_search_resume_fold_mismatch",
                "feta_search_resume_seed_mismatch",
                "feta_search_resume_configuration_invalid",
                "feta_search_resume_source_fidelity_invalid",
                "feta_search_resume_source_fidelity_mismatch",
                "feta_search_resume_configuration_identity_mismatch",
                "feta_search_resume_trajectory_mismatch",
                "feta_search_resume_fidelity_not_higher",
                "feta_search_resume_best_checkpoint_invalid",
                "feta_search_resume_best_checkpoint_identity_mismatch",
                "feta_search_resume_best_checkpoint_unreadable",
                "feta_search_resume_rng_state_invalid",
                "feta_search_resume_best_score_mismatch",
                "feta_search_resume_best_prediction_identity_mismatch",
                "feta_search_resume_best_model_state_invalid",
                "feta_search_resume_optimisation_state_invalid",
            }
            return self._failure(
                experiment,
                safe_reason
                if safe_reason in safe_codes
                else f"feta_search_evaluation_failed:{type(exc).__name__}",
            )

        metrics.update(
            {
                "configuration": configuration.scientific_configuration(),
                "dataset_version": self.metadata.dataset_version,
                "evaluator_version": self.version,
                "evaluator_code_version": self.metadata.code_version,
                "preprocessing_version": PREPROCESSING_VERSION,
                "augmentation_policy_version": AUGMENTATION_POLICY_VERSION,
                "architecture_version": ARCHITECTURE_VERSION,
                "loss_version": LOSS_VERSION,
                "optimiser_version": OPTIMISER_VERSION,
                "inference_version": INFERENCE_VERSION,
                "metric_version": METRIC_PANEL_VERSION
                if expected_metric_tier == "full"
                else SCREEN_METRIC_VERSION,
                "hd95_version": HD95_VERSION,
                "empty_prediction_version": EMPTY_PREDICTION_VERSION,
                "topology_version": TOPOLOGY_VERSION,
                "metric_tier_policy_version": METRIC_TIER_POLICY_VERSION,
                "search_scope": "development-fold-0-only",
            }
        )
        try:
            metrics = require_valid_scientific_json(
                normalise_scientific_json(
                    metrics, policy=FETA_SEARCH_SCIENTIFIC_JSON_POLICY
                ),
                reason_code="feta_search_scientific_json_invalid",
            )
            score = float(metrics["mean_subject_macro_dice"])
        except (ScientificJsonNormalisationError, TypeError, ValueError, KeyError):
            return self._failure(experiment, "feta_search_scientific_json_invalid")
        if not math.isfinite(score):
            return self._failure(experiment, "feta_search_scientific_json_invalid")

        common_required_metric_names = {
            "mean_subject_macro_dice",
            "per_tissue_dice",
            "subject_metrics",
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
            "cache_identity",
            "cache_identity_version",
            "cache_reused",
            "trajectory_identity",
            "resumed",
            "resumed_from_epoch",
            "source_checkpoint_sha256",
            "continuation_version",
            "continuation_semantics",
            "metric_tier",
            "metric_tier_policy_version",
        }
        required_metric_names = set(common_required_metric_names)
        if expected_metric_tier == "full":
            required_metric_names.update(FULL_PANEL_METRIC_NAMES)
        try:
            empty_prediction_count = int(metrics.get("empty_prediction_count", -1))
        except (TypeError, ValueError):
            empty_prediction_count = -1
        constraint_results = {
            "score_finite_and_bounded": math.isfinite(score) and 0 <= score <= 1,
            "dataset_identity_exact": metrics.get("dataset_manifest_hash")
            == EXPECTED_MANIFEST_HASH
            and self.manifest.metadata.get("manifest_hash") == EXPECTED_MANIFEST_HASH,
            "split_identity_exact": metrics.get("split_identity") == SPLIT_ID
            and metrics.get("split_hash") == EXPECTED_SPLIT_HASH,
            "fold_identity_exact": metrics.get("fold_identity") == FOLD_ID
            and metrics.get("fold_hash") == EXPECTED_FOLD_HASH,
            "fold_zero_only": metrics.get("fold") == 0,
            "training_subject_count_exact": metrics.get("training_subject_count")
            == 54,
            "validation_subject_count_exact": metrics.get(
                "validation_subject_count"
            )
            == 14,
            "holdout_sealed": metrics.get("holdout_subjects_evaluated") == 0,
            "valid_prediction_labels": metrics.get("valid_prediction_labels")
            == list(range(8)),
            "required_metrics_complete": required_metric_names.issubset(metrics),
            "metric_tier_exact": metrics.get("metric_tier")
            == expected_metric_tier,
            "metric_tier_policy_exact": metrics.get(
                "metric_tier_policy_version"
            )
            == METRIC_TIER_POLICY_VERSION,
            "screen_metrics_exclude_full_panel": expected_metric_tier == "full"
            or not (FULL_PANEL_METRIC_NAMES & set(metrics)),
            "duplicate_endpoint_inference_avoided": metrics.get(
                "duplicate_endpoint_inference_avoided"
            )
            is True,
            "empty_prediction_count_valid": 0
            <= empty_prediction_count
            <= 14 * len(LABELS),
            "configuration_identity_exact": metrics.get("configuration_identity")
            == payload_hash(configuration),
            "evaluator_identity_exact": metrics.get("evaluator_version")
            == self.version
            and metrics.get("evaluator_code_version") == self.metadata.code_version,
        }
        successful = all(constraint_results.values())
        result = EvaluationResult(
            experiment_id=experiment.experiment_id,
            success=successful,
            primary_score=score if successful else None,
            metrics=metrics,
            constraint_results=constraint_results,
            artefact_references=artefact_references(
                self.context, experiment.experiment_id
            ),
            evaluator_version=self.version,
            provenance=ProvenanceKind.REAL,
            error=None if successful else "feta_search_scientific_constraints_failed",
        )
        return self._persist(experiment, result)
