"""Trusted evaluator for host-interpreted FeTA TrainingPolicy candidates."""

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
from auto_researcher.tasks.feta_seg_evolve.configuration import (
    EVOLVE_CONFIGURATION_VERSION,
    FeTASegEvolveConfiguration,
)
from auto_researcher.tasks.feta_seg_evolve.runner import (
    EVOLVE_DATA_LOADER_VERSION,
    EVOLVE_LOSS_VERSION,
    EVOLVE_OPTIMISER_VERSION,
    EVOLVE_RUNNER_VERSION,
    run_evolve_candidate,
)
from auto_researcher.tasks.feta_seg_evolve.training_policy import (
    AUGMENTATION_RECIPE_VERSION,
    TRAINING_POLICY_VERSION,
)
from auto_researcher.tasks.feta_seg_evolve.transforms import (
    EVOLVE_AUGMENTATION_VERSION,
    PREPROCESSING_VERSION,
)
from auto_researcher.tasks.feta_seg_search.gpu_scheduler import (
    GPU_SCHEDULER_VERSION,
    gpu_scheduler_policy,
    scheduler_telemetry_is_valid,
)
from auto_researcher.tasks.feta_seg_search.metric_tiers import (
    FULL_PANEL_METRIC_NAMES,
    METRIC_TIER_POLICY_VERSION,
    SCREEN_METRIC_VERSION,
    metric_tier_for_fidelity,
)
from auto_researcher.tasks.feta_seg_search.trainer import (
    ARCHITECTURE_VERSION,
    INFERENCE_VERSION,
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

EVALUATOR_ID = "feta-segresnet-evolve-evaluator"
EVALUATOR_VERSION = "feta-segresnet-evolve-evaluator-v1"
POLICY = ScientificJsonPolicy(permitted_nan_paths=frozenset())

EvolveRunner = Callable[
    [TaskRuntimeContext, FeTASegEvolveConfiguration, str], dict[str, Any]
]


def evaluator_code_version(dataset_version: str) -> str:
    return "+".join(
        (
            "feta-seg-evolve-task-1.0",
            dataset_version,
            SPLIT_ID,
            EXPECTED_SPLIT_HASH,
            FOLD_ID,
            EXPECTED_FOLD_HASH,
            EVOLVE_CONFIGURATION_VERSION,
            TRAINING_POLICY_VERSION,
            PREPROCESSING_VERSION,
            AUGMENTATION_RECIPE_VERSION,
            EVOLVE_AUGMENTATION_VERSION,
            ARCHITECTURE_VERSION,
            EVOLVE_LOSS_VERSION,
            EVOLVE_OPTIMISER_VERSION,
            INFERENCE_VERSION,
            METRIC_PANEL_VERSION,
            SCREEN_METRIC_VERSION,
            HD95_VERSION,
            EMPTY_PREDICTION_VERSION,
            TOPOLOGY_VERSION,
            METRIC_TIER_POLICY_VERSION,
            GPU_SCHEDULER_VERSION,
            EVOLVE_RUNNER_VERSION,
            EVOLVE_DATA_LOADER_VERSION,
            SCIENTIFIC_JSON_ENCODING_VERSION,
            ARTEFACT_BUNDLE_SCHEMA_VERSION,
        )
    )


class FeTASegEvolveEvaluator:
    evaluator_id = EVALUATOR_ID
    version = EVALUATOR_VERSION
    cost_per_experiment = 0.0

    def __init__(
        self,
        context: TaskRuntimeContext,
        metadata: ExperimentMetadata,
        manifest: DatasetManifest,
        *,
        runner: EvolveRunner = run_evolve_candidate,
    ) -> None:
        self.context = context
        self.metadata = metadata
        self.manifest = manifest
        self.runner = runner

    def _manifest(self) -> dict[str, Any]:
        return {
            "task_id": "feta_seg_evolve",
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
            "training_policy_version": TRAINING_POLICY_VERSION,
            "gpu_scheduler_version": GPU_SCHEDULER_VERSION,
            "holdout_evaluator_calls": 0,
        }

    def _persist(
        self, experiment: ExperimentSpec, result: EvaluationResult
    ) -> EvaluationResult:
        try:
            write_artefact_bundle(
                self.context, experiment, result, self.manifest, self._manifest()
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
            return self._failure(experiment, "feta_evolve_experiment_metadata_mismatch")
        try:
            configuration = FeTASegEvolveConfiguration.model_validate(
                experiment.configuration
            )
            scheduler = gpu_scheduler_policy(self.context)
        except Exception:
            return self._failure(experiment, "feta_evolve_configuration_invalid")
        expected_tier = metric_tier_for_fidelity(configuration.maximum_epochs)
        try:
            metrics = self.runner(self.context, configuration, experiment.experiment_id)
        except Exception as exc:
            safe = str(exc)
            known = {
                "feta_evolve_runner_paths_missing",
                "feta_evolve_dataset_identity_mismatch",
                "feta_evolve_training_loss_non_finite",
                "feta_evolve_best_checkpoint_missing",
                "feta_search_cuda_unavailable",
                "feta_ml_dependencies_unavailable",
                "feta_search_gpu_binding_mismatch",
                "feta_search_gpu_probe_failed",
                "feta_search_gpu_probe_parse_failed",
                "feta_search_gpu_fidelity_disallowed",
                "feta_search_shared_cache_population_partial",
                "feta_search_shared_cache_lock_failed",
            }
            return self._failure(
                experiment,
                safe
                if safe in known
                else f"feta_evolve_evaluation_failed:{type(exc).__name__}",
            )
        metrics.update(
            {
                "configuration": configuration.model_dump(mode="json"),
                "dataset_version": self.metadata.dataset_version,
                "evaluator_version": self.version,
                "evaluator_code_version": self.metadata.code_version,
                "training_policy_version": TRAINING_POLICY_VERSION,
                "preprocessing_version": PREPROCESSING_VERSION,
                "architecture_version": ARCHITECTURE_VERSION,
                "metric_version": METRIC_PANEL_VERSION
                if expected_tier == "full"
                else SCREEN_METRIC_VERSION,
                "search_scope": "development-fold-0-only",
            }
        )
        try:
            metrics = require_valid_scientific_json(
                normalise_scientific_json(metrics, policy=POLICY),
                reason_code="feta_evolve_scientific_json_invalid",
            )
            score = float(metrics["mean_subject_macro_dice"])
        except (ScientificJsonNormalisationError, KeyError, TypeError, ValueError):
            return self._failure(experiment, "feta_evolve_scientific_json_invalid")
        required = {
            "mean_subject_macro_dice",
            "per_tissue_dice",
            "subject_metrics",
            "reconstruction_macro_dice",
            "reconstruction_gap",
            "empty_prediction_count",
            "best_epoch",
            "validation_score",
            "checkpoint_reference",
            "configuration_identity",
            "trajectory_identity",
            "base_configuration_identity",
            "training_policy_identity",
            "policy_trace",
            "candidate_provenance",
            "seeding_mode",
            "metric_tier",
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
        }
        if expected_tier == "full":
            required.update(FULL_PANEL_METRIC_NAMES)
        if scheduler.mode != "disabled":
            required.update(
                {
                    "gpu_scheduler_version",
                    "gpu_scheduler_mode",
                    "physical_gpu_index",
                    "gpu_admission_wait_seconds",
                    "gpu_admission_poll_count",
                    "gpu_admission_free_memory_mib",
                    "gpu_admission_utilization_percent",
                    "gpu_admission_foreign_process_count",
                    "gpu_admission_stable_idle_seconds",
                }
            )
        try:
            empty_count = int(metrics.get("empty_prediction_count", -1))
        except (TypeError, ValueError):
            empty_count = -1
        constraints = {
            "score_finite_and_bounded": math.isfinite(score) and 0 <= score <= 1,
            "required_metrics_complete": required.issubset(metrics),
            "dataset_identity_exact": metrics.get("dataset_manifest_hash")
            == EXPECTED_MANIFEST_HASH
            and self.manifest.metadata.get("manifest_hash") == EXPECTED_MANIFEST_HASH,
            "split_identity_exact": metrics.get("split_identity") == SPLIT_ID
            and metrics.get("split_hash") == EXPECTED_SPLIT_HASH,
            "fold_identity_exact": metrics.get("fold_identity") == FOLD_ID
            and metrics.get("fold_hash") == EXPECTED_FOLD_HASH
            and metrics.get("fold") == 0,
            "fold_membership_exact": metrics.get("training_subject_count") == 54
            and metrics.get("validation_subject_count") == 14,
            "holdout_sealed": metrics.get("holdout_subjects_evaluated") == 0,
            "valid_prediction_labels": metrics.get("valid_prediction_labels")
            == list(range(8)),
            "empty_prediction_count_valid": 0 <= empty_count <= 14 * len(LABELS),
            "metric_tier_exact": metrics.get("metric_tier") == expected_tier,
            "configuration_identity_exact": metrics.get("configuration_identity")
            == payload_hash(configuration),
            "base_identity_exact": metrics.get("base_configuration_identity")
            == configuration.base_configuration_identity,
            "policy_identity_exact": metrics.get("training_policy_identity")
            == configuration.training_policy_identity,
            "seeding_mode_exact": metrics.get("seeding_mode")
            == configuration.seeding_mode,
            "gpu_scheduler_telemetry_valid": scheduler_telemetry_is_valid(
                metrics, scheduler
            ),
        }
        success = all(constraints.values())
        return self._persist(
            experiment,
            EvaluationResult(
                experiment_id=experiment.experiment_id,
                success=success,
                primary_score=score if success else None,
                metrics=metrics,
                constraint_results=constraints,
                artefact_references=artefact_references(
                    self.context, experiment.experiment_id
                ),
                evaluator_version=self.version,
                provenance=ProvenanceKind.REAL,
                error=None if success else "feta_evolve_scientific_constraints_failed",
            ),
        )
