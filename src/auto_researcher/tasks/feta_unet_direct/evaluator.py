"""Trusted evaluator boundary for the frozen FeTA BasicUNet DIRECT task."""

from __future__ import annotations

import math
import re
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
from auto_researcher.tasks.feta_seg.transforms import (
    AUGMENTATION_VERSION,
    PREPROCESSING_VERSION,
)
from auto_researcher.tasks.feta_unet_direct.configuration import (
    FeTAUNetDirectConfiguration,
)
from auto_researcher.tasks.feta_unet_direct.identities import (
    AMP_POLICY_ID,
    BASELINE_RUNNER_ID,
    DATA_LOADER_ID,
    DEVELOPMENT_BASELINE_RUNNER_ID,
    ENGINEERING_SMOKE_RUNNER_ID,
)
from auto_researcher.tasks.feta_unet_direct.model import (
    ARCHITECTURE_ID,
    MEASURED_ALLOCATOR_CEILING_MIB,
    MEASURED_INPUT_SHAPE,
    MEASURED_OUTPUT_SHAPE,
    MEASURED_PEAK_CUDA_ALLOCATED_MIB,
    MEASURED_PEAK_CUDA_RESERVED_MIB,
    TRAINABLE_PARAMETER_COUNT,
)
from auto_researcher.tasks.feta_unet_direct.runner import run_profile
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

EVALUATOR_ID = "feta-basic-unet-direct-evaluator"
EVALUATOR_VERSION = "feta-basic-unet-direct-evaluator-v1"
RESULT_ID = "feta-basic-unet-direct-result-v1"
SCIENTIFIC_ID = "feta-development-oof-subject-macro-dice-v1"
LOSS_ID = "dice-ce-softmax-onehot-no-background-equal-v1"
OPTIMISER_ID = "adamw-lr1e-4-wd1e-5-v1"
INFERENCE_ID = "sliding-window-128-overlap0.5-gaussian-native-restore-v2"
FETA_UNET_JSON_POLICY = ScientificJsonPolicy(permitted_nan_paths=frozenset())

ProfileRunner = Callable[
    [TaskRuntimeContext, FeTAUNetDirectConfiguration, str], dict[str, Any]
]

_SUBJECT_VALUE = re.compile(r"\bsub-\d{3}\b", re.IGNORECASE)
_PROHIBITED_EVIDENCE_KEYS = frozenset(
    {
        "image_path",
        "patient_id",
        "patient_identifier",
        "segmentation_path",
        "subject_id",
        "subject_identifier",
        "subject_metrics",
    }
)


def _contains_prohibited_evidence(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in _PROHIBITED_EVIDENCE_KEYS
            or _contains_prohibited_evidence(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_prohibited_evidence(item) for item in value)
    return isinstance(value, str) and (
        _SUBJECT_VALUE.search(value) is not None
        or value.casefold().endswith((".nii", ".nii.gz"))
    )


def evaluator_code_version(dataset_version: str) -> str:
    return "+".join(
        (
            "feta-unet-direct-task-1.0",
            dataset_version,
            SPLIT_ID,
            EXPECTED_SPLIT_HASH,
            FOLD_ID,
            EXPECTED_FOLD_HASH,
            PREPROCESSING_VERSION,
            AUGMENTATION_VERSION,
            ARCHITECTURE_ID,
            LOSS_ID,
            OPTIMISER_ID,
            INFERENCE_ID,
            METRIC_PANEL_VERSION,
            HD95_VERSION,
            EMPTY_PREDICTION_VERSION,
            TOPOLOGY_VERSION,
            ENGINEERING_SMOKE_RUNNER_ID,
            DEVELOPMENT_BASELINE_RUNNER_ID,
            BASELINE_RUNNER_ID,
            DATA_LOADER_ID,
            AMP_POLICY_ID,
            RESULT_ID,
            SCIENTIFIC_JSON_ENCODING_VERSION,
            ARTEFACT_BUNDLE_SCHEMA_VERSION,
        )
    )


class FeTAUNetDirectEvaluator:
    evaluator_id = EVALUATOR_ID
    version = EVALUATOR_VERSION
    cost_per_experiment = 0.0

    def __init__(
        self,
        context: TaskRuntimeContext,
        metadata: ExperimentMetadata,
        manifest: DatasetManifest,
        *,
        profile_runner: ProfileRunner = run_profile,
        configuration_model: type = FeTAUNetDirectConfiguration,
        task_id: str = "feta_unet_direct",
        scientific_identity: str = SCIENTIFIC_ID,
        result_identity: str = RESULT_ID,
        augmentation_identity: str = AUGMENTATION_VERSION,
        loss_identity: str = LOSS_ID,
        optimiser_identity: str = OPTIMISER_ID,
    ) -> None:
        self.context = context
        self.metadata = metadata
        self.manifest = manifest
        self.profile_runner = profile_runner
        self.configuration_model = configuration_model
        self.task_id = task_id
        self.scientific_identity = scientific_identity
        self.result_identity = result_identity
        self.augmentation_identity = augmentation_identity
        self.loss_identity = loss_identity
        self.optimiser_identity = optimiser_identity

    def _evaluator_manifest(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_version": "1.0",
            "scientific_identity": self.scientific_identity,
            "architecture_identity": ARCHITECTURE_ID,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.version,
            "result_identity": self.result_identity,
            "code_version": self.metadata.code_version,
            "dataset_version": self.metadata.dataset_version,
            "dataset_manifest_hash": self.manifest.metadata.get("manifest_hash"),
            "split_identity": SPLIT_ID,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_identity": FOLD_ID,
            "fold_hash": EXPECTED_FOLD_HASH,
            "augmentation_identity": self.augmentation_identity,
            "loss_identity": self.loss_identity,
            "optimiser_identity": self.optimiser_identity,
            "inference_identity": INFERENCE_ID,
            "metric_panel_version": METRIC_PANEL_VERSION,
            "result_encoding_version": SCIENTIFIC_JSON_ENCODING_VERSION,
            "artefact_bundle_schema_version": ARTEFACT_BUNDLE_SCHEMA_VERSION,
            "amp_policy_identity": AMP_POLICY_ID,
            "holdout_evaluator_calls": 0,
            "contains_subject_identifiers": False,
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
            return self._failure(experiment, "feta_unet_experiment_metadata_mismatch")
        try:
            configuration = self.configuration_model.model_validate(
                experiment.configuration
            )
        except Exception:
            return self._failure(experiment, "feta_unet_configuration_invalid")
        try:
            metrics = self.profile_runner(
                self.context, configuration, experiment.experiment_id
            )
        except Exception as exc:
            safe_reason = str(exc)
            safe_codes = {
                "feta_cuda_unavailable_for_full_baseline",
                "feta_ml_dependencies_unavailable",
                "feta_unet_runtime_paths_missing",
                "feta_unet_dataset_identity_mismatch",
                "feta_split_identity_mismatch",
                "feta_fold_identity_mismatch",
                "feta_unet_holdout_accessed",
                "feta_unet_oof_membership_invalid",
                "feta_unet_oof_coverage_invalid",
                "feta_unet_training_loss_non_finite",
                "feta_unet_training_gradient_non_finite",
                "feta_unet_repeated_amp_overflow",
                "feta_unet_prediction_non_finite",
                "feta_unet_validation_metric_non_finite",
                "feta_unet_checkpoint_identity_mismatch",
                "feta_unet_fold_restart_checkpoint_identity_mismatch",
            }
            return self._failure(
                experiment,
                safe_reason
                if safe_reason in safe_codes
                else f"feta_unet_evaluation_failed:{type(exc).__name__}",
            )

        if metrics.get(
            "contains_subject_identifiers"
        ) is not False or _contains_prohibited_evidence(metrics):
            return self._failure(experiment, "feta_unet_shareable_evidence_invalid")

        scientific_baseline = configuration.profile == "frozen_baseline"
        development_baseline = configuration.profile == "development_baseline"
        validation_scope = {
            "engineering_smoke": "single-subject-engineering-smoke",
            "development_baseline": "fold0-development-oof",
            "frozen_baseline": "five-fold-development-oof",
        }[configuration.profile]
        metrics.update(
            {
                "scientific_baseline": scientific_baseline,
                "development_baseline": development_baseline,
                "validation_scope": validation_scope,
                "profile": configuration.profile,
                "scientific_identity": self.scientific_identity,
                "architecture_identity": ARCHITECTURE_ID,
                "architecture_trainable_parameters": TRAINABLE_PARAMETER_COUNT,
                "architecture_measured_input_shape": list(MEASURED_INPUT_SHAPE),
                "architecture_measured_output_shape": list(MEASURED_OUTPUT_SHAPE),
                "architecture_measured_peak_cuda_allocated_mib": (
                    MEASURED_PEAK_CUDA_ALLOCATED_MIB
                ),
                "architecture_measured_peak_cuda_reserved_mib": (
                    MEASURED_PEAK_CUDA_RESERVED_MIB
                ),
                "architecture_measured_allocator_ceiling_mib": (
                    MEASURED_ALLOCATOR_CEILING_MIB
                ),
                "architecture_measured_full_step_passed": True,
                "configuration": configuration.scientific_configuration(),
                "configuration_identity": payload_hash(configuration),
                "dataset_version": self.metadata.dataset_version,
                "dataset_manifest_hash": self.manifest.metadata.get("manifest_hash"),
                "split_identity": SPLIT_ID,
                "split_hash": EXPECTED_SPLIT_HASH,
                "fold_identity": FOLD_ID,
                "fold_hash": EXPECTED_FOLD_HASH,
                "evaluator_version": self.version,
                "evaluator_code_version": self.metadata.code_version,
                "result_identity": self.result_identity,
                "preprocessing_version": PREPROCESSING_VERSION,
                "augmentation_version": self.augmentation_identity,
                "loss_identity": self.loss_identity,
                "optimiser_identity": self.optimiser_identity,
                "inference_identity": INFERENCE_ID,
                "metric_version": METRIC_PANEL_VERSION,
                "hd95_version": HD95_VERSION,
                "empty_prediction_version": EMPTY_PREDICTION_VERSION,
                "topology_version": TOPOLOGY_VERSION,
                "amp_policy_identity": AMP_POLICY_ID,
                "contains_subject_identifiers": False,
            }
        )
        try:
            metrics = require_valid_scientific_json(
                normalise_scientific_json(metrics, policy=FETA_UNET_JSON_POLICY),
                reason_code="feta_unet_scientific_json_invalid",
            )
        except (ScientificJsonNormalisationError, TypeError):
            return self._failure(experiment, "feta_unet_scientific_json_invalid")

        required_metrics = {
            "mean_subject_macro_dice",
            "mean_subject_macro_hd95_mm",
            "mean_subject_macro_volume_similarity",
            "mean_subject_macro_euler_distance",
            "per_class_summary",
            "per_tissue_dice",
            "reconstruction_macro_dice",
            "reconstruction_gap",
            "empty_prediction_count",
            "runner_id",
            "data_loader_id",
            "folds_completed",
            "oof_subject_count",
            "holdout_subjects_evaluated",
            "failed_training_folds",
            "valid_prediction_labels",
            "development_baseline",
            "validation_scope",
        }
        if not required_metrics.issubset(metrics):
            return self._failure(experiment, "feta_unet_metrics_incomplete")
        try:
            score = float(metrics["mean_subject_macro_dice"])
            empty_prediction_count = int(metrics["empty_prediction_count"])
        except (TypeError, ValueError):
            return self._failure(experiment, "feta_unet_metrics_invalid")
        if scientific_baseline:
            expected_folds, expected_subjects = 5, 68
            expected_runner = BASELINE_RUNNER_ID
        elif development_baseline:
            expected_folds, expected_subjects = 1, 14
            expected_runner = DEVELOPMENT_BASELINE_RUNNER_ID
        else:
            expected_folds, expected_subjects = 1, 1
            expected_runner = ENGINEERING_SMOKE_RUNNER_ID
        constraint_results = {
            "score_finite_and_bounded": math.isfinite(score) and 0 <= score <= 1,
            "profile_fold_count_exact": metrics["folds_completed"] == expected_folds,
            "holdout_sealed": metrics["holdout_subjects_evaluated"] == 0,
            "valid_prediction_labels": metrics["valid_prediction_labels"]
            == list(range(8)),
            "empty_prediction_count_valid": 0
            <= empty_prediction_count
            <= expected_subjects * len(LABELS),
            "required_metrics_complete": required_metrics.issubset(metrics),
            "dataset_identity_exact": metrics["dataset_manifest_hash"]
            == EXPECTED_MANIFEST_HASH,
            "split_identity_exact": metrics["split_identity"] == SPLIT_ID
            and metrics["split_hash"] == EXPECTED_SPLIT_HASH,
            "fold_identity_exact": metrics["fold_identity"] == FOLD_ID
            and metrics["fold_hash"] == EXPECTED_FOLD_HASH,
            "architecture_identity_exact": metrics["architecture_identity"]
            == ARCHITECTURE_ID
            and metrics["architecture_trainable_parameters"]
            == TRAINABLE_PARAMETER_COUNT,
            "evaluator_identity_exact": metrics["evaluator_version"] == self.version
            and metrics["evaluator_code_version"] == self.metadata.code_version,
            "runner_identity_exact": metrics["runner_id"] == expected_runner,
            "data_loader_identity_exact": metrics["data_loader_id"] == DATA_LOADER_ID,
            "oof_subject_count_exact": metrics["oof_subject_count"]
            == expected_subjects,
            "no_subject_identifiers": metrics["contains_subject_identifiers"] is False
            and "subject_metrics" not in metrics,
            "scientific_identity_correct": metrics["scientific_baseline"]
            is scientific_baseline,
            "development_identity_correct": metrics["development_baseline"]
            is development_baseline,
            "validation_scope_exact": metrics["validation_scope"] == validation_scope,
        }
        success = all(constraint_results.values())
        result = EvaluationResult(
            experiment_id=experiment.experiment_id,
            success=success,
            primary_score=score,
            metrics=metrics,
            constraint_results=constraint_results,
            artefact_references=artefact_references(
                self.context, experiment.experiment_id
            ),
            evaluator_version=self.version,
            provenance=ProvenanceKind.REAL,
            error=None if success else "feta_unet_scientific_constraints_failed",
        )
        return self._persist(experiment, result)
