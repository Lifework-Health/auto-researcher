"""Trusted FeTA SegResNet evaluator boundary."""

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
from auto_researcher.tasks.feta_seg.configuration import FeTASegConfiguration
from auto_researcher.tasks.feta_seg.manifests import EXPECTED_MANIFEST_HASH
from auto_researcher.tasks.feta_seg.metrics import (
    EMPTY_PREDICTION_VERSION,
    HD95_VERSION,
    LABELS,
    LABEL_NAMES,
    METRIC_PANEL_VERSION,
    TOPOLOGY_VERSION,
    aggregate_subject_metrics,
)
from auto_researcher.tasks.feta_seg.runner import RUNNER_VERSION, run_full_baseline
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

EVALUATOR_ID = "feta-segresnet-evaluator"
EVALUATOR_VERSION = "feta-segresnet-evaluator-v2"
ARCHITECTURE_VERSION = "monai-segresnet-3d-32-1224-111-v1"
LOSS_VERSION = "dice-ce-softmax-onehot-no-background-equal-v1"
OPTIMISER_VERSION = "adamw-lr1e-4-wd1e-5-v1"
INFERENCE_VERSION = "sliding-window-128-overlap0.5-gaussian-native-restore-v2"
METRIC_VERSION = METRIC_PANEL_VERSION
FETA_SCIENTIFIC_JSON_POLICY = ScientificJsonPolicy(permitted_nan_paths=frozenset())

FullRunner = Callable[[TaskRuntimeContext, FeTASegConfiguration, str], dict[str, Any]]


def evaluator_code_version(dataset_version: str) -> str:
    return "+".join(
        (
            "feta-seg-task-1.0",
            dataset_version,
            SPLIT_ID,
            EXPECTED_SPLIT_HASH,
            FOLD_ID,
            EXPECTED_FOLD_HASH,
            PREPROCESSING_VERSION,
            AUGMENTATION_VERSION,
            ARCHITECTURE_VERSION,
            LOSS_VERSION,
            OPTIMISER_VERSION,
            INFERENCE_VERSION,
            METRIC_VERSION,
            HD95_VERSION,
            EMPTY_PREDICTION_VERSION,
            TOPOLOGY_VERSION,
            RUNNER_VERSION,
            SCIENTIFIC_JSON_ENCODING_VERSION,
            ARTEFACT_BUNDLE_SCHEMA_VERSION,
        )
    )


def _generated_smoke_metrics() -> dict[str, Any]:
    subjects: list[dict[str, Any]] = []
    for index, method in enumerate(("mial", "mial", "irtk", "irtk")):
        value = 0.5 + index * 0.01
        per_class = {
            str(label): {
                "label_name": LABEL_NAMES[label],
                "dice": value,
                "hd95_mm": 1.0,
                "volume_similarity": value,
                "betti": [2 if label == 2 else 1, 0, 0],
                "expected_betti": [2 if label == 2 else 1, 0, 0],
                "euler_characteristic": 2 if label == 2 else 1,
                "expected_euler_characteristic": 2 if label == 2 else 1,
                "euler_distance": 0,
                "empty_prediction": False,
            }
            for label in LABELS
        }
        subjects.append(
            {
                "subject_id": f"smoke-{index}",
                "reconstruction_method": method,
                "fold": 0,
                "per_class": per_class,
                "macro_dice": value,
                "macro_hd95_mm": 1.0,
                "macro_volume_similarity": value,
                "macro_euler_distance": 0.0,
                "empty_prediction_count": 0,
            }
        )
    return aggregate_subject_metrics(subjects)


class FeTASegEvaluator:
    evaluator_id = EVALUATOR_ID
    version = EVALUATOR_VERSION
    cost_per_experiment = 0.0

    def __init__(
        self,
        context: TaskRuntimeContext,
        metadata: ExperimentMetadata,
        manifest: DatasetManifest,
        *,
        full_runner: FullRunner = run_full_baseline,
    ) -> None:
        self.context = context
        self.metadata = metadata
        self.manifest = manifest
        self.full_runner = full_runner

    def _evaluator_manifest(self) -> dict[str, Any]:
        return {
            "task_id": "feta_seg",
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
            "metric_panel_version": METRIC_VERSION,
            "hd95_version": HD95_VERSION,
            "empty_prediction_version": EMPTY_PREDICTION_VERSION,
            "topology_version": TOPOLOGY_VERSION,
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
            return self._failure(experiment, "feta_experiment_metadata_mismatch")
        try:
            configuration = FeTASegConfiguration.model_validate(
                experiment.configuration
            )
        except Exception:
            return self._failure(experiment, "feta_configuration_invalid")

        try:
            if configuration.mode == "full":
                metrics = self.full_runner(
                    self.context, configuration, experiment.experiment_id
                )
                scientific_baseline = True
            else:
                metrics = _generated_smoke_metrics()
                metrics.update(
                    {
                        "folds_completed": 1,
                        "oof_subject_count": 4,
                        "holdout_subjects_evaluated": 0,
                        "failed_training_folds": 0,
                        "valid_prediction_labels": list(range(8)),
                    }
                )
                scientific_baseline = False
        except Exception as exc:
            safe_reason = str(exc)
            safe_codes = {
                "feta_cuda_unavailable_for_full_baseline",
                "feta_ml_dependencies_unavailable",
                "feta_full_runner_paths_missing",
                "feta_dataset_identity_mismatch",
                "feta_split_identity_mismatch",
                "feta_fold_identity_mismatch",
                "feta_holdout_accessed",
                "feta_oof_membership_invalid",
                "feta_oof_coverage_invalid",
                "feta_training_loss_non_finite",
            }
            return self._failure(
                experiment,
                safe_reason
                if safe_reason in safe_codes
                else f"feta_evaluation_failed:{type(exc).__name__}",
            )

        metrics.update(
            {
                "scientific_baseline": scientific_baseline,
                "mode": configuration.mode,
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
                "preprocessing_version": PREPROCESSING_VERSION,
                "architecture_version": ARCHITECTURE_VERSION,
                "loss_version": LOSS_VERSION,
                "optimiser_version": OPTIMISER_VERSION,
                "inference_version": INFERENCE_VERSION,
                "metric_version": METRIC_VERSION,
                "hd95_version": HD95_VERSION,
                "empty_prediction_version": EMPTY_PREDICTION_VERSION,
                "topology_version": TOPOLOGY_VERSION,
            }
        )
        try:
            metrics = require_valid_scientific_json(
                normalise_scientific_json(metrics, policy=FETA_SCIENTIFIC_JSON_POLICY),
                reason_code="feta_scientific_json_invalid",
            )
        except (ScientificJsonNormalisationError, TypeError):
            return self._failure(experiment, "feta_scientific_json_invalid")

        score = float(metrics["mean_subject_macro_dice"])
        expected_folds, expected_subjects = (
            (5, 68) if configuration.mode == "full" else (1, 4)
        )
        required_metric_names = {
            "mean_subject_macro_dice",
            "mean_subject_macro_hd95_mm",
            "mean_subject_macro_volume_similarity",
            "mean_subject_macro_euler_distance",
            "per_class_summary",
            "subject_metrics",
            "reconstruction_macro_dice",
            "reconstruction_gap",
            "empty_prediction_count",
        }
        constraint_results = {
            "score_finite_and_bounded": math.isfinite(score) and 0 <= score <= 1,
            "all_development_folds_complete": metrics["folds_completed"]
            == expected_folds,
            "holdout_sealed": metrics["holdout_subjects_evaluated"] == 0,
            "valid_prediction_labels": metrics["valid_prediction_labels"]
            == list(range(8)),
            "empty_prediction_count_valid": 0
            <= int(metrics["empty_prediction_count"])
            <= expected_subjects * len(LABELS),
            "required_metrics_complete": required_metric_names.issubset(metrics),
            "dataset_identity_exact": (
                configuration.mode == "smoke"
                or metrics["dataset_manifest_hash"] == EXPECTED_MANIFEST_HASH
            ),
            "split_identity_exact": metrics["split_identity"] == SPLIT_ID
            and metrics["split_hash"] == EXPECTED_SPLIT_HASH,
            "fold_identity_exact": metrics["fold_identity"] == FOLD_ID
            and metrics["fold_hash"] == EXPECTED_FOLD_HASH,
            "evaluator_identity_exact": metrics["evaluator_version"] == self.version
            and metrics["evaluator_code_version"] == self.metadata.code_version,
            "oof_subject_count_exact": metrics["oof_subject_count"]
            == expected_subjects,
            "scientific_identity_correct": metrics["scientific_baseline"]
            is (configuration.mode == "full"),
        }
        result = EvaluationResult(
            experiment_id=experiment.experiment_id,
            success=all(constraint_results.values()),
            primary_score=score,
            metrics=metrics,
            constraint_results=constraint_results,
            artefact_references=artefact_references(
                self.context, experiment.experiment_id
            ),
            evaluator_version=self.version,
            provenance=ProvenanceKind.REAL,
            error=None
            if all(constraint_results.values())
            else "feta_scientific_constraints_failed",
        )
        return self._persist(experiment, result)
