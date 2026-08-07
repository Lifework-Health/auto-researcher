"""Trusted FeTA SegResNet evaluator boundary."""

from __future__ import annotations

import math

from auto_researcher.contracts.enums import ProvenanceKind
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
)
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.artifacts import artefact_references, write_artefact_bundle
from auto_researcher.tasks.feta_seg.configuration import FeTASegConfiguration
from auto_researcher.tasks.feta_seg.metrics import (
    LABEL_NAMES,
    aggregate_subject_metrics,
)
from auto_researcher.tasks.feta_seg.splits import FOLD_ID, SPLIT_ID
from auto_researcher.tasks.feta_seg.transforms import (
    AUGMENTATION_VERSION,
    PREPROCESSING_VERSION,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)

EVALUATOR_ID = "feta-segresnet-evaluator"
EVALUATOR_VERSION = "feta-segresnet-evaluator-v1"
ARCHITECTURE_VERSION = "monai-segresnet-3d-32-1224-111-v1"
LOSS_VERSION = "dice-ce-softmax-onehot-no-background-equal-v1"
OPTIMISER_VERSION = "adamw-lr1e-4-wd1e-5-v1"
INFERENCE_VERSION = "sliding-window-128-overlap0.5-gaussian-v1"
METRIC_VERSION = "mean-subject-macro-dice-labels1-7-v1"


def evaluator_code_version(dataset_version: str) -> str:
    return "+".join(
        (
            "feta-seg-task-1.0",
            dataset_version,
            SPLIT_ID,
            FOLD_ID,
            PREPROCESSING_VERSION,
            AUGMENTATION_VERSION,
            ARCHITECTURE_VERSION,
            LOSS_VERSION,
            OPTIMISER_VERSION,
            INFERENCE_VERSION,
            METRIC_VERSION,
        )
    )


class FeTASegEvaluator:
    evaluator_id = EVALUATOR_ID
    version = EVALUATOR_VERSION
    cost_per_experiment = 0.0

    def __init__(
        self,
        context: TaskRuntimeContext,
        metadata: ExperimentMetadata,
        manifest: DatasetManifest,
    ) -> None:
        self.context = context
        self.metadata = metadata
        self.manifest = manifest

    def _persist(
        self, experiment: ExperimentSpec, result: EvaluationResult
    ) -> EvaluationResult:
        try:
            write_artefact_bundle(
                self.context,
                experiment,
                result,
                self.manifest,
                {
                    "task_id": "feta_seg",
                    "task_version": "1.0",
                    "evaluator_id": self.evaluator_id,
                    "evaluator_version": self.version,
                    "code_version": self.metadata.code_version,
                    "dataset_version": self.metadata.dataset_version,
                    "split_identity": SPLIT_ID,
                    "fold_identity": FOLD_ID,
                    "holdout_evaluator_calls": 0,
                },
            )
            return result
        except Exception as exc:
            return EvaluationResult(
                experiment_id=experiment.experiment_id,
                success=False,
                primary_score=None,
                metrics={
                    "failure_stage": "ARTEFACT_WRITING",
                    "safe_exception_class": type(exc).__name__,
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
        try:
            configuration = FeTASegConfiguration.model_validate(
                experiment.configuration
            )
        except Exception:
            return self._failure(experiment, "feta_configuration_invalid")
        if (
            contract.primary_metric != "mean_subject_macro_dice"
            or experiment.dataset_version != self.metadata.dataset_version
        ):
            return self._failure(experiment, "feta_experiment_metadata_mismatch")
        if configuration.mode == "full":
            try:
                import torch
            except ImportError:
                return self._failure(experiment, "feta_ml_dependencies_unavailable")
            if not torch.cuda.is_available():
                return self._failure(
                    experiment, "feta_cuda_unavailable_for_full_baseline"
                )
            return self._failure(experiment, "feta_full_baseline_runner_not_invoked")

        # Generated-data mechanics only: deliberately not a scientific baseline.
        subjects = [
            {
                "subject_id": f"smoke-{index}",
                "reconstruction_method": method,
                "dice": {str(label): 0.5 + index * 0.01 for label in LABEL_NAMES},
            }
            for index, method in enumerate(("mial", "mial", "irtk", "irtk"))
        ]
        metrics = aggregate_subject_metrics(subjects)
        score = float(metrics["mean_subject_macro_dice"])
        metrics.update(
            {
                "scientific_baseline": False,
                "mode": "smoke",
                "configuration": configuration.scientific_configuration(),
                "configuration_identity": payload_hash(configuration),
                "dataset_version": self.metadata.dataset_version,
                "split_identity": SPLIT_ID,
                "fold_identity": FOLD_ID,
                "evaluator_version": self.version,
                "preprocessing_version": PREPROCESSING_VERSION,
                "architecture_version": ARCHITECTURE_VERSION,
                "loss_version": LOSS_VERSION,
                "optimiser_version": OPTIMISER_VERSION,
                "inference_version": INFERENCE_VERSION,
                "metric_version": METRIC_VERSION,
                "folds_completed": 1,
                "oof_subject_count": 4,
                "holdout_subjects_evaluated": 0,
                "failed_training_folds": 0,
                "valid_prediction_labels": list(range(8)),
            }
        )
        result = EvaluationResult(
            experiment_id=experiment.experiment_id,
            success=True,
            primary_score=score,
            metrics=metrics,
            constraint_results={
                "score_finite_and_bounded": math.isfinite(score) and 0 <= score <= 1,
                "holdout_sealed": True,
                "all_smoke_folds_complete": True,
                "valid_prediction_labels": True,
                "scientific_baseline_false": True,
            },
            artefact_references=artefact_references(
                self.context, experiment.experiment_id
            ),
            evaluator_version=self.version,
            provenance=ProvenanceKind.REAL,
            error=None,
        )
        return self._persist(experiment, result)
