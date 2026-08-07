"""Standard-library, leakage-safe weighted k-NN evaluation on fixed Iris folds."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

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
from auto_researcher.tasks.iris_knn.configuration import IrisKNNConfiguration
from auto_researcher.tasks.iris_knn.manifests import (
    CLASS_NAMES,
    DATASET_VERSION,
    DATA_SHA256,
    FOLD_SHA256,
    FOLD_VERSION,
    IrisRow,
    load_fold_assignments,
    load_iris_rows,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)
from auto_researcher.tasks.scientific_json import SCIENTIFIC_JSON_ENCODING_VERSION

EVALUATOR_ID = "iris-weighted-knn-evaluator"
EVALUATOR_VERSION = "iris-weighted-knn-evaluator-v1"
PREPROCESSING_VERSION = "train-fold-zscore-v1"
DISTANCE_VERSION = "weighted-minkowski-v1"
TIE_BREAK_VERSION = "vote-count-then-distance-then-class-v1"
METRIC_VERSION = "mean-five-fold-balanced-accuracy-v1"
EVALUATOR_CODE_VERSION = (
    "iris-weighted-knn-task-1.0"
    f"+dataset-{DATASET_VERSION}"
    f"+fold-{FOLD_VERSION}"
    f"+{PREPROCESSING_VERSION}"
    f"+{DISTANCE_VERSION}"
    f"+{TIE_BREAK_VERSION}"
    f"+{METRIC_VERSION}"
    f"+{SCIENTIFIC_JSON_ENCODING_VERSION}"
    f"+{ARTEFACT_BUNDLE_SCHEMA_VERSION}"
)


def fit_standardisation(
    rows: Sequence[IrisRow],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Fit means and population standard deviations using training rows only."""

    if not rows:
        raise ValueError("iris_training_fold_empty")
    means = tuple(
        sum(row.features[index] for row in rows) / len(rows) for index in range(4)
    )
    scales = tuple(
        math.sqrt(
            sum((row.features[index] - means[index]) ** 2 for row in rows) / len(rows)
        )
        or 1.0
        for index in range(4)
    )
    return means, scales


def transform_features(
    features: Sequence[float], means: Sequence[float], scales: Sequence[float]
) -> tuple[float, float, float, float]:
    return tuple(
        (float(value) - float(mean)) / float(scale)
        for value, mean, scale in zip(features, means, scales, strict=True)
    )  # type: ignore[return-value]


def weighted_distance(
    left: Sequence[float],
    right: Sequence[float],
    weights: Sequence[float],
    power: int,
) -> float:
    total = sum(
        float(weight) * abs(float(a) - float(b)) ** power
        for a, b, weight in zip(left, right, weights, strict=True)
    )
    return total if power == 1 else math.sqrt(total)


def choose_label(neighbours: Sequence[tuple[float, int, str]]) -> str:
    """Break vote ties by total distance, then canonical species order."""

    if not neighbours:
        raise ValueError("iris_neighbour_set_empty")
    votes = Counter(label for _, _, label in neighbours)
    distances = {
        label: sum(distance for distance, _, item in neighbours if item == label)
        for label in CLASS_NAMES
    }
    return min(
        CLASS_NAMES,
        key=lambda label: (-votes[label], distances[label], CLASS_NAMES.index(label)),
    )


def predict_label(
    training: Sequence[tuple[int, tuple[float, float, float, float], str]],
    features: tuple[float, float, float, float],
    configuration: IrisKNNConfiguration,
) -> str:
    neighbours = sorted(
        (
            weighted_distance(
                row_features,
                features,
                configuration.feature_weights,
                configuration.distance_power,
            ),
            row_index,
            label,
        )
        for row_index, row_features, label in training
    )[: configuration.k]
    return choose_label(neighbours)


def balanced_accuracy(
    actual: Sequence[str],
    predicted: Sequence[str],
    classes: Sequence[str] = CLASS_NAMES,
) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("balanced_accuracy_input_mismatch")
    recalls = []
    for label in classes:
        total = sum(item == label for item in actual)
        if total == 0:
            raise ValueError("balanced_accuracy_class_missing")
        recalls.append(
            sum(
                a == label and p == label
                for a, p in zip(actual, predicted, strict=True)
            )
            / total
        )
    return sum(recalls) / len(recalls)


def evaluate_configuration(
    configuration: IrisKNNConfiguration,
    rows: Sequence[IrisRow],
    assignments: Sequence[int],
) -> dict:
    fold_scores: list[float] = []
    all_actual: list[str] = []
    all_predicted: list[str] = []
    confusion = {
        actual: {predicted: 0 for predicted in CLASS_NAMES} for actual in CLASS_NAMES
    }
    for fold in range(5):
        training_rows = [
            row
            for row, assigned in zip(rows, assignments, strict=True)
            if assigned != fold
        ]
        validation_rows = [
            row
            for row, assigned in zip(rows, assignments, strict=True)
            if assigned == fold
        ]
        means, scales = fit_standardisation(training_rows)
        training = [
            (row.index, transform_features(row.features, means, scales), row.label)
            for row in training_rows
        ]
        actual: list[str] = []
        predicted: list[str] = []
        for row in validation_rows:
            prediction = predict_label(
                training,
                transform_features(row.features, means, scales),
                configuration,
            )
            actual.append(row.label)
            predicted.append(prediction)
            confusion[row.label][prediction] += 1
        fold_scores.append(balanced_accuracy(actual, predicted))
        all_actual.extend(actual)
        all_predicted.extend(predicted)
    recalls = {
        label: confusion[label][label] / sum(confusion[label].values())
        for label in CLASS_NAMES
    }
    score = sum(fold_scores) / len(fold_scores)
    return {
        "mean_balanced_accuracy": round(score, 12),
        "per_fold_balanced_accuracy": [round(value, 12) for value in fold_scores],
        "overall_accuracy": round(
            sum(a == p for a, p in zip(all_actual, all_predicted, strict=True))
            / len(all_actual),
            12,
        ),
        "aggregate_confusion_counts": confusion,
        "per_species_recall": {key: round(value, 12) for key, value in recalls.items()},
    }


class IrisKNNEvaluator:
    evaluator_id = EVALUATOR_ID
    version = EVALUATOR_VERSION
    cost_per_experiment = 0.0

    def __init__(
        self,
        context: TaskRuntimeContext,
        metadata: ExperimentMetadata,
        dataset_manifest: DatasetManifest,
    ) -> None:
        self.context = context
        self.metadata = metadata
        self.dataset_manifest = dataset_manifest

    def _manifest(self) -> dict:
        return {
            "task_id": "iris_knn",
            "task_version": "1.0",
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.version,
            "code_version": self.metadata.code_version,
            "dataset_version": self.metadata.dataset_version,
            "fold_version": FOLD_VERSION,
            "result_encoding_version": SCIENTIFIC_JSON_ENCODING_VERSION,
            "artefact_bundle_schema_version": ARTEFACT_BUNDLE_SCHEMA_VERSION,
            "provenance": self.metadata.provenance.value,
        }

    def _persist(
        self, experiment: ExperimentSpec, result: EvaluationResult
    ) -> EvaluationResult:
        try:
            write_artefact_bundle(
                self.context,
                experiment,
                result,
                self.dataset_manifest,
                self._manifest(),
            )
            return result
        except Exception as exc:
            return EvaluationResult(
                experiment_id=experiment.experiment_id,
                success=False,
                primary_score=None,
                metrics={
                    "failure_diagnostics": {
                        "safe_exception_class": type(exc).__name__,
                        "failure_stage": "ARTEFACT_WRITING",
                    }
                },
                constraint_results={},
                artefact_references=(),
                evaluator_version=self.version,
                provenance=self.metadata.provenance,
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
                provenance=self.metadata.provenance,
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
            or contract.primary_metric != "mean_balanced_accuracy"
        ):
            return self._failure(experiment, "experiment_metadata_mismatch")
        try:
            configuration = IrisKNNConfiguration.model_validate(
                experiment.configuration
            )
            rows = load_iris_rows(self.context)
            assignments = load_fold_assignments(self.context)
            metrics = evaluate_configuration(configuration, rows, assignments)
        except Exception as exc:
            return self._failure(
                experiment, f"iris_evaluation_failed:{type(exc).__name__}"
            )
        score = float(metrics["mean_balanced_accuracy"])
        metrics.update(
            {
                "configuration": configuration.scientific_configuration(),
                "configuration_identity": payload_hash(configuration),
                "dataset_version": DATASET_VERSION,
                "dataset_sha256": DATA_SHA256,
                "fold_version": FOLD_VERSION,
                "fold_sha256": FOLD_SHA256,
                "evaluator_version": self.version,
                "preprocessing_version": PREPROCESSING_VERSION,
                "distance_version": DISTANCE_VERSION,
                "tie_break_version": TIE_BREAK_VERSION,
                "metric_version": METRIC_VERSION,
                "folds_present": [0, 1, 2, 3, 4],
                "class_names": list(CLASS_NAMES),
            }
        )
        result = EvaluationResult(
            experiment_id=experiment.experiment_id,
            success=True,
            primary_score=score,
            metrics=metrics,
            constraint_results={
                "score_finite_and_bounded": math.isfinite(score)
                and 0.0 <= score <= 1.0,
                "five_folds_present": len(metrics["per_fold_balanced_accuracy"]) == 5,
                "three_classes_present": set(metrics["per_species_recall"])
                == set(CLASS_NAMES),
                "dataset_identity_exact": metrics["dataset_version"] == DATASET_VERSION,
                "fold_identity_exact": metrics["fold_version"] == FOLD_VERSION,
                "evaluator_identity_exact": metrics["evaluator_version"]
                == self.version,
                "configuration_valid": True,
            },
            artefact_references=artefact_references(
                self.context, experiment.experiment_id
            ),
            evaluator_version=self.version,
            provenance=self.metadata.provenance,
            error=None,
        )
        return self._persist(experiment, result)
