"""Deterministic synthetic task evaluator and artefact writer."""

from __future__ import annotations

from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
)
from auto_researcher.tasks.artifacts import (
    artefact_references,
    write_artefact_bundle,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)
from auto_researcher.tasks.synthetic.configuration import SyntheticConfiguration


class SyntheticEvaluator:
    evaluator_id = "synthetic-evaluator"
    version = "synthetic-landscape-v2"
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

    def _failure(self, experiment: ExperimentSpec, message: str) -> EvaluationResult:
        result = EvaluationResult(
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
            error=message,
        )
        write_artefact_bundle(
            self.context,
            experiment,
            result,
            self.dataset_manifest,
            self._evaluator_manifest(),
        )
        return result

    def _evaluator_manifest(self) -> dict:
        return {
            "task_id": "synthetic",
            "task_version": "1.0",
            "evaluator_id": self.evaluator_id,
            "code_version": self.metadata.code_version,
            "dataset_version": self.metadata.dataset_version,
            "provenance": self.metadata.provenance.value,
        }

    def evaluate(
        self,
        experiment: ExperimentSpec,
        contract: ResearchContract,
    ) -> EvaluationResult:
        if (
            experiment.evaluator_id != self.metadata.evaluator_id
            or experiment.code_version != self.metadata.code_version
            or experiment.dataset_version != self.metadata.dataset_version
            or experiment.provenance != self.metadata.provenance
        ):
            return self._failure(experiment, "experiment_metadata_mismatch")
        try:
            config = SyntheticConfiguration.model_validate(experiment.configuration)
        except Exception as exc:
            return self._failure(experiment, f"invalid_configuration: {exc}")

        family_bonus = {"linear": 0.0, "tree": 0.06, "neural": 0.1}[config.model_family]
        score = round(
            max(
                0.0,
                min(
                    1.0,
                    0.78
                    + family_bonus
                    - 0.035 * abs(config.complexity - 4)
                    - 0.4 * abs(config.learning_rate - 0.05),
                ),
            ),
            6,
        )
        stability = round(max(0.0, 0.94 - 0.025 * abs(config.complexity - 4)), 6)
        runtime = round(0.2 * config.complexity + 2.0 * config.learning_rate, 6)
        constraint_results = {
            "complexity_within_task_limit": config.complexity <= 8,
            "runtime_within_limit": runtime
            <= float(contract.constraints.get("maximum_runtime", 3.0)),
        }
        result = EvaluationResult(
            experiment_id=experiment.experiment_id,
            success=True,
            primary_score=score,
            metrics={
                "objective_score": score,
                "stability": stability,
                "runtime": runtime,
                "configuration": config.model_dump(mode="json"),
            },
            constraint_results=constraint_results,
            artefact_references=artefact_references(
                self.context, experiment.experiment_id
            ),
            evaluator_version=self.version,
            provenance=self.metadata.provenance,
            error=None,
        )
        write_artefact_bundle(
            self.context,
            experiment,
            result,
            self.dataset_manifest,
            self._evaluator_manifest(),
        )
        return result
