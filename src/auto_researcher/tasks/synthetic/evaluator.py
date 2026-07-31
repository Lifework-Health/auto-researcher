"""Deterministic synthetic task evaluator and artefact writer."""

from __future__ import annotations

from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
)
from auto_researcher.tasks.artifacts import (
    ARTEFACT_BUNDLE_SCHEMA_VERSION,
    artefact_references,
    write_artefact_bundle,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)
from auto_researcher.tasks.synthetic.configuration import SyntheticConfiguration
from auto_researcher.tasks.scientific_json import SCIENTIFIC_JSON_ENCODING_VERSION


class SyntheticEvaluator:
    evaluator_id = "synthetic-evaluator"
    version = "synthetic-landscape-v3"
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
        return self._persist_or_fail(experiment, result)

    def _persist_or_fail(
        self,
        experiment: ExperimentSpec,
        result: EvaluationResult,
    ) -> EvaluationResult:
        try:
            write_artefact_bundle(
                self.context,
                experiment,
                result,
                self.dataset_manifest,
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
                        "safe_exception_class": type(exc).__name__,
                        "failure_stage": "ARTEFACT_WRITING",
                        "artefact_persistence_failure_code": (
                            "bundle_publication_failed"
                        ),
                    }
                },
                constraint_results={},
                artefact_references=(),
                evaluator_version=self.version,
                provenance=self.metadata.provenance,
                error=f"artefact_bundle_publication_failed:{type(exc).__name__}",
            )

    def _evaluator_manifest(self) -> dict:
        return {
            "task_id": "synthetic",
            "task_version": "1.0",
            "evaluator_id": self.evaluator_id,
            "code_version": self.metadata.code_version,
            "result_encoding_version": SCIENTIFIC_JSON_ENCODING_VERSION,
            "artefact_bundle_schema_version": ARTEFACT_BUNDLE_SCHEMA_VERSION,
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
        return self._persist_or_fail(experiment, result)
