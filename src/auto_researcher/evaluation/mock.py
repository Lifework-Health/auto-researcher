"""A deterministic offline landscape with valid and constraint-violating regions."""

from __future__ import annotations

from auto_researcher.contracts.enums import ProvenanceKind
from auto_researcher.contracts.models import EvaluationResult, ExperimentSpec, ResearchContract


class MockEvaluator:
    evaluator_id = "mock-evaluator"
    version = "mock-landscape-v1"
    provenance = ProvenanceKind.MOCK
    cost_per_experiment = 0.0

    def evaluate(
        self,
        experiment: ExperimentSpec,
        contract: ResearchContract,
    ) -> EvaluationResult:
        config = experiment.configuration
        try:
            depth = int(config["model_depth"])
            learning_rate = float(config["learning_rate"])
            regularization = float(config.get("regularization", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            return EvaluationResult(
                experiment_id=experiment.experiment_id,
                success=False,
                primary_score=None,
                metrics={},
                constraint_results={"configuration_complete": False},
                evaluator_version=self.version,
                provenance=self.provenance,
                error=f"invalid_configuration: {exc}",
            )

        constraint_results = {
            "model_depth_in_range": 1 <= depth <= 8,
            "learning_rate_in_range": 0.0 < learning_rate <= 1.0,
            "regularization_non_negative": regularization >= 0.0,
        }
        # Known optimum: depth=3, learning_rate=.1, regularization=0 -> .82.
        score = round(
            max(
                0.0,
                0.82
                - 0.025 * abs(depth - 3)
                - 0.5 * abs(learning_rate - 0.1)
                - 0.1 * regularization,
            ),
            6,
        )
        stability = round(max(0.0, 0.95 - 0.03 * abs(depth - 3)), 6)
        return EvaluationResult(
            experiment_id=experiment.experiment_id,
            success=True,
            primary_score=score,
            metrics={"primary_score": score, "stability": stability},
            constraint_results=constraint_results,
            artefact_references=(),
            evaluator_version=self.version,
            provenance=self.provenance,
            error=None,
        )
