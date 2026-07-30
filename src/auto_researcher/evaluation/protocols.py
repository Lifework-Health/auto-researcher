"""Evaluator protocol: the source of measurements, not scientific proposals."""

from typing import Protocol, runtime_checkable

from auto_researcher.contracts.models import EvaluationResult, ExperimentSpec, ResearchContract


@runtime_checkable
class Evaluator(Protocol):
    evaluator_id: str
    version: str

    def evaluate(
        self,
        experiment: ExperimentSpec,
        contract: ResearchContract,
    ) -> EvaluationResult: ...
