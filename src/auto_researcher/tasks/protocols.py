"""Runtime-checkable task and task-policy plugin boundaries."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from auto_researcher.contracts.models import EvaluationResult, ResearchContract
from auto_researcher.evaluation.protocols import Evaluator
from auto_researcher.tasks.models import (
    ArtefactPolicy,
    DatasetManifest,
    ExperimentMetadata,
    PolicyDecision,
    ReadinessResult,
    TaskDescriptor,
    TaskRuntimeContext,
)


@runtime_checkable
class VerificationPolicy(Protocol):
    policy_id: str
    required_metrics: frozenset[str]

    def evaluate_constraints(
        self,
        evaluation: EvaluationResult,
        contract: ResearchContract,
    ) -> PolicyDecision: ...


@runtime_checkable
class ResearchTask(Protocol):
    task_id: str
    task_version: str

    def descriptor(self) -> TaskDescriptor: ...

    def readiness(self, runtime_context: TaskRuntimeContext) -> ReadinessResult: ...

    def validate_contract(self, contract: ResearchContract) -> None: ...

    def normalise_configuration(
        self,
        configuration: dict[str, JsonValue],
    ) -> dict[str, JsonValue]: ...

    def experiment_metadata(
        self,
        runtime_context: TaskRuntimeContext,
    ) -> ExperimentMetadata: ...

    def create_evaluator(self, runtime_context: TaskRuntimeContext) -> Evaluator: ...

    def create_verification_policy(
        self,
        contract: ResearchContract,
    ) -> VerificationPolicy: ...

    def dataset_manifest(
        self,
        runtime_context: TaskRuntimeContext,
    ) -> DatasetManifest: ...

    def artefact_policy(self) -> ArtefactPolicy: ...
