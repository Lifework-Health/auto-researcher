"""Runtime-checkable task and task-policy plugin boundaries."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
    SearchRequest,
)
from auto_researcher.search.optuna.models import OptunaStudySpec
from auto_researcher.search.optuna.pruning import OptunaIntermediateReporter
from auto_researcher.search.openevolve.protocols import EvolvableComponent
from auto_researcher.search.openevolve.live_dataset import LiveMutationDatasetClass
from auto_researcher.search.openevolve.live_boundary import (
    MetadataOnlyMutationBoundary,
)
from auto_researcher.search.protocols import SearchCapability
from auto_researcher.knowledge.models import (
    KnowledgeGroundingPolicy,
    KnowledgeQueryPlan,
)
from auto_researcher.agents.models import PriorResearchSummary, TaskAgentContext
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


@runtime_checkable
class OptunaCapableTask(Protocol):
    """Optional capability implemented only by tasks with a registered HPO space."""

    def create_optuna_study_spec(
        self,
        contract: ResearchContract,
        request: SearchRequest,
    ) -> OptunaStudySpec: ...


@runtime_checkable
class IntermediateReportingEvaluator(Protocol):
    """Optional cooperative evaluator seam for native Optuna pruning."""

    evaluator_id: str

    def evaluate_with_intermediate_reporting(
        self,
        experiment: ExperimentSpec,
        contract: ResearchContract,
        reporter: OptunaIntermediateReporter,
    ) -> EvaluationResult: ...


@runtime_checkable
class OpenEvolveCapableTask(Protocol):
    """Optional task-owned mutable surface for bounded program search."""

    def create_evolvable_component(
        self,
        contract: ResearchContract,
        runtime_context: TaskRuntimeContext,
    ) -> EvolvableComponent: ...


@runtime_checkable
class CampaignDurationCapableTask(Protocol):
    """Optional conservative wall-time estimate used before admitting a block."""

    def estimate_search_duration_seconds(
        self,
        request: SearchRequest,
        runtime_context: TaskRuntimeContext,
    ) -> float: ...


@runtime_checkable
class CampaignRequestEnrichmentCapableTask(Protocol):
    """Optional deterministic handoff from verified results into a search block."""

    def enrich_search_request(
        self,
        request: SearchRequest,
        prior_verified_findings: tuple[PriorResearchSummary, ...],
    ) -> SearchRequest: ...


@runtime_checkable
class LiveMutationDatasetClassCapableTask(Protocol):
    """Trusted opt-in classification for approved live OpenEvolve mutation."""

    def live_mutation_dataset_class(self) -> LiveMutationDatasetClass: ...


@runtime_checkable
class MetadataOnlyLiveMutationCapableTask(Protocol):
    """Explicit opt-in for the separately attested metadata-only v2 path."""

    def live_mutation_boundary(self) -> MetadataOnlyMutationBoundary: ...


@runtime_checkable
class AgentContextCapableTask(Protocol):
    """Optional safe model-facing context supplied by a task plugin."""

    def create_agent_context(
        self,
        contract: ResearchContract,
        runtime_context: TaskRuntimeContext,
        search_capabilities: dict[SearchType, SearchCapability],
    ) -> TaskAgentContext: ...


@runtime_checkable
class KnowledgeGroundingCapableTask(Protocol):
    """Optional deterministic retrieval plan owned by a scientific task."""

    def create_knowledge_query_plan(
        self,
        contract: ResearchContract,
        runtime_context: TaskRuntimeContext,
        search_capabilities: dict[SearchType, SearchCapability],
    ) -> KnowledgeQueryPlan: ...

    def create_grounding_policy(
        self,
        contract: ResearchContract,
    ) -> KnowledgeGroundingPolicy: ...
