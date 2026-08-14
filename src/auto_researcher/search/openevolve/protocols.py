"""Task and mutation boundaries for the bounded OpenEvolve backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from auto_researcher.contracts.models import (
    ExperimentSpec,
    ResearchContract,
    SearchRequest,
)
from auto_researcher.search.openevolve.models import (
    CandidatePreparationResult,
    EvolvableComponentSpec,
    MutationReservation,
    OpenEvolveCandidate,
)
from auto_researcher.tasks.models import ExperimentMetadata


@runtime_checkable
class EvolvableComponent(Protocol):
    """The task-owned, explicitly bounded surface that may change."""

    def component_spec(self) -> EvolvableComponentSpec: ...

    def seed_configuration(self) -> dict: ...

    def candidate_to_experiment(
        self,
        candidate: OpenEvolveCandidate,
        preparation: CandidatePreparationResult,
        request: SearchRequest,
        contract: ResearchContract,
        metadata: ExperimentMetadata,
        *,
        run_id: str,
    ) -> ExperimentSpec: ...


@runtime_checkable
class ScientificCandidateComponent(Protocol):
    """Optional task-owned canonical identity projection for semantic reuse."""

    def canonical_scientific_configuration(
        self,
        preparation: CandidatePreparationResult,
    ) -> dict: ...


@runtime_checkable
class MutationOperator(Protocol):
    operator_id: str
    operator_version: str
    model_calls_per_mutation: int
    provenance: str

    def mutate(
        self,
        reservation: MutationReservation,
        parent: OpenEvolveCandidate,
        component: EvolvableComponentSpec,
    ) -> tuple[str, str, str | None]:
        """Return replacement source, bounded description and optional call ID."""


class StructuredMutationClient(Protocol):
    def propose_mutation(self, request: dict) -> dict: ...
