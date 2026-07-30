"""Protocol shared by current and future search backends."""

from typing import Protocol, runtime_checkable

from auto_researcher.contracts.models import ExperimentSpec, ResearchContract, SearchRequest


@runtime_checkable
class SearchBackend(Protocol):
    def create_experiment(
        self,
        request: SearchRequest,
        contract: ResearchContract,
        *,
        run_id: str,
    ) -> ExperimentSpec: ...


class OptunaSearchBackend(SearchBackend, Protocol):
    """Future ask/tell backend boundary. No PR 2 implementation exists."""


class OpenEvolveSearchBackend(SearchBackend, Protocol):
    """Future program-search boundary. No PR 2 implementation exists."""
