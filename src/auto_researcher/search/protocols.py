"""Protocol and capability metadata shared by search backends."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import (
    ExperimentSpec,
    ResearchContract,
    SearchRequest,
)


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
    """Generic ask/tell backend marker implemented by the PR 3 adapter."""


class OpenEvolveSearchBackend(SearchBackend, Protocol):
    """Bounded program-search boundary implemented by the PR 6 subgraph."""


@dataclass(frozen=True)
class SearchCapability:
    search_type: SearchType
    available: bool
    code: str
    message: str
