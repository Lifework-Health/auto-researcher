"""Protocols for components that make scientific proposals, never measurements."""

from typing import Protocol, runtime_checkable

from auto_researcher.contracts.models import Hypothesis, ResearchContract, SearchRequest


@runtime_checkable
class HypothesisAgent(Protocol):
    def generate(self, contract: ResearchContract, *, cycle: int) -> Hypothesis: ...


@runtime_checkable
class PlannerAgent(Protocol):
    def plan(
        self,
        contract: ResearchContract,
        hypothesis: Hypothesis,
        *,
        cycle: int,
    ) -> SearchRequest: ...
