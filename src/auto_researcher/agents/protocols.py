"""Protocols for components that make scientific proposals, never measurements."""

from typing import Protocol, runtime_checkable

from auto_researcher.agents.models import (
    AgentCallTelemetry,
    HypothesisAgentContext,
    PlannerAgentContext,
    ResearchDirective,
    ResearchDirectorContext,
)
from auto_researcher.contracts.models import Hypothesis, SearchRequest


@runtime_checkable
class HypothesisAgent(Protocol):
    def generate(self, context: HypothesisAgentContext) -> Hypothesis: ...


@runtime_checkable
class PlannerAgent(Protocol):
    def plan(self, context: PlannerAgentContext) -> SearchRequest: ...


@runtime_checkable
class ResearchDirectorAgent(Protocol):
    def decide(self, context: ResearchDirectorContext) -> ResearchDirective: ...


@runtime_checkable
class AgentTelemetrySource(Protocol):
    def consume_telemetry(self) -> AgentCallTelemetry | None: ...
