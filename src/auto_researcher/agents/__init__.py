"""Scientific judgement interfaces and offline deterministic agents."""

from auto_researcher.agents.mock import MockHypothesisAgent, MockPlannerAgent
from auto_researcher.agents.protocols import HypothesisAgent, PlannerAgent

__all__ = ["HypothesisAgent", "MockHypothesisAgent", "MockPlannerAgent", "PlannerAgent"]
