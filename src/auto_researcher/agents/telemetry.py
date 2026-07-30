"""Safe transfer of model usage from agents to deterministic graph budget state."""

from auto_researcher.agents.models import AgentCallTelemetry
from auto_researcher.agents.protocols import AgentTelemetrySource
from auto_researcher.contracts.models import BudgetState


def consume_agent_telemetry(agent: object) -> AgentCallTelemetry | None:
    if isinstance(agent, AgentTelemetrySource):
        return agent.consume_telemetry()
    return None


def apply_agent_telemetry(
    budget: BudgetState,
    telemetry: AgentCallTelemetry | None,
) -> BudgetState:
    if telemetry is None or telemetry.provider_attempts == 0:
        return budget
    return budget.record_model_usage(
        calls=telemetry.provider_attempts,
        input_tokens=telemetry.input_tokens,
        output_tokens=telemetry.output_tokens,
        cache_creation_tokens=telemetry.cache_creation_input_tokens,
        cache_read_tokens=telemetry.cache_read_input_tokens,
        cost=telemetry.estimated_cost,
    )
