"""Bounded live search planning and deterministic reconciliation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from auto_researcher.agents.call_store import AgentCallStore
from auto_researcher.agents.live.base import BoundedStructuredCall
from auto_researcher.agents.models import (
    AgentBudgetPolicy,
    AgentCallTelemetry,
    ModelCallConfig,
    PlannerAgentContext,
    PlannerProposal,
)
from auto_researcher.agents.prompts import load_prompt
from auto_researcher.agents.reconciliation import PlannerReconciler
from auto_researcher.contracts.enums import AgentRole
from auto_researcher.contracts.models import ResearchContract, SearchRequest
from auto_researcher.providers.protocols import StructuredModelClient
from auto_researcher.tasks.protocols import ResearchTask


class LivePlannerAgent:
    def __init__(
        self,
        *,
        client: StructuredModelClient,
        call_config: ModelCallConfig,
        budget_policy: AgentBudgetPolicy,
        call_store: AgentCallStore,
        clock: Callable[[], datetime],
        task: ResearchTask,
        contract: ResearchContract,
    ) -> None:
        self.provider = call_config.provider
        self.model_id = call_config.model_id
        self._call = BoundedStructuredCall(
            client=client,
            config=call_config,
            budget_policy=budget_policy,
            store=call_store,
            clock=clock,
        )
        self._prompt = load_prompt("planner", call_config.prompt_version)
        self._reconciler = PlannerReconciler(task, contract)
        self._telemetry: AgentCallTelemetry | None = None

    def plan(self, context: PlannerAgentContext) -> SearchRequest:
        try:
            request, telemetry = self._call.run(
                run_id=context.run_id,
                cycle=context.cycle,
                role=AgentRole.PLANNER,
                context_hash=context.context_hash,
                context_json=context.model_dump_json(),
                remaining_cost_budget=context.remaining_cost_budget,
                model_calls_used=context.model_calls_used,
                prompt=self._prompt,
                response_model=PlannerProposal,
                reconcile=lambda proposal, call_id: self._reconciler.reconcile(
                    proposal,
                    context,
                    call_id=call_id,
                    prompt_version=self._prompt.version,
                ),
            )
            self._telemetry = telemetry.model_copy(
                update={"grounding_status": request.grounding_status}
            )
            return request
        except Exception as exc:
            telemetry = getattr(exc, "telemetry", None)
            if telemetry is not None:
                self._telemetry = telemetry
            raise

    def consume_telemetry(self) -> AgentCallTelemetry | None:
        telemetry, self._telemetry = self._telemetry, None
        return telemetry
