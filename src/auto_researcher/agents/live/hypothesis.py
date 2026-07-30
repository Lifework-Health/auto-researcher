"""Bounded live hypothesis proposal and deterministic reconciliation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from auto_researcher.agents.call_store import AgentCallStore
from auto_researcher.agents.live.base import BoundedStructuredCall
from auto_researcher.agents.models import (
    AgentBudgetPolicy,
    AgentCallTelemetry,
    HypothesisAgentContext,
    HypothesisProposal,
    ModelCallConfig,
)
from auto_researcher.agents.prompts import load_prompt
from auto_researcher.agents.reconciliation import HypothesisReconciler
from auto_researcher.contracts.enums import AgentRole
from auto_researcher.contracts.models import Hypothesis
from auto_researcher.providers.protocols import StructuredModelClient


class LiveHypothesisAgent:
    def __init__(
        self,
        *,
        client: StructuredModelClient,
        call_config: ModelCallConfig,
        budget_policy: AgentBudgetPolicy,
        call_store: AgentCallStore,
        clock: Callable[[], datetime],
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
        self._prompt = load_prompt("hypothesis", call_config.prompt_version)
        self._reconciler = HypothesisReconciler()
        self._telemetry: AgentCallTelemetry | None = None

    def generate(self, context: HypothesisAgentContext) -> Hypothesis:
        try:
            hypothesis, telemetry = self._call.run(
                run_id=context.run_id,
                cycle=context.cycle,
                role=AgentRole.HYPOTHESIS,
                context_hash=context.context_hash,
                context_json=context.model_dump_json(),
                remaining_cost_budget=context.remaining_cost_budget,
                model_calls_used=context.model_calls_used,
                prompt=self._prompt,
                response_model=HypothesisProposal,
                reconcile=lambda proposal, call_id: self._reconciler.reconcile(
                    proposal,
                    context,
                    call_id=call_id,
                    prompt_version=self._prompt.version,
                ),
            )
            self._telemetry = telemetry.model_copy(
                update={"grounding_status": hypothesis.grounding_status}
            )
            return hypothesis
        except Exception as exc:
            telemetry = getattr(exc, "telemetry", None)
            if telemetry is not None:
                self._telemetry = telemetry
            raise

    def consume_telemetry(self) -> AgentCallTelemetry | None:
        telemetry, self._telemetry = self._telemetry, None
        return telemetry
