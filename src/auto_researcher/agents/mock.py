"""Offline agents whose outputs are stable functions of their inputs."""

from __future__ import annotations

import hashlib

from auto_researcher.agents.models import HypothesisAgentContext, PlannerAgentContext
from auto_researcher.contracts.enums import (
    GroundingStatus,
    HypothesisStatus,
    ProposalSource,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import Hypothesis, ResearchContract, SearchRequest


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


class MockHypothesisAgent:
    """A deterministic stand-in for a future live hypothesis agent."""

    def generate(
        self,
        context: HypothesisAgentContext | ResearchContract,
        *,
        cycle: int | None = None,
    ) -> Hypothesis:
        contract = context.contract if isinstance(context, HypothesisAgentContext) else context
        active_cycle = context.cycle if isinstance(context, HypothesisAgentContext) else cycle
        assert active_cycle is not None
        hypothesis_id = _stable_id(
            "hyp",
            contract.contract_id,
            contract.objective_version,
            str(active_cycle),
        )
        return Hypothesis(
            hypothesis_id=hypothesis_id,
            statement=f"A bounded task configuration can improve {contract.objective}.",
            rationale=(
                "The active task exposes a bounded deterministic configuration "
                "suitable for testing the control-plane invariants."
            ),
            predicted_subspace={"candidate_region": "task-defined bounded configuration"},
            expected_observation="The task primary metric satisfies its registered policy.",
            falsification_condition="The task verification policy rejects the observation.",
            evidence_references=(),
            prior_weight=0.5,
            status=HypothesisStatus.OPEN,
            provenance=ProvenanceKind.MOCK,
            proposal_source=ProposalSource.DETERMINISTIC,
            grounding_status=GroundingStatus.CONTRACT_GROUNDED,
        )


class MockPlannerAgent:
    """A deterministic DIRECT planner; a search type can be injected for routing tests."""

    def __init__(
        self,
        search_type: SearchType = SearchType.DIRECT,
        configuration: dict | None = None,
        experiment_budget: int = 1,
    ) -> None:
        self.search_type = search_type
        self.experiment_budget = experiment_budget
        self.configuration = (
            configuration
            if configuration is not None
            else {
                "model_family": "linear",
                "complexity": 4,
                "learning_rate": 0.05,
            }
        )

    def plan(
        self,
        context: PlannerAgentContext | ResearchContract,
        hypothesis: Hypothesis | None = None,
        *,
        cycle: int | None = None,
    ) -> SearchRequest:
        if isinstance(context, PlannerAgentContext):
            contract = context.contract
            hypothesis = context.hypothesis
            active_cycle = context.cycle
        else:
            contract = context
            active_cycle = cycle
        assert hypothesis is not None and active_cycle is not None
        request_id = _stable_id(
            "search",
            contract.contract_id,
            hypothesis.hypothesis_id,
            str(active_cycle),
            self.search_type.value,
        )
        return SearchRequest(
            request_id=request_id,
            hypothesis_id=hypothesis.hypothesis_id,
            search_type=self.search_type,
            target="maximise primary_score while satisfying all declared constraints",
            search_space=self.configuration,
            experiment_budget=self.experiment_budget,
            rationale="Run one bounded, deterministic experiment for the proposed hypothesis.",
            requires_human_approval=self.search_type in contract.requires_approval_for,
            proposal_source=ProposalSource.DETERMINISTIC,
            grounding_status=hypothesis.grounding_status,
        )


class ConfiguredPlannerAgent(MockPlannerAgent):
    """Deterministic planner fed a task-normalised DIRECT configuration."""

    def __init__(
        self,
        configuration: dict,
        search_type: SearchType = SearchType.DIRECT,
        experiment_budget: int = 1,
    ) -> None:
        super().__init__(
            search_type=search_type,
            configuration=configuration,
            experiment_budget=experiment_budget,
        )
