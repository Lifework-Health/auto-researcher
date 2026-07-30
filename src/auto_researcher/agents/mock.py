"""Offline agents whose outputs are stable functions of their inputs."""

from __future__ import annotations

import hashlib

from auto_researcher.contracts.enums import HypothesisStatus, ProvenanceKind, SearchType
from auto_researcher.contracts.models import Hypothesis, ResearchContract, SearchRequest


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


class MockHypothesisAgent:
    """A deterministic stand-in for a future live hypothesis agent."""

    def generate(self, contract: ResearchContract, *, cycle: int) -> Hypothesis:
        hypothesis_id = _stable_id(
            "hyp",
            contract.contract_id,
            contract.objective_version,
            str(cycle),
        )
        return Hypothesis(
            hypothesis_id=hypothesis_id,
            statement=f"A bounded direct configuration can improve {contract.objective}.",
            rationale=(
                "The offline reference landscape contains a smooth, deterministic "
                "region suitable for testing the control-plane invariants."
            ),
            predicted_subspace={
                "model_depth": {"minimum": 2, "maximum": 5},
                "learning_rate": {"minimum": 0.05, "maximum": 0.2},
            },
            expected_observation="A valid configuration yields a primary score above 0.70.",
            falsification_condition="No valid direct configuration yields a score above 0.70.",
            evidence_references=(),
            prior_weight=0.5,
            status=HypothesisStatus.OPEN,
            provenance=ProvenanceKind.MOCK,
        )


class MockPlannerAgent:
    """A deterministic DIRECT planner; a search type can be injected for routing tests."""

    def __init__(self, search_type: SearchType = SearchType.DIRECT) -> None:
        self.search_type = search_type

    def plan(
        self,
        contract: ResearchContract,
        hypothesis: Hypothesis,
        *,
        cycle: int,
    ) -> SearchRequest:
        request_id = _stable_id(
            "search",
            contract.contract_id,
            hypothesis.hypothesis_id,
            str(cycle),
            self.search_type.value,
        )
        return SearchRequest(
            request_id=request_id,
            hypothesis_id=hypothesis.hypothesis_id,
            search_type=self.search_type,
            target="maximise primary_score while satisfying all declared constraints",
            search_space={
                "model_depth": [3],
                "learning_rate": [0.1],
                "regularization": [0.0],
            },
            experiment_budget=1,
            rationale="Run one bounded, deterministic experiment for the proposed hypothesis.",
            requires_human_approval=self.search_type in contract.requires_approval_for,
        )
