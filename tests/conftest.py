from __future__ import annotations

from datetime import UTC, datetime

import pytest

from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.runtime.dependencies import memory_dependencies


class SequenceIds:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, prefix: str) -> str:
        self.value += 1
        return f"{prefix}-{self.value:04d}"


def fixed_clock() -> datetime:
    return datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


@pytest.fixture
def contract_factory():
    def make(
        *,
        allowed: frozenset[SearchType] = frozenset({SearchType.DIRECT}),
        approval: frozenset[SearchType] = frozenset(),
        maximum_cycles: int = 1,
        maximum_experiments: int = 1,
        maximum_cost: float = 10.0,
    ) -> ResearchContract:
        return ResearchContract(
            contract_id="contract-test",
            schema_version="1.0",
            objective_version="objective-v1",
            question="Which bounded configuration performs best?",
            objective="maximise the deterministic primary score",
            constraints={"score_floor": 0.7, "nested": {"values": [1, 2]}},
            allowed_search_types=allowed,
            evaluator_id="mock-evaluator",
            verifier_id="deterministic-verifier",
            maximum_cycles=maximum_cycles,
            maximum_experiments=maximum_experiments,
            maximum_cost=maximum_cost,
            requires_approval_for=approval,
            provenance=ProvenanceKind.MOCK,
        )

    return make


@pytest.fixture
def deterministic_dependencies():
    return memory_dependencies(clock=fixed_clock, id_generator=SequenceIds())
