"""Explicit runtime dependency injection; graph nodes contain no global services."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from auto_researcher.agents.mock import MockHypothesisAgent, MockPlannerAgent
from auto_researcher.agents.protocols import HypothesisAgent, PlannerAgent
from auto_researcher.evaluation.mock import MockEvaluator
from auto_researcher.evaluation.protocols import Evaluator
from auto_researcher.provenance.protocols import ProvenanceStore
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore
from auto_researcher.runtime.checkpoints import memory_checkpointer, sqlite_checkpointer
from auto_researcher.search.direct import DirectSearchBackend
from auto_researcher.search.protocols import SearchBackend
from auto_researcher.verification.verifier import DeterministicVerifier, Verifier


@dataclass(frozen=True)
class RuntimeDependencies:
    hypothesis_agent: HypothesisAgent
    planner_agent: PlannerAgent
    direct_search_backend: SearchBackend
    evaluator: Evaluator
    verifier: Verifier
    provenance_store: ProvenanceStore
    checkpointer: Any
    clock: Callable[[], datetime]
    id_generator: Callable[[str], str]


def utc_now() -> datetime:
    return datetime.now(UTC)


def random_id(prefix: str) -> str:
    return f"{prefix}-{uuid4()}"


def memory_dependencies(
    *,
    hypothesis_agent: HypothesisAgent | None = None,
    planner_agent: PlannerAgent | None = None,
    evaluator: Evaluator | None = None,
    verifier: Verifier | None = None,
    provenance_store: ProvenanceStore | None = None,
    clock: Callable[[], datetime] = utc_now,
    id_generator: Callable[[str], str] = random_id,
) -> RuntimeDependencies:
    return RuntimeDependencies(
        hypothesis_agent=hypothesis_agent or MockHypothesisAgent(),
        planner_agent=planner_agent or MockPlannerAgent(),
        direct_search_backend=DirectSearchBackend(),
        evaluator=evaluator or MockEvaluator(),
        verifier=verifier or DeterministicVerifier(),
        provenance_store=provenance_store or SQLiteProvenanceStore(),
        checkpointer=memory_checkpointer(),
        clock=clock,
        id_generator=id_generator,
    )


@contextmanager
def sqlite_dependencies(
    checkpoint_path: str | Path,
    provenance_path: str | Path,
    *,
    hypothesis_agent: HypothesisAgent | None = None,
    planner_agent: PlannerAgent | None = None,
    evaluator: Evaluator | None = None,
    verifier: Verifier | None = None,
    clock: Callable[[], datetime] = utc_now,
    id_generator: Callable[[str], str] = random_id,
) -> Iterator[RuntimeDependencies]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    provenance = Path(provenance_path).expanduser().resolve()
    if checkpoint == provenance:
        raise ValueError("checkpoint and provenance stores must use separate files")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    saver, connection = sqlite_checkpointer(checkpoint)
    store = SQLiteProvenanceStore(provenance)
    try:
        yield RuntimeDependencies(
            hypothesis_agent=hypothesis_agent or MockHypothesisAgent(),
            planner_agent=planner_agent or MockPlannerAgent(),
            direct_search_backend=DirectSearchBackend(),
            evaluator=evaluator or MockEvaluator(),
            verifier=verifier or DeterministicVerifier(),
            provenance_store=store,
            checkpointer=saver,
            clock=clock,
            id_generator=id_generator,
        )
    finally:
        store.close()
        connection.close()
