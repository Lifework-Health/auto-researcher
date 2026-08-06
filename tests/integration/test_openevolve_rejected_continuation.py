from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from auto_researcher.agents.call_store import InMemoryAgentCallStore
from auto_researcher.agents.models import StructuredModelResponse
from auto_researcher.agents.mock import MockHypothesisAgent, MockPlannerAgent
from auto_researcher.contracts.enums import EventType, RunStatus, SearchType
from auto_researcher.graph.builder import build_graph
from auto_researcher.graph.nodes.openevolve import record_openevolve_candidate
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.execution import resume_run, start_run
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.live_models import (
    LiveMutationApproval,
    OpenEvolveModelCallContext,
)
from auto_researcher.search.openevolve.models import (
    CandidateStatus,
    OpenEvolveCandidateCollection,
)
from auto_researcher.search.openevolve.production_bridge import (
    DurableOpenEvolveModelBridge,
)
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
from auto_researcher.search.openevolve.upstream import (
    UpstreamOpenEvolveAdapter,
    default_adapter_contract,
)
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_contract,
    default_synthetic_openevolve_configuration,
)
from auto_researcher.tasks.synthetic.openevolve import SEED_SOURCE
from tests.unit.test_openevolve_production_bridge import (
    NOW,
    approval_payload,
    contract as bridge_contract,
)

FIXED_TIME = datetime(2026, 8, 6, 12, tzinfo=UTC)
FORBIDDEN_IMPORT_SOURCE = """import math

def evolve(configuration):
    return {"model_family": "tree", "complexity": 4, "learning_rate": 0.05}
"""
SECOND_FORBIDDEN_IMPORT_SOURCE = """import json

def evolve(configuration):
    return {"model_family": "tree", "complexity": 4, "learning_rate": 0.05}
"""
VALID_SOURCE = """def evolve(configuration):
    return {"model_family": "neural", "complexity": 4, "learning_rate": 0.05}
"""


def _response(source: str, description: str) -> dict:
    return {
        "protocol_version": "upstream-mutation-envelope-v1",
        "mutable_file": "candidate.py",
        "source": source,
        "description": description,
    }


class _SequentialProvider:
    provider = "fake-production"
    model_id = "fake-model-20260101"

    def __init__(self, responses: tuple[dict, ...]) -> None:
        self.responses = responses
        self.invocation_count = 0

    def generate_structured(self, **kwargs) -> StructuredModelResponse:
        response = self.responses[self.invocation_count]
        self.invocation_count += 1
        output = (
            kwargs["response_model"].model_validate(response).model_dump(mode="json")
        )
        return StructuredModelResponse(
            call_id=kwargs["call_id"],
            provider=self.provider,
            model_id=self.model_id,
            structured_output=output,
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.0002,
            latency_ms=1,
            attempts=1,
            finish_reason="fake_complete",
            provider_request_id=f"fake-request-{self.invocation_count}",
            prompt_version=kwargs["call_config"].prompt_version,
            context_hash=kwargs["context_hash"],
            response_hash=payload_hash(output),
        )


class _CountingService:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0
        for name in ("evaluator_id", "verifier_id", "version"):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))

    def evaluate(self, *args, **kwargs):
        self.calls += 1
        return self.inner.evaluate(*args, **kwargs)

    def verify(self, *args, **kwargs):
        self.calls += 1
        return self.inner.verify(*args, **kwargs)


class _RecordingRunner(LocalSandboxRunner):
    def __init__(self, workspace_root: Path) -> None:
        super().__init__(workspace_root)
        self.candidate_ids: list[str] = []
        self.candidate_generations: list[tuple[str, int]] = []

    def prepare(self, candidate, *args, **kwargs):
        self.candidate_ids.append(candidate.candidate_id)
        self.candidate_generations.append(
            (candidate.candidate_id, candidate.generation)
        )
        return (
            super()
            .prepare(candidate, *args, **kwargs)
            .model_copy(update={"runtime_seconds": 0.0})
        )


def _runtime(tmp_path, responses, *, interrupt_after=None):
    pytest.importorskip("openevolve", reason="pinned optional dependency absent")
    run_id = "reject-continuation"
    thread_id = "reject-continuation-thread"
    contract = default_synthetic_contract(
        maximum_cycles=1,
        search_types=frozenset({SearchType.OPENEVOLVE}),
        maximum_experiments=3,
    )
    configuration = default_synthetic_openevolve_configuration()
    configuration["openevolve"].update(
        {
            "population_size": 1,
            "maximum_generations": 2,
            "maximum_candidate_evaluations": 3,
            "maximum_model_calls": 2,
            "maximum_failed_candidates": 2,
            "maximum_consecutive_failures": 2,
            "objective_threshold": None,
        }
    )
    context = TaskRuntimeContext(
        run_id=run_id,
        output_dir=tmp_path / "artefacts",
        workspace_dir=tmp_path / "workspace",
        manifest_created_at=FIXED_TIME,
    )
    hypothesis_agent = MockHypothesisAgent()
    hypothesis = hypothesis_agent.generate(contract, cycle=1)
    planner_agent = MockPlannerAgent(
        search_type=SearchType.OPENEVOLVE,
        configuration=configuration,
        experiment_budget=3,
    )
    request = planner_agent.plan(contract, hypothesis, cycle=1)
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        context,
        contract,
        configuration,
        hypothesis_agent=hypothesis_agent,
        planner_agent=planner_agent,
        search_type=SearchType.OPENEVOLVE,
        clock=lambda: NOW,
        id_generator=lambda prefix: f"{prefix}-constant",
    )
    base = dependencies.openevolve_backend
    assert base is not None
    adapter_contract = default_adapter_contract(
        Path(__file__).parents[2] / "constraints/openevolve-0.3.2.lock"
    )
    adapter_hash = payload_hash(adapter_contract)
    approval = LiveMutationApproval.model_validate(
        approval_payload(
            run_id=run_id,
            contract_id=contract.contract_id,
            contract_hash=payload_hash(contract),
            task_version=contract.task_version,
            component_id=base.component_spec.component_id,
            component_version=base.component_spec.component_version,
            adapter_identity_hash=adapter_hash,
            maximum_model_calls=2,
            maximum_total_cost=0.04,
        )
    )
    provider = _SequentialProvider(tuple(responses))
    call_store = InMemoryAgentCallStore()
    bridge = DurableOpenEvolveModelBridge(
        contract=bridge_contract(),
        context=OpenEvolveModelCallContext(
            run_id=run_id,
            thread_id=thread_id,
            contract_id=contract.contract_id,
            contract_hash=payload_hash(contract),
            task_id="synthetic",
            task_version=contract.task_version,
            search_request_id=request.request_id,
            generation=1,
            parent_candidate_id="seed-placeholder",
            component_id=base.component_spec.component_id,
            component_version=base.component_spec.component_version,
            component_interface_hash=base.interface_hash,
            adapter_id=adapter_contract.adapter_id,
            adapter_version=adapter_contract.adapter_version,
            adapter_identity_hash=adapter_hash,
            executor_policy_hash="a" * 64,
            image_digest="sha256:" + "b" * 64,
            mutable_file="candidate.py",
            model_budget_identity="rejected-continuation-budget",
            maximum_model_calls=2,
            maximum_model_cost=0.04,
        ),
        approval=approval,
        store=call_store,
        provider_factory=lambda: provider,
        now=lambda: NOW,
        system_prompt="bounded prompt",
    )
    evaluator = _CountingService(dependencies.evaluator)
    verifier = _CountingService(dependencies.verifier)
    runner = _RecordingRunner(tmp_path / "sandbox")
    backend = OpenEvolveBackend(
        base.component,
        base.metadata,
        base.verifier_identity,
        UpstreamOpenEvolveAdapter(adapter_contract, bridge),
        runner,
    )
    dependencies = replace(
        dependencies,
        evaluator=evaluator,
        verifier=verifier,
        agent_call_store=call_store,
        openevolve_backend=backend,
    )
    graph = build_graph(dependencies, interrupt_after=interrupt_after)
    config = {"configurable": {"thread_id": thread_id}}
    return (
        graph,
        dependencies,
        provider,
        runner,
        evaluator,
        verifier,
        contract,
        request,
        run_id,
        thread_id,
        config,
    )


def _start(runtime):
    graph, _, _, _, _, _, contract, _, run_id, thread_id, config = runtime
    return start_run(
        graph,
        {"run_id": run_id, "thread_id": thread_id, "contract": contract},
        config,
    )


def test_rejected_first_mutation_continues_to_valid_second_mutation(tmp_path):
    runtime = _runtime(
        tmp_path,
        (
            _response(FORBIDDEN_IMPORT_SOURCE, "Invalid import."),
            _response(VALID_SOURCE, "Valid plain-Python mutation."),
        ),
    )
    final = _start(runtime)
    _, dependencies, provider, runner, evaluator, verifier, *_ = runtime
    population = final["openevolve_population_state"]
    candidates = sorted(
        final["openevolve_candidates"].candidates, key=lambda item: item.generation
    )
    seed, rejected, accepted = candidates

    assert final["status"] == RunStatus.COMPLETED
    assert provider.invocation_count == 2
    assert (
        len({item.call_id for item in dependencies.agent_call_store.list_records()})
        == 2
    )
    assert evaluator.calls == verifier.calls == 2
    assert population.budget.model_calls == 2
    assert population.budget.candidate_evaluations == 2
    assert population.budget.verifier_calls == 2
    assert population.budget.failed_candidates == 1
    assert population.budget.consecutive_failures == 0
    assert population.budget.generations_used == 2
    assert population.selected_parent_ids == (seed.candidate_id, seed.candidate_id)
    assert rejected.status == CandidateStatus.REJECTED
    assert rejected.validation_result.safe_error_code == "candidate_forbidden_import"
    assert rejected.preparation_result is None
    assert rejected.candidate_id not in runner.candidate_ids
    rejected_outcome = next(
        item
        for item in population.outcomes
        if item.candidate_id == rejected.candidate_id
    )
    assert rejected_outcome.evaluation is None
    assert rejected_outcome.verification is None
    assert rejected_outcome.rejection_reason == "candidate_forbidden_import"
    accepted_outcome = next(
        item
        for item in population.outcomes
        if item.candidate_id == accepted.candidate_id
    )
    assert accepted_outcome.status == CandidateStatus.VERIFIED
    assert accepted.model_call_id is not None
    assert (
        accepted_outcome.experiment.experiment_id
        == accepted.preparation_result.generated_experiment_id
    )
    assert (
        accepted_outcome.evaluation.experiment_id
        == accepted_outcome.experiment.experiment_id
    )
    assert (
        accepted_outcome.verification.experiment_id
        == accepted_outcome.experiment.experiment_id
    )
    assert accepted_outcome.objective_value == 0.88
    assert (
        final["openevolve_search_result"].stop_reason == "maximum_generations_reached"
    )

    events = dependencies.provenance_store.list_events(runtime[8])
    types = [event.event_type for event in events]
    assert types.count(EventType.OPENEVOLVE_CANDIDATE_REJECTED) == 1
    assert types.count(EventType.OPENEVOLVE_CANDIDATE_PROPOSED) == 2
    rejected_event = next(
        event
        for event in events
        if event.event_type == EventType.OPENEVOLVE_CANDIDATE_REJECTED
    )
    assert rejected.candidate_id in rejected_event.input_references
    openevolve_event_ids = [
        event.event_id
        for event in events
        if event.event_type.value.startswith("OPENEVOLVE_")
    ]
    assert len(openevolve_event_ids) == len(set(openevolve_event_ids))


def test_two_static_rejections_stop_normally_without_candidate_execution(tmp_path):
    runtime = _runtime(
        tmp_path,
        (
            _response(FORBIDDEN_IMPORT_SOURCE, "First invalid import."),
            _response(SECOND_FORBIDDEN_IMPORT_SOURCE, "Second invalid import."),
        ),
    )
    final = _start(runtime)
    _, _, provider, runner, evaluator, verifier, *_ = runtime
    population = final["openevolve_population_state"]
    candidates = sorted(
        final["openevolve_candidates"].candidates, key=lambda item: item.generation
    )
    rejected = candidates[1:]

    assert final["status"] == RunStatus.COMPLETED
    assert provider.invocation_count == 2
    assert evaluator.calls == verifier.calls == 1
    assert population.budget.model_calls == 2
    assert population.budget.candidate_evaluations == 1
    assert population.budget.verifier_calls == 1
    assert population.budget.failed_candidates == 2
    assert population.budget.consecutive_failures == 2
    assert population.budget.generations_used == 2
    assert all(item.status == CandidateStatus.REJECTED for item in rejected)
    assert all(item.candidate_id not in runner.candidate_ids for item in rejected)
    assert (
        final["openevolve_search_result"].stop_reason == "maximum_generations_reached"
    )


def test_duplicate_first_mutation_continues_without_duplicate_evaluation(tmp_path):
    runtime = _runtime(
        tmp_path,
        (
            _response(SEED_SOURCE, "Duplicate seed source."),
            _response(VALID_SOURCE, "Valid plain-Python mutation."),
        ),
    )
    final = _start(runtime)
    _, _, provider, runner, evaluator, verifier, *_ = runtime
    population = final["openevolve_population_state"]
    candidates = sorted(
        final["openevolve_candidates"].candidates, key=lambda item: item.generation
    )
    seed, accepted = candidates
    duplicate_lineage = next(
        item for item in population.lineage if item.generation == 1
    )

    assert final["status"] == RunStatus.COMPLETED
    assert provider.invocation_count == 2
    assert evaluator.calls == verifier.calls == 2
    assert population.budget.model_calls == 2
    assert population.budget.candidate_evaluations == 2
    assert population.budget.failed_candidates == 1
    assert population.selected_parent_ids == (seed.candidate_id, seed.candidate_id)
    assert duplicate_lineage.validation_code == "candidate_duplicate"
    assert duplicate_lineage.rejection_reason == "candidate_duplicate"
    assert (duplicate_lineage.candidate_id, 1) not in runner.candidate_generations
    assert accepted.status == CandidateStatus.VERIFIED


def test_rejected_candidate_wins_over_deliberately_stale_seed_results(tmp_path):
    runtime = _runtime(
        tmp_path,
        (
            _response(FORBIDDEN_IMPORT_SOURCE, "Invalid import."),
            _response(VALID_SOURCE, "Valid plain-Python mutation."),
        ),
    )
    final = _start(runtime)
    _, dependencies, _, _, _, _, _, request, *_ = runtime
    population = final["openevolve_population_state"]
    candidates = sorted(
        final["openevolve_candidates"].candidates, key=lambda item: item.generation
    )
    seed, rejected = candidates[:2]
    seed_outcome = next(
        item for item in population.outcomes if item.candidate_id == seed.candidate_id
    )
    backend = dependencies.openevolve_backend
    search_contract = final["openevolve_search_contract"]
    seed_population = backend.update_population(
        backend.initialise_population(search_contract),
        search_contract,
        seed,
        seed_outcome,
    )
    direct_state = {
        "run_id": "stale-result-defensive-test",
        "cycle": 1,
        "search_request": request,
        "openevolve_current_candidate": rejected,
        "openevolve_population_state": seed_population,
        "openevolve_search_contract": search_contract,
        "openevolve_candidates": OpenEvolveCandidateCollection(
            candidates=(seed, rejected)
        ),
        "experiment_spec": seed_outcome.experiment,
        "evaluation_result": seed_outcome.evaluation,
        "verification_result": seed_outcome.verification,
        "decision_event_ids": [],
    }

    update = record_openevolve_candidate(direct_state, dependencies)
    outcome = update["openevolve_population_state"].outcomes[-1]
    assert outcome.candidate_id == rejected.candidate_id
    assert outcome.status == CandidateStatus.REJECTED
    assert outcome.rejection_reason == "candidate_forbidden_import"
    assert outcome.experiment is None
    assert outcome.evaluation is None
    assert outcome.verification is None


def test_non_rejected_candidate_with_mismatched_result_identity_fails_closed(tmp_path):
    runtime = _runtime(
        tmp_path,
        (
            _response(FORBIDDEN_IMPORT_SOURCE, "Invalid import."),
            _response(VALID_SOURCE, "Valid plain-Python mutation."),
        ),
    )
    final = _start(runtime)
    _, dependencies, _, _, _, _, _, request, *_ = runtime
    candidates = sorted(
        final["openevolve_candidates"].candidates, key=lambda item: item.generation
    )
    seed, _, accepted = candidates
    population = final["openevolve_population_state"]
    seed_outcome = next(
        item for item in population.outcomes if item.candidate_id == seed.candidate_id
    )
    backend = dependencies.openevolve_backend
    search_contract = final["openevolve_search_contract"]
    seed_population = backend.update_population(
        backend.initialise_population(search_contract),
        search_contract,
        seed,
        seed_outcome,
    )
    contradictory = {
        "run_id": "contradictory-result-test",
        "cycle": 1,
        "search_request": request,
        "openevolve_current_candidate": accepted,
        "openevolve_population_state": seed_population,
        "openevolve_search_contract": search_contract,
        "openevolve_candidates": OpenEvolveCandidateCollection(
            candidates=(seed, accepted)
        ),
        "experiment_spec": seed_outcome.experiment,
        "evaluation_result": seed_outcome.evaluation,
        "verification_result": seed_outcome.verification,
        "decision_event_ids": [],
    }

    with pytest.raises(
        ValueError, match="^openevolve_candidate_result_state_conflict$"
    ):
        record_openevolve_candidate(contradictory, dependencies)


def test_new_proposal_clears_seed_scoped_generic_result_state(tmp_path):
    runtime = _runtime(
        tmp_path,
        (
            _response(FORBIDDEN_IMPORT_SOURCE, "Invalid import."),
            _response(VALID_SOURCE, "Unused second mutation."),
        ),
        interrupt_after=["propose_openevolve_candidate"],
    )
    state = _start(runtime)
    _, _, provider, *_ = runtime

    assert provider.invocation_count == 1
    assert state["openevolve_current_candidate"].generation == 1
    assert state["experiment_spec"] is None
    assert state["evaluation_result"] is None
    assert state["verification_result"] is None


def test_resume_after_recorded_rejection_does_not_repeat_first_mutation(tmp_path):
    runtime = _runtime(
        tmp_path,
        (
            _response(FORBIDDEN_IMPORT_SOURCE, "Invalid import."),
            _response(VALID_SOURCE, "Valid plain-Python mutation."),
        ),
        interrupt_after=["record_openevolve_candidate"],
    )
    state = _start(runtime)
    graph, dependencies, provider, _, _, _, _, _, run_id, _, config = runtime
    assert state["openevolve_population_state"].budget.model_calls == 0

    state = resume_run(graph, config)
    assert provider.invocation_count == 1
    assert state["openevolve_population_state"].budget.failed_candidates == 1
    rejected_id = state["openevolve_current_candidate"].candidate_id
    rejected_events = [
        event
        for event in dependencies.provenance_store.list_events(run_id)
        if event.event_type == EventType.OPENEVOLVE_CANDIDATE_REJECTED
    ]
    assert len(rejected_events) == 1

    state = resume_run(graph, config)
    assert provider.invocation_count == 2
    assert state["openevolve_population_state"].budget.failed_candidates == 1
    assert state["openevolve_population_state"].failed_candidate_ids == (rejected_id,)
    final = resume_run(graph, config)
    assert final["status"] == RunStatus.COMPLETED
    assert provider.invocation_count == 2
    openevolve_event_ids = [
        event.event_id
        for event in dependencies.provenance_store.list_events(run_id)
        if event.event_type.value.startswith("OPENEVOLVE_")
    ]
    assert len(openevolve_event_ids) == len(set(openevolve_event_ids))
