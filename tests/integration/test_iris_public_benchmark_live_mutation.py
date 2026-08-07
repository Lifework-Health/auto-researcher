from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from auto_researcher.agents.call_store import InMemoryAgentCallStore
from auto_researcher.agents.mock import MockHypothesisAgent, MockPlannerAgent
from auto_researcher.agents.models import StructuredModelResponse
from auto_researcher.contracts.enums import EventType, RunStatus, SearchType
from auto_researcher.contracts.models import SearchRequest
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.execution import start_run
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.hardened_executor import (
    HardenedDockerExecutor,
    docker_policy,
)
from auto_researcher.search.openevolve.live_models import (
    LiveMutationApproval,
    OpenEvolveModelCallContext,
)
from auto_researcher.search.openevolve.models import CandidateStatus
from auto_researcher.search.openevolve.production_bridge import (
    DurableOpenEvolveModelBridge,
)
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
from auto_researcher.search.openevolve.upstream import (
    UpstreamOpenEvolveAdapter,
    build_approved_live_upstream_runtime,
    default_adapter_contract,
)
from auto_researcher.tasks.iris_knn import (
    IrisKNNTask,
    default_iris_contract,
    default_iris_openevolve_configuration,
)
from auto_researcher.tasks.iris_knn.manifests import DATA_SHA256, FOLD_SHA256
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.verification.verifier import DeterministicVerifier
from tests.unit.test_openevolve_production_bridge import (
    NOW,
    approval_payload,
    contract as bridge_contract,
)

FIXED_TIME = datetime(2026, 8, 7, 12, tzinfo=UTC)
VALID_SOURCE = """def evolve(configuration):
    return {"feature_weights": [0.2, 0.2, 4.0, 4.0], "k": 5, "distance_power": 2}
"""
INVALID_SOURCE = """import os

def evolve(configuration):
    return {"feature_weights": [1.0, 1.0, 1.0, 1.0], "k": 3, "distance_power": 2}
"""


def _response(source: str, description: str) -> dict:
    return {
        "protocol_version": "upstream-mutation-envelope-v1",
        "mutable_file": "candidate.py",
        "source": source,
        "description": description,
    }


class RecordingSequentialProvider:
    provider = "fake-production"
    model_id = "fake-model-20260101"

    def __init__(self, responses: tuple[dict, ...]) -> None:
        self.responses = responses
        self.invocation_count = 0
        self.user_prompts: list[str] = []
        self.system_prompts: list[str] = []

    def generate_structured(self, **kwargs) -> StructuredModelResponse:
        self.user_prompts.append(kwargs["user_prompt"])
        self.system_prompts.append(kwargs["system_prompt"])
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


class CountingService:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0
        for name in ("evaluator_id", "verifier_id", "version", "cost_per_experiment"):
            if hasattr(inner, name):
                setattr(self, name, getattr(inner, name))

    def evaluate(self, *args, **kwargs):
        self.calls += 1
        return self.inner.evaluate(*args, **kwargs)

    def verify(self, *args, **kwargs):
        self.calls += 1
        return self.inner.verify(*args, **kwargs)


def _runtime(tmp_path, responses: tuple[dict, ...]):
    pytest.importorskip("openevolve", reason="pinned optional dependency absent")
    task = IrisKNNTask()
    run_id = "iris-public-live-fixture"
    thread_id = "iris-public-live-fixture-thread"
    maximum_experiments = len(responses) + 1
    research_contract = default_iris_contract(
        maximum_cycles=1,
        search_types=frozenset({SearchType.OPENEVOLVE}),
        maximum_experiments=maximum_experiments,
    )
    configuration = default_iris_openevolve_configuration()
    configuration["openevolve"].update(
        {
            "maximum_generations": len(responses),
            "maximum_candidate_evaluations": maximum_experiments,
            "maximum_model_calls": len(responses),
            "maximum_failed_candidates": len(responses),
            "maximum_consecutive_failures": len(responses),
            "objective_threshold": None,
        }
    )
    runtime_context = TaskRuntimeContext(
        run_id=run_id,
        output_dir=tmp_path / "artefacts",
        workspace_dir=tmp_path / "workspace",
        manifest_created_at=FIXED_TIME,
    )
    hypothesis_agent = MockHypothesisAgent()
    hypothesis = hypothesis_agent.generate(research_contract, cycle=1)
    planner_agent = MockPlannerAgent(
        search_type=SearchType.OPENEVOLVE,
        configuration=configuration,
        experiment_budget=maximum_experiments,
    )
    request = planner_agent.plan(research_contract, hypothesis, cycle=1)
    dependencies = task_memory_dependencies(
        task,
        runtime_context,
        research_contract,
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
            contract_id=research_contract.contract_id,
            contract_hash=payload_hash(research_contract),
            task_id=task.task_id,
            task_version=task.task_version,
            component_id=base.component_spec.component_id,
            component_version=base.component_spec.component_version,
            adapter_identity_hash=adapter_hash,
            permitted_dataset_class="public_benchmark",
            maximum_model_calls=len(responses),
            maximum_total_cost=0.02 * len(responses),
        )
    )
    provider = RecordingSequentialProvider(responses)
    call_store = InMemoryAgentCallStore()
    model_context = OpenEvolveModelCallContext(
        run_id=run_id,
        thread_id=thread_id,
        contract_id=research_contract.contract_id,
        contract_hash=payload_hash(research_contract),
        task_id=task.task_id,
        task_version=task.task_version,
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
        dataset_class="public_benchmark",
        model_budget_identity="iris-public-model-budget",
        maximum_model_calls=len(responses),
        maximum_model_cost=0.02 * len(responses),
    )
    prompt = (
        Path(__file__).parents[2]
        / "src/auto_researcher/prompts/openevolve/openevolve-mutation-prompt-v2.md"
    ).read_text(encoding="utf-8")
    bridge = DurableOpenEvolveModelBridge(
        contract=bridge_contract(),
        context=model_context,
        approval=approval,
        store=call_store,
        provider_factory=lambda: provider,
        now=lambda: NOW,
        system_prompt=prompt,
    )
    evaluator = CountingService(dependencies.evaluator)
    verifier = CountingService(dependencies.verifier)
    backend = OpenEvolveBackend(
        base.component,
        base.metadata,
        base.verifier_identity,
        UpstreamOpenEvolveAdapter(adapter_contract, bridge),
        LocalSandboxRunner(tmp_path / "sandbox"),
    )
    dependencies = replace(
        dependencies,
        evaluator=evaluator,
        verifier=verifier,
        agent_call_store=call_store,
        openevolve_backend=backend,
    )
    final = start_run(
        build_graph(dependencies),
        {"run_id": run_id, "thread_id": thread_id, "contract": research_contract},
        {"configurable": {"thread_id": thread_id}},
    )
    return final, dependencies, provider, evaluator, verifier, model_context


def test_iris_public_benchmark_fake_production_lifecycle_and_raw_data_exclusion(
    tmp_path,
):
    final, dependencies, provider, evaluator, verifier, model_context = _runtime(
        tmp_path,
        (_response(VALID_SOURCE, "Bounded public-benchmark configuration mutation."),),
    )
    population = final["openevolve_population_state"]
    assert final["status"] == RunStatus.COMPLETED
    assert model_context.dataset_class == "public_benchmark"
    assert provider.invocation_count == 1
    assert evaluator.calls == verifier.calls == 2
    assert [item.objective_value for item in population.outcomes] == [
        0.94,
        0.953333333333,
    ]
    assert all(item.verified for item in population.outcomes)
    assert population.budget.model_calls == 1
    assert final["verification_result"].verified is True

    model_input = "\n".join((*provider.system_prompts, *provider.user_prompts))
    assert "feature_weights" in model_input
    assert {
        item.prompt_version for item in dependencies.agent_call_store.list_records()
    } == {"openevolve-mutation-prompt-v2"}
    for forbidden in (
        "5.1,3.5,1.4,0.2,Iris-setosa",
        DATA_SHA256,
        FOLD_SHA256,
        '"assignments"',
        "aggregate_confusion_counts",
        "row_predictions",
    ):
        assert forbidden not in model_input
    persisted = json.dumps(
        [
            item.model_dump(mode="json")
            for item in dependencies.agent_call_store.list_records()
        ],
        sort_keys=True,
    )
    assert "5.1,3.5,1.4,0.2,Iris-setosa" not in persisted


def test_iris_public_benchmark_rejected_candidate_continues_safely(tmp_path):
    final, dependencies, provider, evaluator, verifier, _ = _runtime(
        tmp_path,
        (
            _response(INVALID_SOURCE, "Forbidden import mutation."),
            _response(VALID_SOURCE, "Valid bounded continuation."),
        ),
    )
    population = final["openevolve_population_state"]
    candidates = sorted(
        final["openevolve_candidates"].candidates, key=lambda item: item.generation
    )
    rejected = candidates[1]
    accepted = candidates[2]
    assert final["status"] == RunStatus.COMPLETED
    assert provider.invocation_count == 2
    assert evaluator.calls == verifier.calls == 2
    assert rejected.status == CandidateStatus.REJECTED
    assert rejected.validation_result.safe_error_code == "candidate_forbidden_import"
    assert rejected.preparation_result is None
    rejected_outcome = next(
        item
        for item in population.outcomes
        if item.candidate_id == rejected.candidate_id
    )
    assert rejected_outcome.evaluation is rejected_outcome.verification is None
    accepted_outcome = next(
        item
        for item in population.outcomes
        if item.candidate_id == accepted.candidate_id
    )
    assert accepted_outcome.status == CandidateStatus.VERIFIED
    assert accepted_outcome.objective_value == 0.953333333333
    assert population.budget.failed_candidates == 1
    assert population.budget.candidate_evaluations == 2
    events = dependencies.provenance_store.list_events("iris-public-live-fixture")
    assert [event.event_type for event in events].count(
        EventType.OPENEVOLVE_CANDIDATE_REJECTED
    ) == 1


@pytest.mark.hardened_executor
def test_public_benchmark_fake_production_candidate_uses_retained_image(tmp_path):
    image = os.getenv("AUTO_RESEARCHER_HARDENED_IMAGE")
    digest = os.getenv("AUTO_RESEARCHER_HARDENED_IMAGE_DIGEST")
    if not image or not digest:
        pytest.skip("retained hardened image and digest were not explicitly selected")
    task = IrisKNNTask()
    contract = default_iris_contract(
        search_types=frozenset({SearchType.OPENEVOLVE}), maximum_experiments=2
    )
    runtime_context = TaskRuntimeContext(
        run_id="iris-public-hardened",
        output_dir=tmp_path / "artefacts",
        workspace_dir=tmp_path / "workspace",
        manifest_created_at=FIXED_TIME,
    )
    component = task.create_evolvable_component(contract, runtime_context)
    metadata = task.experiment_metadata(runtime_context)
    verifier_identity = "deterministic-verifier-v1@iris-knn-evidence-policy-v1"
    preliminary = OpenEvolveBackend(
        component,
        metadata,
        verifier_identity,
        object(),  # type: ignore[arg-type]
        LocalSandboxRunner(tmp_path / "preliminary"),
    )
    adapter_contract = default_adapter_contract(
        Path(__file__).parents[2] / "constraints/openevolve-0.3.2.lock"
    )
    docker_version = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    policy = docker_policy(
        image,
        digest,
        Path(__file__).parents[2] / "docker/openevolve-executor/Dockerfile",
        Path(__file__).parents[2] / "docker/openevolve-executor/worker.py",
        docker_version,
    )
    request_id = "iris-public-hardened-request"
    approval = LiveMutationApproval.model_validate(
        approval_payload(
            run_id=runtime_context.run_id,
            contract_id=contract.contract_id,
            contract_hash=payload_hash(contract),
            task_id=task.task_id,
            task_version=task.task_version,
            component_id=preliminary.component_spec.component_id,
            component_version=preliminary.component_spec.component_version,
            adapter_identity_hash=payload_hash(adapter_contract),
            executor_policy_hash=payload_hash(policy),
            image_digest=policy.image_digest,
            permitted_dataset_class="public_benchmark",
        )
    )
    provider = RecordingSequentialProvider(
        (_response(VALID_SOURCE, "Retained-image public benchmark mutation."),)
    )
    bridge = DurableOpenEvolveModelBridge(
        contract=bridge_contract(),
        context=OpenEvolveModelCallContext(
            run_id=runtime_context.run_id,
            thread_id="iris-public-hardened-thread",
            contract_id=contract.contract_id,
            contract_hash=payload_hash(contract),
            task_id=task.task_id,
            task_version=task.task_version,
            search_request_id=request_id,
            generation=1,
            parent_candidate_id="seed-placeholder",
            component_id=preliminary.component_spec.component_id,
            component_version=preliminary.component_spec.component_version,
            component_interface_hash=preliminary.interface_hash,
            adapter_id=adapter_contract.adapter_id,
            adapter_version=adapter_contract.adapter_version,
            adapter_identity_hash=payload_hash(adapter_contract),
            executor_policy_hash=payload_hash(policy),
            image_digest=policy.image_digest,
            mutable_file="candidate.py",
            dataset_class="public_benchmark",
            model_budget_identity="iris-public-hardened-budget",
            maximum_model_calls=1,
            maximum_model_cost=0.02,
        ),
        approval=approval,
        store=InMemoryAgentCallStore(),
        provider_factory=lambda: provider,
        now=lambda: NOW,
        system_prompt="bounded prompt-v2 fixture",
    )
    evidence_executor = HardenedDockerExecutor(policy, tmp_path / "isolation-evidence")
    isolation = evidence_executor.verify_isolation()
    adapter, executor = build_approved_live_upstream_runtime(
        adapter_contract,
        bridge,
        policy,
        isolation,
        task=task,
        workspace_root=tmp_path / "candidate-workspace",
    )
    backend = OpenEvolveBackend(
        component, metadata, verifier_identity, adapter, executor
    )
    configuration = default_iris_openevolve_configuration()
    configuration["openevolve"].update(
        {
            "maximum_generations": 1,
            "maximum_model_calls": 1,
            "sandbox_policy_id": "openevolve-hardened-executor-v2",
        }
    )
    request = SearchRequest(
        request_id=request_id,
        hypothesis_id="iris-public-hardened-hypothesis",
        search_type=SearchType.OPENEVOLVE,
        target="bounded Iris configuration mutation",
        search_space=configuration,
        experiment_budget=2,
        rationale="retained-image offline public benchmark smoke",
    )
    search = backend.create_search_contract(request, contract)
    seed = backend.seed_candidate(search)
    population = backend.initialise_population(search)
    reservation = backend.reserve_mutation(search, population, seed)
    candidate = backend.mutate_candidate(reservation, seed, search)
    validation = backend.validate(candidate)
    candidate = candidate.model_copy(update={"validation_result": validation})
    preparation = backend.prepare(candidate, search)
    experiment = component.candidate_to_experiment(
        candidate,
        preparation,
        request,
        contract,
        metadata,
        run_id=runtime_context.run_id,
    )
    evaluation = task.create_evaluator(runtime_context).evaluate(experiment, contract)
    verification = DeterministicVerifier(
        task.create_verification_policy(contract)
    ).verify(experiment, evaluation, contract)
    assert provider.invocation_count == 1
    assert preparation.execution_status.value == "COMPLETED"
    assert evaluation.primary_score == 0.953333333333
    assert verification.verified is True
