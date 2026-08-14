from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from auto_researcher.resources import (
    CourtesyResourceAdmissionPolicy,
    InMemoryResourceLeaseStore,
    ResourceBroker,
    ResourceCandidate,
    ResourceCapacity,
    ResourceRequest,
    ResourceRequirement,
)
from auto_researcher.search.openevolve.native_engine import (
    ApprovedModel,
    ApprovedModelBridgeLLM,
    AutoResearcherEvaluatorAdapter,
    EmbeddedEvaluationRequest,
    EmbeddedOpenEvolveSearch,
    NativeEvolutionConfiguration,
    NativeEvolutionLimits,
    NativeEvolutionResult,
    SafeEmbeddingAdapter,
    ScientificEvaluation,
    TaskOwnedCandidateNormalizer,
    TaskOwnedScientificEvaluator,
    native_configuration_from_search_space,
    native_limits_from_search_space,
    scientific_candidate_identity,
)
from auto_researcher.search.openevolve.live_boundary import (
    metadata_only_model_exposure_identity,
    validate_metadata_only_request,
)
from auto_researcher.search.openevolve.models import EvolvableComponentSpec

pytestmark = pytest.mark.upstream_openevolve


class ScriptedModel:
    model = "offline-scripted-model"

    def __init__(self, sources: Sequence[str]) -> None:
        self.sources = tuple(sources)
        self.calls = 0
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.generate_with_context(
            "", [{"role": "user", "content": prompt}], **kwargs
        )

    async def generate_with_context(
        self,
        system_message: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        del kwargs
        rendered = (
            system_message + "\n" + "\n".join(item["content"] for item in messages)
        )
        with self._lock:
            index = self.calls
            self.calls += 1
            self.prompts.append(rendered)
        if index >= len(self.sources):
            raise AssertionError("unexpected offline model call")
        return f"```python\n{self.sources[index]}\n```"


class SimulatedGPUProvider:
    def __init__(self, count: int) -> None:
        self.snapshot = tuple(
            ResourceCandidate(
                resource_id=f"gpu:{index}",
                resource_type="gpu",
                quantity=1,
                capacities=(ResourceCapacity(name="memory_mib", value=48_000),),
                utilization_percent=0,
                equivalence_tags=frozenset({"equivalent-a4-gpu"}),
            )
            for index in range(count)
        )

    def candidates(self, _request: ResourceRequest) -> tuple[ResourceCandidate, ...]:
        return self.snapshot


class RecordingEvaluator:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.requests: list[EmbeddedEvaluationRequest] = []
        self.concurrent_resources: set[str] = set()
        self.maximum_parallel = 0
        self._active_resources: set[str] = set()
        self._lock = threading.Lock()

    def __call__(self, request: EmbeddedEvaluationRequest) -> ScientificEvaluation:
        resource_id = (
            request.resource_admission.lease.resource_id
            if request.resource_admission is not None
            and request.resource_admission.lease is not None
            else None
        )
        with self._lock:
            self.requests.append(request)
            if resource_id is not None:
                assert resource_id not in self._active_resources
                self._active_resources.add(resource_id)
                self.concurrent_resources.add(resource_id)
            self.maximum_parallel = max(
                self.maximum_parallel, len(self._active_resources)
            )
        try:
            if self.delay:
                time.sleep(self.delay)
            policy = int(request.identity.canonical_configuration["policy"])
            return ScientificEvaluation(
                primary_score=0.5 + policy / 10,
                secondary_metrics={
                    "policy_complexity": float(policy),
                    "robustness": 1.0 - policy / 100,
                },
                verified=True,
                constraint_compliant=True,
                safe_artifact_summaries=(f"policy {policy} verified",),
            )
        finally:
            with self._lock:
                if resource_id is not None:
                    self._active_resources.remove(resource_id)


def _normalise(source: str) -> dict[str, int]:
    match = re.search(r"POLICY\s*=\s*(\d+)", source)
    if match is None:
        raise ValueError("offline_policy_missing")
    return {"policy": int(match.group(1))}


def _configuration(
    search_identity: str,
    *,
    maximum_iterations: int = 4,
    parallel_evaluations: int = 3,
) -> NativeEvolutionConfiguration:
    return NativeEvolutionConfiguration(
        search_identity=search_identity,
        population_size=12,
        archive_size=12,
        num_islands=3,
        migration_interval=1,
        migration_rate=0.5,
        feature_dimensions=("policy_complexity", "primary_score"),
        feature_bins=6,
        parallel_evaluations=parallel_evaluations,
        checkpoint_interval=1,
        random_seed=20260814,
        diff_based_evolution=False,
        use_template_stochasticity=True,
        num_top_programs=3,
        num_diverse_programs=2,
        native_max_iterations=maximum_iterations,
    )


def _limits(maximum_iterations: int = 4) -> NativeEvolutionLimits:
    return NativeEvolutionLimits(
        maximum_iterations=maximum_iterations,
        maximum_model_calls=maximum_iterations,
        maximum_candidate_evaluations=20,
        maximum_wall_time_seconds=30,
    )


def _gpu_broker(count: int) -> ResourceBroker:
    return ResourceBroker(
        SimulatedGPUProvider(count),
        CourtesyResourceAdmissionPolicy(maximum_utilization_percent=80),
        lease_store=InMemoryResourceLeaseStore(),
        poll_seconds=0.005,
    )


def _gpu_request(identity: Any) -> ResourceRequest:
    return ResourceRequest(
        request_id=f"openevolve-{identity.evaluation_identity}",
        requirements=(
            ResourceRequirement(
                resource_type="gpu",
                minimum_capacities=(ResourceCapacity(name="memory_mib", value=40_000),),
            ),
        ),
        maximum_wait_seconds=5,
        equivalence_requirements=frozenset({"equivalent-a4-gpu"}),
    )


def _adapter(
    evaluator: RecordingEvaluator,
    *,
    broker: ResourceBroker | None = None,
) -> AutoResearcherEvaluatorAdapter:
    return AutoResearcherEvaluatorAdapter(
        normalizer=_normalise,
        evaluator=evaluator,
        component_identity="offline-training-policy-v1",
        evaluator_identity="offline-scientific-evaluator-v1",
        dataset_version="synthetic-no-data-v1",
        code_version="test-openevolve-full-strength-v1",
        maximum_evaluations=20,
        resource_broker=broker,
        resource_request_factory=_gpu_request if broker is not None else None,
        worker_id="offline-openevolve-worker",
    )


@dataclass(frozen=True)
class A3Run:
    result: NativeEvolutionResult
    model: ScriptedModel
    evaluator: RecordingEvaluator
    output_dir: Path


@pytest.fixture(scope="module")
def a3_run(tmp_path_factory: pytest.TempPathFactory) -> A3Run:
    pytest.importorskip("openevolve")
    output_dir = tmp_path_factory.mktemp("openevolve-a3")
    model = ScriptedModel(
        (
            "POLICY = 1  # first scientific configuration, source a",
            "POLICY=1  # first scientific configuration, source b",
            "POLICY = 2  # second scientific configuration, source a",
            "POLICY=2  # second scientific configuration, source b",
        )
    )
    evaluator = RecordingEvaluator(delay=0.05)
    runtime = EmbeddedOpenEvolveSearch(
        output_dir=output_dir,
        initial_source="POLICY = 0  # seed\n",
        configuration=_configuration("offline-a3-regression-v1"),
        limits=_limits(),
        models=(ApprovedModel(name=model.model, weight=1.0, adapter=model),),
        evaluator=_adapter(evaluator),
    )
    return A3Run(
        result=asyncio.run(runtime.run()),
        model=model,
        evaluator=evaluator,
        output_dir=output_dir,
    )


def test_a3_semantic_dedup_and_feedback_regression(a3_run: A3Run) -> None:
    children = tuple(
        feedback for feedback in a3_run.result.feedback if feedback.generation > 0
    )
    executed = tuple(
        feedback for feedback in children if feedback.evaluation_status == "EXECUTED"
    )
    reused = tuple(
        feedback for feedback in children if feedback.evaluation_status == "REUSED"
    )

    assert len({item.source_candidate_id for item in children}) == 4
    assert len({item.canonical_candidate_id for item in children}) == 2
    assert len(executed) == 2
    assert len(reused) == 2
    assert len(children) == 4
    assert all(item.parent_source_candidate_id for item in children)
    assert {item.evaluation_identity for item in children} == {
        item.evaluation_identity for item in executed
    }
    assert {item.evaluation_artifact_identity for item in children} == {
        item.evaluation_artifact_identity for item in executed
    }
    assert a3_run.result.expensive_evaluations == 3  # seed + two children
    assert a3_run.result.reused_evaluations == 2
    assert len(a3_run.evaluator.requests) == 3


def test_later_generation_prompt_contains_safe_evaluation_feedback(
    a3_run: A3Run,
) -> None:
    later_prompts = tuple(
        prompt
        for prompt in a3_run.result.prompts
        if "auto_researcher_safe_feedback.json" in prompt
    )
    assert later_prompts
    assert any("evaluation_status" in prompt for prompt in later_prompts)
    rendered = "\n".join(a3_run.result.prompts).casefold()
    for protected in (
        "api_key",
        "patient",
        "secret",
        "subject_id",
        "/protected/",
    ):
        assert protected not in rendered


def test_native_controller_owns_population_archive_and_selection(
    a3_run: A3Run,
) -> None:
    assert len(a3_run.result.programme_ids) > 1
    assert len(a3_run.result.archive_ids) > 1
    assert len(a3_run.result.island_programme_ids) == 3
    assert len(a3_run.result.prompts) == 4
    assert a3_run.model.calls == 4
    assert a3_run.result.best_source_candidate_id in a3_run.result.programme_ids


def test_map_elites_and_native_feature_dimensions_are_active(a3_run: A3Run) -> None:
    assert a3_run.result.feature_cells > 1
    assert all(
        "policy_complexity" in feedback.secondary_metrics
        for feedback in a3_run.result.feedback
    )


def test_native_islands_migrate_without_changing_scientific_identity(
    a3_run: A3Run,
) -> None:
    occupied = tuple(island for island in a3_run.result.island_programme_ids if island)
    assert len(occupied) == 3
    assert any(len(island) > 1 for island in occupied)
    for feedback in a3_run.result.feedback:
        assert feedback.canonical_candidate_id.startswith("scientific-")


def test_evaluator_adapter_returns_verified_safe_metrics() -> None:
    evaluator = RecordingEvaluator()
    adapter = _adapter(evaluator)
    adapter.bind_candidate(
        "source-direct", parent_source_candidate_id=None, generation=0
    )

    metrics = asyncio.run(adapter.evaluate_program("POLICY = 3", "source-direct"))
    artifact = adapter.get_pending_artifacts("source-direct")

    assert metrics["verified"] == 1.0
    assert metrics["constraint_compliant"] == 1.0
    assert metrics["primary_score"] == pytest.approx(0.8)
    assert artifact is not None
    assert "evaluation_identity" in artifact["auto_researcher_safe_feedback.json"]


def test_task_owned_normalizer_uses_hardened_component_projection(
    tmp_path: Path,
) -> None:
    from auto_researcher.tasks.feta_seg_evolve import default_feta_evolve_contract
    from auto_researcher.tasks.feta_seg_evolve.openevolve import COSINE_SOURCE
    from tests.unit.test_feta_seg_evolve_task import _backend, _request

    backend = _backend(workspace=tmp_path)
    search_contract = backend.create_search_contract(
        _request(), default_feta_evolve_contract()
    )
    normalizer = TaskOwnedCandidateNormalizer(backend, search_contract)

    first = normalizer(COSINE_SOURCE)
    second = normalizer(COSINE_SOURCE.replace("    return", "        return"))
    candidate, preparation = normalizer.prepared_candidate(COSINE_SOURCE)

    assert first == second
    assert first["policy_version"] == "feta-training-policy-v1"
    assert candidate.validation_result is not None
    assert preparation.execution_status.value == "COMPLETED"


def test_task_owned_scientific_evaluator_invokes_verifier_and_evidence_sink() -> None:
    from auto_researcher.contracts.enums import SearchType
    from auto_researcher.runtime.dependencies import memory_dependencies
    from tests.unit.test_openevolve_contracts import _contract, _request

    dependencies = memory_dependencies(search_type=SearchType.OPENEVOLVE)
    assert dependencies.openevolve_backend is not None
    backend = dependencies.openevolve_backend
    search_request = _request()
    research_contract = _contract()
    search_contract = backend.create_search_contract(search_request, research_contract)
    normalizer = TaskOwnedCandidateNormalizer(backend, search_contract)
    source = backend.component_spec.seed_source
    canonical = normalizer(source)
    identity = scientific_candidate_identity(
        source_candidate_id="native-source-seed",
        source=source,
        canonical_configuration=canonical,
        component_identity=backend.interface_hash,
        evaluator_identity=backend.evaluator_identity,
        dataset_version=dependencies.experiment_metadata.dataset_version,
        code_version=dependencies.experiment_metadata.code_version,
    )
    evidence: list[tuple[Any, ...]] = []

    def coordinate(experiment: Any, embedded_request: Any):
        evaluation = dependencies.evaluator.evaluate(experiment, research_contract)
        verification = dependencies.verifier.verify(
            experiment,
            evaluation,
            research_contract,
            claimed_score=evaluation.primary_score,
        )
        evidence.append((experiment, evaluation, verification, embedded_request))
        return evaluation, verification

    evaluator = TaskOwnedScientificEvaluator(
        normalizer=normalizer,
        search_request=search_request,
        research_contract=research_contract,
        metadata=dependencies.experiment_metadata,
        run_id="native-scientific-boundary-test",
        coordinator=coordinate,
    )

    result = evaluator(
        EmbeddedEvaluationRequest(
            source_candidate_id="native-source-seed",
            parent_source_candidate_id=None,
            generation=0,
            source=source,
            identity=identity,
        )
    )

    assert result.evaluation_artifact_identity is not None
    assert result.verified
    assert result.safe_failure_classification is None
    assert len(evidence) == 1
    experiment, evaluation, verification, embedded_request = evidence[0]
    assert experiment.experiment_id == evaluation.experiment_id
    assert verification.experiment_id == evaluation.experiment_id
    assert embedded_request.identity == identity


def test_feta_a4_template_maps_to_executable_native_configuration() -> None:
    from auto_researcher.tasks.feta_seg_evolve import (
        default_feta_evolve_a4_openevolve_configuration,
    )

    search_space = default_feta_evolve_a4_openevolve_configuration()
    configuration = native_configuration_from_search_space(
        "feta-a4-acceptance-search", search_space
    )
    limits = native_limits_from_search_space(search_space)

    assert configuration.population_size == 12
    assert configuration.num_islands == 3
    assert configuration.parallel_evaluations == 3
    assert configuration.feature_dimensions == (
        "primary_score",
        "policy_complexity",
    )
    assert limits.maximum_iterations == 4
    assert limits.maximum_model_calls == 24
    assert limits.maximum_candidate_evaluations == 24


def test_parallel_evaluations_use_three_simulated_gpus(tmp_path: Path) -> None:
    pytest.importorskip("openevolve")
    model = ScriptedModel(("POLICY = 1", "POLICY = 2", "POLICY = 3"))
    evaluator = RecordingEvaluator(delay=0.12)
    runtime = EmbeddedOpenEvolveSearch(
        output_dir=tmp_path,
        initial_source="POLICY = 0\n",
        configuration=_configuration(
            "offline-three-gpu-placement-v1", maximum_iterations=3
        ),
        limits=_limits(3),
        models=(ApprovedModel(name=model.model, weight=1.0, adapter=model),),
        evaluator=_adapter(evaluator, broker=_gpu_broker(3)),
    )

    result = asyncio.run(runtime.run())
    child_resources = {
        feedback.resource_id for feedback in result.feedback if feedback.generation > 0
    }

    assert child_resources == {"gpu:0", "gpu:1", "gpu:2"}
    assert evaluator.maximum_parallel == 3
    assert result.expensive_evaluations == 4
    assert len({identity for identity, _resource in runtime.evaluator.placements}) == 4


def test_resource_lease_heartbeats_cover_scientific_evaluation() -> None:
    class RecordingBroker(ResourceBroker):
        renewals = 0

        def renew_lease(self, *args: Any, **kwargs: Any):
            self.renewals += 1
            return super().renew_lease(*args, **kwargs)

    broker = RecordingBroker(
        SimulatedGPUProvider(1),
        CourtesyResourceAdmissionPolicy(maximum_utilization_percent=80),
        lease_store=InMemoryResourceLeaseStore(),
        poll_seconds=0.005,
    )
    evaluator = RecordingEvaluator(delay=0.05)
    adapter = AutoResearcherEvaluatorAdapter(
        normalizer=_normalise,
        evaluator=evaluator,
        component_identity="offline-training-policy-v1",
        evaluator_identity="offline-scientific-evaluator-v1",
        dataset_version="synthetic-no-data-v1",
        code_version="heartbeat-test-v1",
        maximum_evaluations=2,
        resource_broker=broker,
        resource_request_factory=_gpu_request,
        resource_lease_ttl=timedelta(milliseconds=60),
        resource_lease_heartbeat_interval=timedelta(milliseconds=10),
    )
    adapter.bind_candidate(
        "source-heartbeat", parent_source_candidate_id=None, generation=0
    )

    metrics = asyncio.run(adapter.evaluate_program("POLICY = 1", "source-heartbeat"))

    assert metrics["primary_score"] == pytest.approx(0.6)
    assert broker.renewals >= 2


def test_native_checkpoint_resume_preserves_population_archive_and_reuse(
    tmp_path: Path,
) -> None:
    pytest.importorskip("openevolve")
    configuration = _configuration("offline-checkpoint-resume-v1")
    limits = _limits()
    first_model = ScriptedModel(("POLICY = 1", "POLICY = 2"))
    first_evaluator = RecordingEvaluator()
    first = EmbeddedOpenEvolveSearch(
        output_dir=tmp_path,
        initial_source="POLICY = 0\n",
        configuration=configuration,
        limits=limits,
        models=(
            ApprovedModel(name=first_model.model, weight=1.0, adapter=first_model),
        ),
        evaluator=_adapter(first_evaluator),
    )
    before = asyncio.run(first.run(iterations=2))
    checkpoint = next(
        Path(path) for path in before.checkpoint_paths if path.endswith("checkpoint_2")
    )

    second_model = ScriptedModel(("POLICY = 2  # semantic reuse", "POLICY = 3"))
    second_evaluator = RecordingEvaluator()
    second = EmbeddedOpenEvolveSearch(
        output_dir=tmp_path,
        initial_source="POLICY = 0\n",
        configuration=configuration,
        limits=limits,
        models=(
            ApprovedModel(name=second_model.model, weight=1.0, adapter=second_model),
        ),
        evaluator=_adapter(second_evaluator),
    )
    after = asyncio.run(second.run(checkpoint_path=checkpoint))

    assert after.resumed_from_iteration == 2
    assert after.resumed_checkpoint_path == str(checkpoint)
    assert set(before.programme_ids).intersection(after.programme_ids)
    assert any(
        feedback.parent_source_candidate_id in before.programme_ids
        for feedback in after.feedback
    )
    assert after.expensive_evaluations == 4  # seed, 1, 2, then new 3
    assert after.reused_evaluations == 1
    assert len(second_evaluator.requests) == 1
    assert second_model.calls == 2
    assert any(
        "auto_researcher_safe_feedback.json" in prompt for prompt in after.prompts
    )
    assert after.search_identity == before.search_identity
    serialized = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in tmp_path.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".py"}
    ).casefold()
    assert "api_key" not in serialized
    assert "secret_should_not_serialize" not in serialized


def test_native_and_outer_budget_ceiling_use_minimum(tmp_path: Path) -> None:
    pytest.importorskip("openevolve")
    model = ScriptedModel(tuple(f"POLICY = {item}" for item in range(1, 6)))
    evaluator = RecordingEvaluator()
    runtime = EmbeddedOpenEvolveSearch(
        output_dir=tmp_path,
        initial_source="POLICY = 0\n",
        configuration=_configuration("offline-outer-budget-v1", maximum_iterations=5),
        limits=_limits(2),
        models=(ApprovedModel(name=model.model, weight=1.0, adapter=model),),
        evaluator=_adapter(evaluator),
    )

    result = asyncio.run(runtime.run())

    assert model.calls == 2
    assert len(result.prompts) == 2
    assert result.expensive_evaluations == 3


def test_native_weighted_model_ensemble_uses_approved_adapters() -> None:
    pytest.importorskip("openevolve")
    from openevolve.llm.ensemble import LLMEnsemble

    first = ScriptedModel(tuple("POLICY = 1" for _ in range(20)))
    second = ScriptedModel(tuple("POLICY = 2" for _ in range(20)))
    runtime = EmbeddedOpenEvolveSearch(
        output_dir=Path("unused"),
        initial_source="POLICY = 0\n",
        configuration=_configuration("offline-ensemble-v1"),
        limits=_limits(),
        models=(
            ApprovedModel(name="approved-first", weight=1.0, adapter=first),
            ApprovedModel(name="approved-second", weight=2.0, adapter=second),
        ),
        evaluator=_adapter(RecordingEvaluator()),
    )
    ensemble = LLMEnsemble(runtime._native_config().llm.models)

    async def sample() -> None:
        for _ in range(12):
            await ensemble.generate("bounded-safe-prompt")

    asyncio.run(sample())

    assert first.calls > 0
    assert second.calls > first.calls
    assert all(
        "bounded-safe-prompt" in prompt for prompt in first.prompts + second.prompts
    )


def test_approved_model_bridge_uses_durable_boundary_without_credentials() -> None:
    component = EvolvableComponentSpec(
        component_id="offline-component",
        component_version="1",
        mutable_file="candidate.py",
        allowed_files=("candidate.py",),
        entry_point="evolve",
        immutable_interface_contract="evolve(configuration) returns policy",
        parameter_schema={"type": "object"},
        output_schema={"type": "object"},
        seed_source="POLICY = 0",
        maximum_source_bytes=1_000,
    )

    class Bridge:
        def __init__(self) -> None:
            self.request: dict[str, Any] | None = None
            self.search_request_id: str | None = None

        def bind_search_request(self, search_request_id: str) -> None:
            self.search_request_id = search_request_id

        def complete(self, request: dict[str, Any], reservation_id: str):
            assert reservation_id == "native-mutation-test"
            self.request = request
            return {"source": "POLICY = 1"}, object()

    bridge = Bridge()
    model = ApprovedModelBridgeLLM(
        bridge,
        model="approved-offline",
        component=component,
        search_request_id="search-approved-native",
    )
    response = asyncio.run(
        model.generate_with_context(
            "safe system strategy",
            [{"role": "user", "content": "safe aggregate score improved"}],
            auto_researcher_parent={
                "id": "source-parent",
                "authoritative_candidate_id": "source-parent",
                "code": "POLICY = 0",
                "generation": 0,
            },
            auto_researcher_mutation_reservation_id="native-mutation-test",
        )
    )

    assert response == "```python\nPOLICY = 1\n```"
    assert bridge.search_request_id == "search-approved-native"
    assert bridge.request is not None
    validate_metadata_only_request(
        bridge.request,
        expected_exposure_identity=metadata_only_model_exposure_identity(component),
    )
    rendered = str(bridge.request).casefold()
    assert "credential" not in rendered
    assert "api_key" not in rendered
    assert "safe aggregate score improved" in rendered


def test_native_template_stochasticity_is_seeded_and_traceable() -> None:
    pytest.importorskip("openevolve")
    import random

    from openevolve.config import PromptConfig
    from openevolve.prompt.sampler import PromptSampler

    def sequence() -> tuple[str, ...]:
        random.seed(71)
        sampler = PromptSampler(
            PromptConfig(
                use_template_stochasticity=True,
                template_variations={"strategy": ["explore", "exploit"]},
            )
        )
        sampler.template_manager.templates["full_rewrite_user"] += "\n{strategy}"
        return tuple(
            sampler.build_prompt(diff_based_evolution=False)["user"].splitlines()[-1]
            for _ in range(6)
        )

    first = sequence()
    assert first == sequence()
    assert set(first) == {"explore", "exploit"}


def test_native_full_rewrite_and_diff_modes() -> None:
    pytest.importorskip("openevolve")
    from openevolve.utils.code_utils import apply_diff, parse_full_rewrite

    source = "def score():\n    return 1\n"
    rewritten = parse_full_rewrite(
        "```python\ndef score():\n    return 2\n```", "python"
    )
    patched = apply_diff(
        source,
        "<<<<<<< SEARCH\n    return 1\n=======\n    return 3\n>>>>>>> REPLACE",
    )

    assert "return 2" in rewritten
    assert "return 3" in patched


def test_safe_embedding_adapter_never_receives_protected_context() -> None:
    class Provider:
        def __init__(self) -> None:
            self.values: list[str] = []

        def embed(self, permitted_candidate_source: str) -> tuple[float, ...]:
            self.values.append(permitted_candidate_source)
            return (0.1, 0.2)

    provider = Provider()
    adapter = SafeEmbeddingAdapter(provider, maximum_source_bytes=100)

    assert adapter.get_embedding("POLICY = 2") == [0.1, 0.2]
    assert provider.values == ["POLICY = 2"]
    with pytest.raises(ValueError, match="source_invalid"):
        adapter.get_embedding("x" * 101)


def test_offline_a4_like_information_flow_is_complete(a3_run: A3Run) -> None:
    children = tuple(
        feedback for feedback in a3_run.result.feedback if feedback.generation > 0
    )
    assert len(a3_run.result.programme_ids) > 1
    assert max(item.generation for item in children) >= 1
    evolutionary_context_ids = {
        identity
        for decision in a3_run.result.decisions
        for identity in (
            decision.parent_source_candidate_id,
            *decision.inspiration_source_candidate_ids,
        )
    }
    assert len(evolutionary_context_ids) > 1
    assert a3_run.result.archive_ids
    assert a3_run.result.feature_cells > 1
    assert a3_run.result.reused_evaluations == 2
    assert a3_run.result.best_source_candidate_id in a3_run.result.programme_ids
