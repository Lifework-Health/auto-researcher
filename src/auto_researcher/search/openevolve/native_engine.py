"""Full embedded OpenEvolve controller with Auto Researcher boundary adapters."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field, field_validator

from auto_researcher.resources import (
    ResourceAdmission,
    ResourceBroker,
    ResourceRequest,
    cuda_environment_for_lease,
)
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
    SearchRequest,
    VerificationResult,
)
from auto_researcher.search.openevolve.capabilities import (
    CAPABILITY_MANIFEST_VERSION,
)
from auto_researcher.search.openevolve.identity import (
    candidate_id,
    openevolve_hash,
    source_hash,
)
from auto_researcher.search.openevolve.models import (
    CandidateExecutionStatus,
    CandidateStatus,
    CandidateValidationStatus,
    EvolvableComponentSpec,
    OpenEvolveCandidate,
)
from auto_researcher.search.openevolve.protocols import ScientificCandidateComponent
from auto_researcher.search.openevolve.upstream import mutation_constraints
from auto_researcher.search.openevolve.upstream_models import (
    UPSTREAM_COMMIT,
    UPSTREAM_PACKAGE_VERSION,
)

SAFE_FEEDBACK_VERSION = "openevolve-safe-feedback-v1"
SCIENTIFIC_CANDIDATE_IDENTITY_VERSION = "scientific-candidate-v1"
SCIENTIFIC_EVALUATION_IDENTITY_VERSION = "scientific-evaluation-v1"


class NativeEngineModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScientificCandidateIdentity(NativeEngineModel):
    identity_version: str = SCIENTIFIC_CANDIDATE_IDENTITY_VERSION
    source_candidate_id: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_candidate_id: str = Field(pattern=r"^scientific-[0-9a-f]{24}$")
    evaluation_identity: str = Field(pattern=r"^evaluation-[0-9a-f]{24}$")
    canonical_configuration: dict[str, Any]


class SafeEvolutionFeedback(NativeEngineModel):
    protocol_version: str = SAFE_FEEDBACK_VERSION
    source_candidate_id: str = Field(min_length=1)
    canonical_candidate_id: str = Field(pattern=r"^scientific-[0-9a-f]{24}$")
    evaluation_identity: str = Field(pattern=r"^evaluation-[0-9a-f]{24}$")
    evaluation_artifact_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_scientific_summary: dict[str, Any]
    parent_source_candidate_id: str | None = None
    generation: int = Field(ge=0)
    primary_score: float
    secondary_metrics: dict[str, float] = Field(default_factory=dict)
    verified: bool
    constraint_compliant: bool
    evaluation_status: str = Field(pattern=r"^(EXECUTED|REUSED)$")
    delta_from_parent: float | None = None
    delta_from_champion: float | None = None
    safe_artifact_summaries: tuple[str, ...] = ()
    safe_failure_classification: str | None = None
    resource_id: str | None = None

    @field_validator("canonical_scientific_summary")
    @classmethod
    def scientific_summary_is_bounded_and_safe(
        cls, value: dict[str, Any]
    ) -> dict[str, Any]:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        prohibited = (
            "credential",
            "holdout_label",
            "patient",
            "secret",
            "subject_id",
        )
        if len(encoded.encode("utf-8")) > 16_000 or any(
            token in encoded.casefold() for token in prohibited
        ):
            raise ValueError("unsafe_openevolve_scientific_summary")
        return value

    @field_validator("safe_artifact_summaries")
    @classmethod
    def feedback_never_contains_protected_context(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        prohibited = (
            "/",
            "\\",
            "credential",
            "holdout label",
            "mri",
            "patient",
            "secret",
            "subject",
            "voxel",
        )
        if len(values) > 16 or any(
            not value.strip()
            or len(value) > 500
            or any(token in value.casefold() for token in prohibited)
            for value in values
        ):
            raise ValueError("unsafe_openevolve_feedback")
        return values


class ScientificEvaluation(NativeEngineModel):
    primary_score: float
    secondary_metrics: dict[str, float] = Field(default_factory=dict)
    verified: bool = True
    constraint_compliant: bool = True
    safe_artifact_summaries: tuple[str, ...] = ()
    safe_failure_classification: str | None = None
    evaluation_artifact_identity: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class EmbeddedEvaluationRequest(NativeEngineModel):
    source_candidate_id: str
    parent_source_candidate_id: str | None
    generation: int
    source: str
    identity: ScientificCandidateIdentity
    resource_admission: ResourceAdmission | None = None
    process_environment: Mapping[str, str] | None = None


class NativeEvolutionLimits(NativeEngineModel):
    maximum_iterations: int = Field(gt=0)
    maximum_model_calls: int = Field(gt=0)
    maximum_candidate_evaluations: int = Field(gt=0)
    maximum_wall_time_seconds: float = Field(gt=0)
    target_score: float | None = None


class NativeEvolutionConfiguration(NativeEngineModel):
    search_identity: str = Field(min_length=1)
    population_size: int = Field(gt=1)
    archive_size: int = Field(gt=0)
    num_islands: int = Field(gt=0)
    migration_interval: int = Field(gt=0)
    migration_rate: float = Field(gt=0, le=1)
    feature_dimensions: tuple[str, ...] = Field(min_length=1)
    feature_bins: int = Field(gt=1)
    parallel_evaluations: int = Field(gt=0)
    checkpoint_interval: int = Field(gt=0)
    random_seed: int
    diff_based_evolution: bool = False
    use_template_stochasticity: bool = True
    template_variations: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    num_top_programs: int = Field(default=3, ge=0)
    num_diverse_programs: int = Field(default=2, ge=0)
    native_max_iterations: int = Field(gt=0)


class NativeEvolutionResult(NativeEngineModel):
    search_identity: str
    resumed_from_iteration: int = Field(ge=0)
    resumed_checkpoint_path: str | None = None
    best_source_candidate_id: str
    best_canonical_candidate_id: str
    programme_ids: tuple[str, ...]
    archive_ids: tuple[str, ...]
    island_programme_ids: tuple[tuple[str, ...], ...]
    feature_cells: int
    expensive_evaluations: int
    reused_evaluations: int
    feedback: tuple[SafeEvolutionFeedback, ...]
    prompts: tuple[str, ...]
    checkpoint_paths: tuple[str, ...]
    decisions: tuple["NativeEvolutionDecision", ...]


class NativeEvolutionDecision(NativeEngineModel):
    source_candidate_id: str
    parent_source_candidate_id: str
    inspiration_source_candidate_ids: tuple[str, ...]
    generation: int = Field(ge=1)
    island: int = Field(ge=0)


def native_configuration_from_search_space(
    search_identity: str,
    search_space: Mapping[str, Any],
) -> NativeEvolutionConfiguration:
    """Translate the bounded public search mapping into the native engine model."""

    raw = search_space.get("openevolve", search_space)
    if not isinstance(raw, Mapping) or raw.get("native_controller") is not True:
        raise ValueError("openevolve_native_controller_configuration_required")
    population_size = int(raw["population_size"])
    return NativeEvolutionConfiguration(
        search_identity=search_identity,
        population_size=population_size,
        archive_size=int(raw.get("archive_size", population_size)),
        num_islands=int(raw.get("num_islands", 1)),
        migration_interval=int(raw.get("migration_interval", 10)),
        migration_rate=float(raw.get("migration_rate", 0.1)),
        feature_dimensions=tuple(raw.get("feature_dimensions", ("complexity",))),
        feature_bins=int(raw.get("feature_bins", 10)),
        parallel_evaluations=int(raw.get("parallel_evaluations", 1)),
        checkpoint_interval=int(raw.get("checkpoint_interval", 10)),
        random_seed=int(raw["random_seed"]),
        diff_based_evolution=bool(raw.get("diff_based_evolution", False)),
        use_template_stochasticity=bool(raw.get("use_template_stochasticity", True)),
        template_variations={
            str(key): tuple(str(item) for item in values)
            for key, values in dict(raw.get("template_variations", {})).items()
        },
        num_top_programs=int(raw.get("num_top_programs", 3)),
        num_diverse_programs=int(raw.get("num_diverse_programs", 2)),
        native_max_iterations=int(raw["maximum_generations"]),
    )


def native_limits_from_search_space(
    search_space: Mapping[str, Any],
) -> NativeEvolutionLimits:
    raw = search_space.get("openevolve", search_space)
    if not isinstance(raw, Mapping):
        raise ValueError("openevolve_native_limits_required")
    return NativeEvolutionLimits(
        maximum_iterations=int(raw["maximum_generations"]),
        maximum_model_calls=int(raw["maximum_model_calls"]),
        maximum_candidate_evaluations=int(raw["maximum_candidate_evaluations"]),
        maximum_wall_time_seconds=float(raw["maximum_wall_time_seconds"]),
        target_score=(
            None
            if raw.get("objective_threshold") is None
            else float(raw["objective_threshold"])
        ),
    )


class CandidateNormalizer(Protocol):
    def __call__(self, source: str) -> Mapping[str, Any]: ...


class ScientificEvaluator(Protocol):
    def __call__(self, request: EmbeddedEvaluationRequest) -> ScientificEvaluation: ...


class ScientificEvaluationCoordinator(Protocol):
    def __call__(
        self,
        experiment: ExperimentSpec,
        request: EmbeddedEvaluationRequest,
    ) -> tuple[EvaluationResult, VerificationResult]: ...


class SafeEmbeddingProvider(Protocol):
    def embed(self, permitted_candidate_source: str) -> Sequence[float]: ...


class NativeLLMAdapter(Protocol):
    model: str

    async def generate(self, prompt: str, **kwargs: Any) -> str: ...

    async def generate_with_context(
        self,
        system_message: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str: ...


class SafeEmbeddingAdapter:
    """Bound optional novelty embeddings to permitted candidate source only."""

    def __init__(
        self,
        provider: SafeEmbeddingProvider,
        *,
        maximum_source_bytes: int,
    ) -> None:
        self.provider = provider
        self.maximum_source_bytes = maximum_source_bytes

    def get_embedding(self, permitted_candidate_source: str) -> list[float]:
        if (
            not permitted_candidate_source
            or len(permitted_candidate_source.encode("utf-8"))
            > self.maximum_source_bytes
        ):
            raise ValueError("openevolve_embedding_source_invalid")
        values = [
            float(value) for value in self.provider.embed(permitted_candidate_source)
        ]
        if not values:
            raise ValueError("openevolve_embedding_response_invalid")
        return values


class TaskOwnedCandidateNormalizer:
    """Validate and execute source through the existing evolvable component."""

    def __init__(self, backend: Any, search_contract: Any) -> None:
        self.backend = backend
        self.search_contract = search_contract
        self._prepared: dict[str, tuple[OpenEvolveCandidate, Any]] = {}
        self._lock = threading.RLock()

    def __call__(self, source: str) -> Mapping[str, Any]:
        digest = source_hash(source)
        with self._lock:
            cached = self._prepared.get(digest)
            if cached is not None:
                return self._canonical(cached[1])
        seed = self.backend.seed_candidate(self.search_contract)
        source_candidate = seed.model_copy(
            update={
                "candidate_id": candidate_id(
                    search_request_id=self.search_contract.search_request_id,
                    component_interface_hash=self.backend.interface_hash,
                    source_sha256=digest,
                ),
                "source_payload": source,
                "source_hash": digest,
                "mutation_operator": "native-openevolve",
                "mutation_description": "Pinned native OpenEvolve source candidate.",
                "status": CandidateStatus.PROPOSED,
                "creation_provenance": "NATIVE_OPENEVOLVE",
            }
        )
        validation = self.backend.validate(source_candidate)
        if validation.status is not CandidateValidationStatus.VALID:
            raise ValueError(
                validation.safe_error_code or "openevolve_candidate_validation_failed"
            )
        source_candidate = source_candidate.model_copy(
            update={
                "validation_result": validation,
                "status": CandidateStatus.VALIDATED,
            }
        )
        preparation = self.backend.prepare(source_candidate, self.search_contract)
        if preparation.execution_status is not CandidateExecutionStatus.COMPLETED:
            raise ValueError(
                preparation.safe_error_code or "openevolve_candidate_preparation_failed"
            )
        with self._lock:
            self._prepared[digest] = (source_candidate, preparation)
        return self._canonical(preparation)

    def _canonical(self, preparation: Any) -> Mapping[str, Any]:
        component = self.backend.component
        if isinstance(component, ScientificCandidateComponent):
            return component.canonical_scientific_configuration(preparation)
        return dict(preparation.generated_configuration)

    def prepared_candidate(self, source: str) -> tuple[OpenEvolveCandidate, Any]:
        """Return the exact validated preparation for experiment construction."""

        digest = source_hash(source)
        self(source)
        with self._lock:
            return self._prepared[digest]


class TaskOwnedScientificEvaluator:
    """Delegate evaluation, verification, reuse, and evidence to AR coordination."""

    def __init__(
        self,
        *,
        normalizer: TaskOwnedCandidateNormalizer,
        search_request: SearchRequest,
        research_contract: ResearchContract,
        metadata: Any,
        run_id: str,
        coordinator: ScientificEvaluationCoordinator,
    ) -> None:
        self.normalizer = normalizer
        self.search_request = search_request
        self.research_contract = research_contract
        self.metadata = metadata
        self.run_id = run_id
        self.coordinator = coordinator

    def __call__(self, request: EmbeddedEvaluationRequest) -> ScientificEvaluation:
        candidate, preparation = self.normalizer.prepared_candidate(request.source)
        candidate = candidate.model_copy(
            update={
                "parent_candidate_ids": (
                    ()
                    if request.parent_source_candidate_id is None
                    else (request.parent_source_candidate_id,)
                ),
                "generation": request.generation,
            }
        )
        experiment = self.normalizer.backend.component.candidate_to_experiment(
            candidate,
            preparation,
            self.search_request,
            self.research_contract,
            self.metadata,
            run_id=self.run_id,
        )
        evaluation, verification = self.coordinator(experiment, request)
        if (
            evaluation.experiment_id != experiment.experiment_id
            or verification.experiment_id != experiment.experiment_id
        ):
            raise ValueError("openevolve_scientific_coordinator_identity_conflict")
        artifact_identity = openevolve_hash(
            "openevolve-scientific-evaluation-artifact-v1",
            {
                "experiment": experiment,
                "evaluation": evaluation,
                "verification": verification,
            },
        )
        numeric_metrics = {
            str(key): float(value)
            for key, value in evaluation.metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        numeric_metrics = dict(tuple(sorted(numeric_metrics.items()))[:32])
        return ScientificEvaluation(
            primary_score=(
                float(evaluation.primary_score)
                if evaluation.primary_score is not None
                else 0.0
            ),
            secondary_metrics=numeric_metrics,
            verified=verification.verified,
            constraint_compliant=verification.constraint_compliant,
            safe_artifact_summaries=(
                f"evaluation artifacts {len(evaluation.artefact_references)}",
                f"verification reasons {len(verification.reasons)}",
            ),
            safe_failure_classification=(
                None if evaluation.success else "scientific_evaluation_failed"
            ),
            evaluation_artifact_identity=artifact_identity,
        )


class ApprovedModelBridgeLLM:
    """Expose an approved durable model bridge as an upstream LLMInterface."""

    def __init__(
        self,
        bridge: Any,
        *,
        model: str,
        component: EvolvableComponentSpec,
        search_request_id: str,
    ) -> None:
        self.bridge = bridge
        self.model = model
        self.component = component
        self.search_request_id = search_request_id

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.generate_with_context(
            system_message="",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self,
        system_message: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        parent = kwargs.pop("auto_researcher_parent", None)
        reservation_id = kwargs.pop("auto_researcher_mutation_reservation_id", None)
        if (
            kwargs
            or not isinstance(parent, dict)
            or not isinstance(reservation_id, str)
        ):
            raise ValueError("openevolve_native_model_context_invalid")
        constraints = mutation_constraints(self.component)
        native_prompt = (
            system_message + "\n" + "\n".join(item["content"] for item in messages)
        )
        request = {
            "protocol": "upstream-adapter-mutation-request-v2",
            "parent": parent,
            "mutable_file": self.component.mutable_file,
            "interface_contract": self.component.immutable_interface_contract,
            "maximum_source_bytes": self.component.maximum_source_bytes,
            "mutation_constraints": constraints.model_dump(mode="json"),
            "native_evolution_prompt": native_prompt,
        }
        bind_search_request = getattr(self.bridge, "bind_search_request", None)
        if bind_search_request is not None:
            bind_search_request(self.search_request_id)
        response, _reservation = await asyncio.to_thread(
            self.bridge.complete,
            request,
            reservation_id,
        )
        source = response.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("openevolve_native_model_response_invalid")
        return f"```python\n{source}\n```"


@dataclass(frozen=True)
class ApprovedModel:
    name: str
    weight: float
    adapter: NativeLLMAdapter


class _NativeModelProxy:
    """Satisfy the exact upstream LLM surface without exposing provider config."""

    def __init__(self, name: str, adapter: NativeLLMAdapter) -> None:
        # OpenEvolve 0.3.2 reads ``model`` from the instance dictionary while
        # logging ensemble selection, so a class attribute is insufficient.
        self.model = name
        self._adapter = adapter

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self._adapter.generate(prompt, **kwargs)

    async def generate_with_context(
        self,
        system_message: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        return await self._adapter.generate_with_context(
            system_message,
            messages,
            **kwargs,
        )


def scientific_candidate_identity(
    *,
    source_candidate_id: str,
    source: str,
    canonical_configuration: Mapping[str, Any],
    component_identity: str,
    evaluator_identity: str,
    dataset_version: str,
    code_version: str,
) -> ScientificCandidateIdentity:
    canonical_id_hash = openevolve_hash(
        SCIENTIFIC_CANDIDATE_IDENTITY_VERSION,
        {
            "component_identity": component_identity,
            "canonical_configuration": dict(canonical_configuration),
        },
    )
    canonical_id = f"scientific-{canonical_id_hash[:24]}"
    evaluation_hash = openevolve_hash(
        SCIENTIFIC_EVALUATION_IDENTITY_VERSION,
        {
            "canonical_candidate_id": canonical_id,
            "evaluator_identity": evaluator_identity,
            "dataset_version": dataset_version,
            "code_version": code_version,
        },
    )
    return ScientificCandidateIdentity(
        source_candidate_id=source_candidate_id,
        source_hash=source_hash(source),
        canonical_candidate_id=canonical_id,
        evaluation_identity=f"evaluation-{evaluation_hash[:24]}",
        canonical_configuration=dict(canonical_configuration),
    )


class AutoResearcherEvaluatorAdapter:
    """Native evaluator surface with semantic reuse and safe feedback artifacts."""

    def __init__(
        self,
        *,
        normalizer: CandidateNormalizer,
        evaluator: ScientificEvaluator,
        component_identity: str,
        evaluator_identity: str,
        dataset_version: str,
        code_version: str,
        maximum_evaluations: int,
        resource_broker: ResourceBroker | None = None,
        resource_request_factory: Callable[
            [ScientificCandidateIdentity], ResourceRequest | None
        ]
        | None = None,
        worker_id: str = "openevolve-worker",
        resource_lease_ttl: timedelta = timedelta(hours=24),
        resource_lease_heartbeat_interval: timedelta | None = None,
        base_process_environment: Mapping[str, str] | None = None,
        reuse_validator: Callable[[SafeEvolutionFeedback], None] | None = None,
    ) -> None:
        if (resource_broker is None) != (resource_request_factory is None):
            raise ValueError("resource broker and request factory must be paired")
        self.normalizer = normalizer
        self.evaluator = evaluator
        self.component_identity = component_identity
        self.evaluator_identity = evaluator_identity
        self.dataset_version = dataset_version
        self.code_version = code_version
        self.maximum_evaluations = maximum_evaluations
        self.resource_broker = resource_broker
        self.resource_request_factory = resource_request_factory
        self.worker_id = worker_id
        self.resource_lease_ttl = resource_lease_ttl
        self.resource_lease_heartbeat_interval = (
            resource_lease_heartbeat_interval or resource_lease_ttl / 4
        )
        if (
            self.resource_lease_ttl <= timedelta(0)
            or self.resource_lease_heartbeat_interval <= timedelta(0)
            or self.resource_lease_heartbeat_interval > resource_lease_ttl / 3
        ):
            raise ValueError("openevolve_resource_lease_heartbeat_invalid")
        self.base_process_environment = base_process_environment
        self.reuse_validator = reuse_validator
        self.database: Any | None = None
        self.expensive_evaluations = 0
        self.reused_evaluations = 0
        self.feedback: list[SafeEvolutionFeedback] = []
        self.placements: list[tuple[str, str]] = []
        self._cache: dict[str, tuple[dict[str, float], SafeEvolutionFeedback]] = {}
        self._pending_artifacts: dict[str, dict[str, str]] = {}
        self._pending_metadata: dict[str, dict[str, Any]] = {}
        self._bindings: dict[str, tuple[str | None, int]] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._inflight_errors: dict[str, BaseException] = {}
        self._lock = threading.RLock()
        self._cache_loaded = False
        self._reuse_index_path: Path | None = None

    def attach_database(self, database: Any) -> None:
        self.database = database

    def attach_reuse_index(self, path: Path) -> None:
        """Bind durable, safe evaluator reuse evidence to one search identity."""

        self._reuse_index_path = path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = payload["evaluations"]
            if payload["protocol"] != "scientific-evaluation-reuse-index-v1":
                raise ValueError
            loaded: dict[str, tuple[dict[str, float], SafeEvolutionFeedback]] = {}
            for evaluation_identity, item in entries.items():
                feedback = SafeEvolutionFeedback.model_validate(item["feedback"])
                metrics = {
                    str(key): float(value) for key, value in item["metrics"].items()
                }
                if (
                    evaluation_identity != feedback.evaluation_identity
                    or feedback.evaluation_status != "EXECUTED"
                ):
                    raise ValueError
                loaded[evaluation_identity] = (metrics, feedback)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("openevolve_scientific_reuse_index_invalid") from exc
        with self._lock:
            self._cache.update(loaded)
            self.expensive_evaluations = len(self._cache)

    def _persist_reuse_index(self) -> None:
        path = self._reuse_index_path
        if path is None:
            return
        with self._lock:
            payload = {
                "protocol": "scientific-evaluation-reuse-index-v1",
                "evaluations": {
                    evaluation_identity: {
                        "metrics": metrics,
                        "feedback": feedback.model_dump(mode="json"),
                    }
                    for evaluation_identity, (metrics, feedback) in sorted(
                        self._cache.items()
                    )
                },
            }
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)

    def bind_candidate(
        self,
        source_candidate_id: str,
        *,
        parent_source_candidate_id: str | None,
        generation: int,
    ) -> None:
        with self._lock:
            self._bindings[source_candidate_id] = (
                parent_source_candidate_id,
                generation,
            )

    def _load_cache_from_native_database(self) -> None:
        with self._lock:
            if self._cache_loaded:
                return
            self._cache_loaded = True
        if self.database is None:
            return
        for program_id in tuple(self.database.programs):
            artifacts = self.database.get_artifacts(program_id) or {}
            encoded = artifacts.get("auto_researcher_safe_feedback.json")
            metrics = self.database.programs[program_id].metrics
            if not isinstance(encoded, str):
                continue
            try:
                feedback = SafeEvolutionFeedback.model_validate_json(encoded)
                evaluation_identity = feedback.evaluation_identity
            except ValueError:
                continue
            with self._lock:
                self._cache.setdefault(
                    evaluation_identity,
                    (dict(metrics), feedback),
                )

    @staticmethod
    def _metrics(
        evaluation: ScientificEvaluation,
        *,
        reused: bool,
    ) -> dict[str, float]:
        return {
            "combined_score": evaluation.primary_score,
            "primary_score": evaluation.primary_score,
            "verified": float(evaluation.verified),
            "constraint_compliant": float(evaluation.constraint_compliant),
            "evaluation_reused": float(reused),
            **evaluation.secondary_metrics,
        }

    def _reuse(
        self,
        program_id: str,
        identity: ScientificCandidateIdentity,
        parent_id: str | None,
        generation: int,
        cached: tuple[dict[str, float], SafeEvolutionFeedback],
    ) -> dict[str, float]:
        metrics, previous = cached
        if self.reuse_validator is not None:
            self.reuse_validator(previous)
        feedback = previous.model_copy(
            update={
                "source_candidate_id": program_id,
                "parent_source_candidate_id": parent_id,
                "generation": generation,
                "evaluation_status": "REUSED",
            }
        )
        reused_metrics = {**metrics, "evaluation_reused": 1.0}
        self.reused_evaluations += 1
        self._record(program_id, identity, reused_metrics, feedback)
        return reused_metrics

    def _record(
        self,
        program_id: str,
        identity: ScientificCandidateIdentity,
        metrics: dict[str, float],
        feedback: SafeEvolutionFeedback,
    ) -> None:
        encoded = feedback.model_dump_json()
        with self._lock:
            self.feedback.append(feedback)
            self._pending_artifacts[program_id] = {
                "auto_researcher_safe_feedback.json": encoded,
            }
            self._pending_metadata[program_id] = {
                "source_candidate_id": program_id,
                "source_hash": identity.source_hash,
                "canonical_candidate_id": identity.canonical_candidate_id,
                "scientific_evaluation_identity": identity.evaluation_identity,
                "evaluation_artifact_identity": (feedback.evaluation_artifact_identity),
                "evaluation_status": feedback.evaluation_status,
                "resource_id": feedback.resource_id,
            }

    async def evaluate_program(
        self,
        program_code: str,
        program_id: str = "",
    ) -> dict[str, float]:
        return await asyncio.to_thread(self._evaluate_sync, program_code, program_id)

    def _evaluate_sync(self, program_code: str, program_id: str) -> dict[str, float]:
        self._load_cache_from_native_database()
        canonical = dict(self.normalizer(program_code))
        identity = scientific_candidate_identity(
            source_candidate_id=program_id,
            source=program_code,
            canonical_configuration=canonical,
            component_identity=self.component_identity,
            evaluator_identity=self.evaluator_identity,
            dataset_version=self.dataset_version,
            code_version=self.code_version,
        )
        with self._lock:
            parent_id, generation = self._bindings.get(program_id, (None, 0))
            cached = self._cache.get(identity.evaluation_identity)
            if cached is not None:
                return self._reuse(program_id, identity, parent_id, generation, cached)
            event = self._inflight.get(identity.evaluation_identity)
            owner = event is None
            if event is None:
                event = threading.Event()
                self._inflight[identity.evaluation_identity] = event
                self._inflight_errors.pop(identity.evaluation_identity, None)
        if not owner:
            event.wait()
            with self._lock:
                error = self._inflight_errors.get(identity.evaluation_identity)
                if error is not None:
                    raise RuntimeError(
                        "openevolve_shared_scientific_evaluation_failed"
                    ) from error
                return self._reuse(
                    program_id,
                    identity,
                    parent_id,
                    generation,
                    self._cache[identity.evaluation_identity],
                )

        admission: ResourceAdmission | None = None
        process_environment: Mapping[str, str] | None = None
        heartbeat_stop = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        heartbeat_errors: list[BaseException] = []
        try:
            with self._lock:
                if self.expensive_evaluations >= self.maximum_evaluations:
                    raise RuntimeError(
                        "openevolve_candidate_evaluation_budget_exhausted"
                    )
                self.expensive_evaluations += 1
            if self.resource_request_factory is not None:
                assert self.resource_broker is not None
                resource_request = self.resource_request_factory(identity)
                if resource_request is not None:
                    admission = self.resource_broker.acquire(
                        resource_request,
                        worker_id=self.worker_id,
                        lease_ttl=self.resource_lease_ttl,
                    )
                    if (
                        admission.lease is not None
                        and admission.lease.resource_id.startswith("gpu:")
                    ):
                        process_environment = cuda_environment_for_lease(
                            admission.lease,
                            base_environment=self.base_process_environment,
                        )
                    if admission.lease is not None:
                        lease_id = admission.lease.lease_id

                        def heartbeat() -> None:
                            interval = (
                                self.resource_lease_heartbeat_interval.total_seconds()
                            )
                            while not heartbeat_stop.wait(interval):
                                try:
                                    assert self.resource_broker is not None
                                    self.resource_broker.renew_lease(
                                        lease_id,
                                        worker_id=self.worker_id,
                                        lease_ttl=self.resource_lease_ttl,
                                    )
                                except BaseException as exc:
                                    heartbeat_errors.append(exc)
                                    heartbeat_stop.set()
                                    return

                        heartbeat_thread = threading.Thread(
                            target=heartbeat,
                            name=f"openevolve-resource-heartbeat-{lease_id[:12]}",
                            daemon=True,
                        )
                        heartbeat_thread.start()
            request = EmbeddedEvaluationRequest(
                source_candidate_id=program_id,
                parent_source_candidate_id=parent_id,
                generation=generation,
                source=program_code,
                identity=identity,
                resource_admission=admission,
                process_environment=process_environment,
            )
            evaluation = self.evaluator(request)
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join()
            if heartbeat_errors:
                raise RuntimeError(
                    "openevolve_resource_lease_heartbeat_failed"
                ) from heartbeat_errors[0]
            evaluation_artifact_identity = (
                evaluation.evaluation_artifact_identity
                or openevolve_hash(
                    "openevolve-scientific-evaluation-artifact-v1",
                    evaluation.model_dump(
                        mode="json",
                        exclude={"evaluation_artifact_identity"},
                    ),
                )
            )
            metrics = self._metrics(evaluation, reused=False)
            resource_id = (
                admission.lease.resource_id
                if admission is not None and admission.lease is not None
                else None
            )
            if resource_id is not None:
                self.placements.append((identity.evaluation_identity, resource_id))
            parent_score = self._parent_score(parent_id)
            champion_score = self._champion_score()
            feedback = SafeEvolutionFeedback(
                source_candidate_id=program_id,
                canonical_candidate_id=identity.canonical_candidate_id,
                evaluation_identity=identity.evaluation_identity,
                evaluation_artifact_identity=evaluation_artifact_identity,
                canonical_scientific_summary=identity.canonical_configuration,
                parent_source_candidate_id=parent_id,
                generation=generation,
                primary_score=evaluation.primary_score,
                secondary_metrics=evaluation.secondary_metrics,
                verified=evaluation.verified,
                constraint_compliant=evaluation.constraint_compliant,
                evaluation_status="EXECUTED",
                delta_from_parent=(
                    None
                    if parent_score is None
                    else evaluation.primary_score - parent_score
                ),
                delta_from_champion=(
                    None
                    if champion_score is None
                    else evaluation.primary_score - champion_score
                ),
                safe_artifact_summaries=evaluation.safe_artifact_summaries,
                safe_failure_classification=evaluation.safe_failure_classification,
                resource_id=resource_id,
            )
            with self._lock:
                self._cache[identity.evaluation_identity] = (metrics, feedback)
            self._persist_reuse_index()
            self._record(program_id, identity, metrics, feedback)
            return metrics
        except BaseException as exc:
            with self._lock:
                self._inflight_errors[identity.evaluation_identity] = exc
            raise
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None and heartbeat_thread.is_alive():
                heartbeat_thread.join()
            if (
                admission is not None
                and admission.lease is not None
                and self.resource_broker is not None
            ):
                self.resource_broker.release_lease(
                    admission.lease.lease_id,
                    worker_id=self.worker_id,
                )
            with self._lock:
                completed = self._inflight.pop(identity.evaluation_identity, None)
                if completed is not None:
                    completed.set()

    def _parent_score(self, parent_id: str | None) -> float | None:
        if parent_id is None or self.database is None:
            return None
        parent = self.database.get(parent_id)
        return None if parent is None else parent.metrics.get("combined_score")

    def _champion_score(self) -> float | None:
        if self.database is None:
            return None
        champion = self.database.get_best_program(metric="combined_score")
        return None if champion is None else champion.metrics.get("combined_score")

    def get_pending_artifacts(self, program_id: str) -> dict[str, str] | None:
        with self._lock:
            return self._pending_artifacts.pop(program_id, None)

    def consume_metadata(self, program_id: str) -> dict[str, Any]:
        with self._lock:
            return self._pending_metadata.pop(program_id, {})


class ResourceBrokerParallelController:
    """Thread execution adapter retaining upstream selection and lifecycle logic."""

    config: Any
    database: Any
    num_workers: int
    shutdown_event: Any

    def bind_auto_researcher_adapters(
        self,
        *,
        llm_ensemble: Any,
        prompt_sampler: Any,
        evaluator: AutoResearcherEvaluatorAdapter,
        prompt_observer: Callable[[str], None],
    ) -> None:
        self.llm_ensemble = llm_ensemble
        self.prompt_sampler = prompt_sampler
        self.safe_evaluator = evaluator
        self.prompt_observer = prompt_observer
        self.executor: ThreadPoolExecutor | None = None
        self._model_lock = threading.Lock()

    def start(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def stop(self) -> None:
        self.shutdown_event.set()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def _submit_iteration(
        self,
        iteration: int,
        island_id: int | None = None,
    ) -> Future | None:
        if self.executor is None:
            return None
        target_island = (
            island_id if island_id is not None else self.database.current_island
        )
        parent, inspirations = self.database.sample_from_island(
            island_id=target_island,
            num_inspirations=self.config.prompt.num_diverse_programs,
        )
        return self.executor.submit(
            self._run_iteration,
            iteration,
            parent,
            inspirations,
            target_island,
        )

    def _run_iteration(
        self,
        iteration: int,
        parent: Any,
        inspirations: list[Any],
        target_island: int,
    ) -> Any:
        from openevolve.database import Program  # type: ignore[import-untyped]
        from openevolve.process_parallel import (  # type: ignore[import-untyped]
            SerializableResult,
        )
        from openevolve.utils.code_utils import (  # type: ignore[import-untyped]
            apply_diff,
            extract_diffs,
            format_diff_summary,
            parse_full_rewrite,
        )

        started = time.monotonic()
        try:
            island_programs = [
                self.database.get(program_id)
                for program_id in self.database.islands[target_island]
            ]
            island_programs = [item for item in island_programs if item is not None]
            island_programs.sort(
                key=lambda item: item.metrics.get("combined_score", 0.0),
                reverse=True,
            )
            visible = island_programs[
                : self.config.prompt.num_top_programs
                + self.config.prompt.num_diverse_programs
            ]
            prompt = self.prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,
                program_metrics=parent.metrics,
                previous_programs=[
                    item.to_dict()
                    for item in island_programs[: self.config.prompt.num_top_programs]
                ],
                top_programs=[item.to_dict() for item in visible],
                inspirations=[item.to_dict() for item in inspirations],
                language=self.config.language,
                evolution_round=iteration,
                diff_based_evolution=self.config.diff_based_evolution,
                program_artifacts=self.database.get_artifacts(parent.id),
                feature_dimensions=self.config.database.feature_dimensions,
            )
            rendered_prompt = f"{prompt['system']}\n{prompt['user']}"
            self.prompt_observer(rendered_prompt)
            with self._model_lock:
                reservation_id = (
                    "native-mutation-"
                    + openevolve_hash(
                        "openevolve-native-mutation-reservation-v1",
                        {
                            "iteration": iteration,
                            "parent_id": parent.id,
                            "parent_source_hash": source_hash(parent.code),
                            "target_island": target_island,
                        },
                    )[:24]
                )
                response = asyncio.run(
                    self.llm_ensemble.generate_with_context(
                        system_message=prompt["system"],
                        messages=[{"role": "user", "content": prompt["user"]}],
                        auto_researcher_parent={
                            "id": parent.id,
                            "authoritative_candidate_id": parent.id,
                            "code": parent.code,
                            "generation": parent.generation,
                        },
                        auto_researcher_mutation_reservation_id=reservation_id,
                    )
                )
            if self.config.diff_based_evolution:
                blocks = extract_diffs(response, self.config.diff_pattern)
                if not blocks:
                    return SerializableResult(
                        error="No valid diffs found in response",
                        iteration=iteration,
                    )
                child_code = apply_diff(parent.code, response, self.config.diff_pattern)
                changes = format_diff_summary(blocks)
            else:
                child_code = parse_full_rewrite(response, self.config.language)
                changes = "Full rewrite"
            if not child_code or len(child_code) > self.config.max_code_length:
                return SerializableResult(
                    error="Generated code is empty or exceeds maximum length",
                    iteration=iteration,
                )
            child_hash = openevolve_hash(
                "openevolve-source-candidate-v1",
                {
                    "iteration": iteration,
                    "parent_id": parent.id,
                    "source_hash": source_hash(child_code),
                },
            )
            child_id = f"source-candidate-{child_hash[:24]}"
            self.safe_evaluator.bind_candidate(
                child_id,
                parent_source_candidate_id=parent.id,
                generation=parent.generation + 1,
            )
            metrics = asyncio.run(
                self.safe_evaluator.evaluate_program(child_code, child_id)
            )
            artifacts = self.safe_evaluator.get_pending_artifacts(child_id)
            metadata = {
                "changes": changes,
                "parent_metrics": parent.metrics,
                "island": target_island,
                "inspiration_ids": [item.id for item in inspirations],
                **self.safe_evaluator.consume_metadata(child_id),
            }
            child = Program(
                id=child_id,
                code=child_code,
                changes_description=changes,
                language=self.config.language,
                parent_id=parent.id,
                generation=parent.generation + 1,
                metrics=metrics,
                iteration_found=iteration,
                metadata=metadata,
            )
            return SerializableResult(
                child_program_dict=child.to_dict(),
                parent_id=parent.id,
                iteration_time=time.monotonic() - started,
                prompt=prompt,
                llm_response=response,
                artifacts=artifacts,
                iteration=iteration,
                target_island=target_island,
            )
        except Exception as exc:
            return SerializableResult(error=str(exc), iteration=iteration)


class EmbeddedOpenEvolveSearch:
    """Run the pinned upstream controller with boundary-only substitutions."""

    def __init__(
        self,
        *,
        output_dir: Path,
        initial_source: str,
        configuration: NativeEvolutionConfiguration,
        limits: NativeEvolutionLimits,
        models: Sequence[ApprovedModel],
        evaluator: AutoResearcherEvaluatorAdapter,
    ) -> None:
        if not models or any(model.weight <= 0 for model in models):
            raise ValueError("approved OpenEvolve model ensemble is required")
        self.output_dir = output_dir
        self.initial_source = initial_source
        self.configuration = configuration
        self.limits = limits
        self.models = tuple(models)
        self.evaluator = evaluator
        self.evaluator.maximum_evaluations = min(
            self.evaluator.maximum_evaluations,
            limits.maximum_candidate_evaluations,
        )
        self.prompts: list[str] = []

    def _search_envelope(self) -> dict[str, Any]:
        return {
            "protocol": "auto-researcher-openevolve-search-envelope-v1",
            "search_identity": self.configuration.search_identity,
            "upstream_version": UPSTREAM_PACKAGE_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "capability_manifest_version": CAPABILITY_MANIFEST_VERSION,
            "configuration": self.configuration.model_dump(mode="json"),
            "limits": self.limits.model_dump(mode="json"),
        }

    def _bind_search_envelope(self) -> None:
        path = self.output_dir / "search-envelope.json"
        envelope = self._search_envelope()
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("openevolve_search_envelope_invalid") from exc
            if existing != envelope:
                raise ValueError("openevolve_search_envelope_mismatch")
            return
        path.write_text(
            json.dumps(envelope, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _checkpoint_iteration(checkpoint_path: Path | None) -> int:
        if checkpoint_path is None:
            return 0
        metadata_path = checkpoint_path / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            iteration = metadata["last_iteration"]
        except (OSError, KeyError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("openevolve_checkpoint_metadata_invalid") from exc
        if not isinstance(iteration, int) or iteration < 0:
            raise ValueError("openevolve_checkpoint_iteration_invalid")
        return iteration

    def _native_config(self) -> Any:
        from openevolve.config import (  # type: ignore[import-untyped]
            Config,
            DatabaseConfig,
            EvaluatorConfig,
            EvolutionTraceConfig,
            LLMConfig,
            LLMModelConfig,
            PromptConfig,
        )

        model_configs = [
            LLMModelConfig(
                name=model.name,
                weight=model.weight,
                random_seed=self.configuration.random_seed,
                init_client=lambda _config, item=model: _NativeModelProxy(
                    item.name,
                    item.adapter,
                ),
            )
            for model in self.models
        ]
        return Config(
            max_iterations=min(
                self.configuration.native_max_iterations,
                self.limits.maximum_iterations,
                self.limits.maximum_model_calls,
            ),
            checkpoint_interval=self.configuration.checkpoint_interval,
            random_seed=self.configuration.random_seed,
            language="python",
            file_suffix=".py",
            diff_based_evolution=self.configuration.diff_based_evolution,
            llm=LLMConfig(models=model_configs, evaluator_models=model_configs),
            prompt=PromptConfig(
                num_top_programs=self.configuration.num_top_programs,
                num_diverse_programs=self.configuration.num_diverse_programs,
                use_template_stochasticity=(
                    self.configuration.use_template_stochasticity
                ),
                template_variations={
                    key: list(values)
                    for key, values in self.configuration.template_variations.items()
                },
            ),
            database=DatabaseConfig(
                in_memory=True,
                population_size=self.configuration.population_size,
                archive_size=self.configuration.archive_size,
                num_islands=self.configuration.num_islands,
                migration_interval=self.configuration.migration_interval,
                migration_rate=self.configuration.migration_rate,
                feature_dimensions=list(self.configuration.feature_dimensions),
                feature_bins=self.configuration.feature_bins,
                random_seed=self.configuration.random_seed,
                embedding_model=None,
            ),
            evaluator=EvaluatorConfig(
                timeout=max(1, int(self.limits.maximum_wall_time_seconds)),
                max_retries=0,
                cascade_evaluation=False,
                parallel_evaluations=self.configuration.parallel_evaluations,
                enable_artifacts=True,
            ),
            evolution_trace=EvolutionTraceConfig(
                enabled=True,
                include_code=False,
                include_prompts=True,
                output_path=str(self.output_dir / "evolution_trace.jsonl"),
                buffer_size=1,
            ),
        )

    async def run(
        self,
        *,
        checkpoint_path: Path | None = None,
        iterations: int | None = None,
    ) -> NativeEvolutionResult:
        from openevolve.controller import OpenEvolve  # type: ignore[import-untyped]
        from openevolve.process_parallel import (  # type: ignore[import-untyped]
            ProcessParallelController,
        )
        import openevolve.controller as controller_module  # type: ignore[import-untyped]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._bind_search_envelope()
        approved_iterations = min(
            self.configuration.native_max_iterations,
            self.limits.maximum_iterations,
            self.limits.maximum_model_calls,
        )
        completed_iterations = self._checkpoint_iteration(checkpoint_path)
        remaining_iterations = max(0, approved_iterations - completed_iterations)
        if iterations is not None:
            if iterations <= 0:
                raise ValueError("openevolve_requested_iterations_invalid")
            remaining_iterations = min(remaining_iterations, iterations)
        initial_path = self.output_dir / "initial_program.py"
        evaluation_path = self.output_dir / "boundary_evaluator.py"
        initial_path.write_text(self.initial_source, encoding="utf-8")
        evaluation_path.write_text(
            "def evaluate(program_path):\n    return {'combined_score': 0.0}\n",
            encoding="utf-8",
        )
        controller = OpenEvolve(
            str(initial_path),
            str(evaluation_path),
            self._native_config(),
            output_dir=str(self.output_dir),
        )
        controller.evaluator = self.evaluator
        self.evaluator.attach_database(controller.database)
        self.evaluator.attach_reuse_index(
            self.output_dir / "scientific-evaluation-reuse.json"
        )
        created_controllers: list[ResourceBrokerParallelController] = []

        def controller_factory(
            config: Any,
            evaluation_file: str,
            database: Any,
            evolution_tracer: Any,
            file_suffix: str = ".py",
        ) -> ResourceBrokerParallelController:
            bound_type = type(
                "BoundResourceBrokerParallelController",
                (ResourceBrokerParallelController, ProcessParallelController),
                {},
            )
            item = bound_type(
                config,
                evaluation_file,
                database,
                evolution_tracer,
                file_suffix,
            )
            item.bind_auto_researcher_adapters(
                llm_ensemble=controller.llm_ensemble,
                prompt_sampler=controller.prompt_sampler,
                evaluator=self.evaluator,
                prompt_observer=self.prompts.append,
            )
            created_controllers.append(item)
            return item

        with patch.object(
            controller_module,
            "ProcessParallelController",
            controller_factory,
        ):
            if remaining_iterations:
                best = await asyncio.wait_for(
                    controller.run(
                        iterations=remaining_iterations,
                        target_score=self.limits.target_score,
                        checkpoint_path=(
                            None if checkpoint_path is None else str(checkpoint_path)
                        ),
                    ),
                    timeout=self.limits.maximum_wall_time_seconds,
                )
            else:
                assert checkpoint_path is not None
                controller.database.load(str(checkpoint_path))
                best = controller.database.get_best_program()
        if best is None:
            raise RuntimeError("openevolve_native_search_produced_no_program")
        canonical_id = str(best.metadata.get("canonical_candidate_id", ""))
        if not canonical_id:
            identity = scientific_candidate_identity(
                source_candidate_id=best.id,
                source=best.code,
                canonical_configuration=dict(self.evaluator.normalizer(best.code)),
                component_identity=self.evaluator.component_identity,
                evaluator_identity=self.evaluator.evaluator_identity,
                dataset_version=self.evaluator.dataset_version,
                code_version=self.evaluator.code_version,
            )
            canonical_id = identity.canonical_candidate_id
        checkpoint_root = self.output_dir / "checkpoints"
        checkpoints = (
            tuple(
                str(path)
                for path in sorted(checkpoint_root.glob("checkpoint_*"))
                if path.is_dir()
            )
            if checkpoint_root.exists()
            else ()
        )
        return NativeEvolutionResult(
            search_identity=self.configuration.search_identity,
            resumed_from_iteration=completed_iterations,
            resumed_checkpoint_path=(
                None if checkpoint_path is None else str(checkpoint_path)
            ),
            best_source_candidate_id=best.id,
            best_canonical_candidate_id=canonical_id,
            programme_ids=tuple(sorted(controller.database.programs)),
            archive_ids=tuple(sorted(controller.database.archive)),
            island_programme_ids=tuple(
                tuple(sorted(island)) for island in controller.database.islands
            ),
            feature_cells=sum(
                len(feature_map)
                for feature_map in controller.database.island_feature_maps
            ),
            expensive_evaluations=self.evaluator.expensive_evaluations,
            reused_evaluations=self.evaluator.reused_evaluations,
            feedback=tuple(self.evaluator.feedback),
            prompts=tuple(self.prompts),
            checkpoint_paths=checkpoints,
            decisions=tuple(
                NativeEvolutionDecision(
                    source_candidate_id=program.id,
                    parent_source_candidate_id=program.parent_id,
                    inspiration_source_candidate_ids=tuple(
                        program.metadata.get("inspiration_ids", ())
                    ),
                    generation=program.generation,
                    island=int(program.metadata["island"]),
                )
                for program in sorted(
                    controller.database.programs.values(),
                    key=lambda item: item.id,
                )
                if program.id.startswith("source-candidate-")
                and program.parent_id is not None
                and "island" in program.metadata
            ),
        )
