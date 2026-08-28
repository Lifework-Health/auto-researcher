"""Standard-runtime assembly for the embedded native OpenEvolve controller."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
    SearchRequest,
    VerificationResult,
)
from auto_researcher.provenance.reuse import EvaluationReuseRecord
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.native_engine import (
    ApprovedModel,
    ApprovedModelBridgeLLM,
    AutoResearcherEvaluatorAdapter,
    EmbeddedEvaluationRequest,
    EmbeddedOpenEvolveSearch,
    NativeEvolutionResult,
    SafeEvolutionFeedback,
    ScientificCandidateIdentity,
    ScientificCoordinationOutcome,
    TaskOwnedCandidateNormalizer,
    TaskOwnedScientificEvaluator,
    native_configuration_from_search_space,
    native_limits_from_search_space,
)


class NativeScientificEvaluationCoordinator:
    """Publish and revalidate authoritative evaluation-reuse-v2 evidence."""

    def __init__(
        self,
        *,
        run_id: str,
        contract: ResearchContract,
        evaluator: Any,
        verifier: Any,
        provenance_store: Any,
        runtime_context: Any,
        dataset_manifest: Any,
        verification_policy: Any,
        dataset_version: str,
        code_version: str,
        clock: Callable,
    ) -> None:
        self.run_id = run_id
        self.contract = contract
        self.dataset_version = dataset_version
        self.code_version = code_version
        self._persistence_lock = threading.RLock()
        self.dependencies = SimpleNamespace(
            evaluator=evaluator,
            verifier=verifier,
            provenance_store=provenance_store,
            runtime_context=runtime_context,
            dataset_manifest=dataset_manifest,
            verification_policy=verification_policy,
            clock=clock,
        )

    @property
    def evaluator_version(self) -> str:
        evaluator = self.dependencies.evaluator
        return str(
            getattr(
                evaluator,
                "reuse_version",
                getattr(evaluator, "version", evaluator.evaluator_id),
            )
        )

    def _verification(
        self,
        experiment: ExperimentSpec,
        evaluation: EvaluationResult,
    ) -> VerificationResult:
        from auto_researcher.graph.nodes.verify import verify_evidence

        state = {
            "run_id": self.run_id,
            "experiment_spec": experiment,
            "evaluation_result": evaluation,
            "contract": self.contract,
        }
        return cast(Any, verify_evidence)(state, self.dependencies)[
            "verification_result"
        ]

    def _outcome_from_record(
        self,
        record: EvaluationReuseRecord,
        identity: ScientificCandidateIdentity,
    ) -> ScientificCoordinationOutcome:
        from auto_researcher.graph.nodes.evaluate import (
            _published_payload,
            validate_reused_evaluation,
        )

        if (
            record.run_id != self.run_id
            or record.scientific_identity_hash != identity.scientific_identity_hash
            or record.evaluator_version != self.evaluator_version
            or record.dataset_version != self.dataset_version
            or record.code_version != self.code_version
        ):
            raise RuntimeError("openevolve_evaluation_reuse_identity_conflict")
        evaluation = cast(Any, validate_reused_evaluation)(record, self.dependencies)
        experiment = _published_payload(
            self.dependencies,  # type: ignore[arg-type]
            record.expected_artefact_references,
            "experiment_spec.json",
            ExperimentSpec,
        )
        verification = self._verification(experiment, evaluation)
        return ScientificCoordinationOutcome(
            evaluation=evaluation,
            verification=verification,
            evaluation_reuse_experiment_id=record.experiment_id,
            evaluation_reuse_identity_hash=payload_hash(record),
        )

    def __call__(
        self,
        experiment: ExperimentSpec,
        request: EmbeddedEvaluationRequest,
    ) -> ScientificCoordinationOutcome:
        from auto_researcher.graph.nodes.evaluate import _validated_published_bundle

        with self._persistence_lock:
            existing = self.dependencies.provenance_store.get_evaluation_reuse(
                self.run_id,
                experiment.experiment_id,
            )
        if existing is not None:
            with self._persistence_lock:
                return self._outcome_from_record(existing, request.identity)

        evaluation = self.dependencies.evaluator.evaluate(experiment, self.contract)
        reuse_record: EvaluationReuseRecord | None = None
        if evaluation.success and evaluation.artefact_references:
            if (
                evaluation.experiment_id != experiment.experiment_id
                or evaluation.evaluator_version != self.evaluator_version
            ):
                raise RuntimeError("completed_evaluation_identity_conflict")
            bundle = _validated_published_bundle(
                experiment,
                evaluation,
                self.dependencies,  # type: ignore[arg-type]
            )
            reuse_record = EvaluationReuseRecord(
                run_id=self.run_id,
                experiment_id=experiment.experiment_id,
                scientific_identity_hash=request.identity.scientific_identity_hash,
                experiment_payload_hash=payload_hash(experiment),
                result_payload_hash=payload_hash(evaluation),
                evaluator_version=self.evaluator_version,
                dataset_version=experiment.dataset_version,
                code_version=experiment.code_version,
                artefact_bundle_hash=bundle.bundle_sha256,
                artefact_bundle_schema_version=bundle.schema_version,
                result_encoding_version=bundle.result_encoding_version,
                expected_artefact_references=bundle.references,
                evaluator_manifest_payload_hash=(
                    bundle.evaluator_manifest_payload_hash
                ),
                completed_at=self.dependencies.clock(),
                result=evaluation,
            )
            with self._persistence_lock:
                self.dependencies.provenance_store.append_evaluation_reuse(reuse_record)
        with self._persistence_lock:
            verification = self._verification(experiment, evaluation)
        return ScientificCoordinationOutcome(
            evaluation=evaluation,
            verification=verification,
            evaluation_reuse_experiment_id=(
                reuse_record.experiment_id if reuse_record is not None else None
            ),
            evaluation_reuse_identity_hash=(
                payload_hash(reuse_record) if reuse_record is not None else None
            ),
        )

    def validate_reuse(
        self,
        identity: ScientificCandidateIdentity,
        feedback: SafeEvolutionFeedback,
    ) -> None:
        experiment_id = feedback.evaluation_reuse_experiment_id
        expected_record_hash = feedback.evaluation_reuse_identity_hash
        if experiment_id is None or expected_record_hash is None:
            raise RuntimeError("openevolve_evaluation_reuse_v2_reference_missing")
        with self._persistence_lock:
            record = self.dependencies.provenance_store.get_evaluation_reuse(
                self.run_id,
                experiment_id,
            )
            if record is None or payload_hash(record) != expected_record_hash:
                raise RuntimeError("openevolve_evaluation_reuse_v2_record_missing")
            outcome = self._outcome_from_record(record, identity)
        evaluation = outcome.evaluation
        verification = outcome.verification
        if (
            evaluation.primary_score != feedback.primary_score
            or verification.verified != feedback.verified
            or verification.constraint_compliant != feedback.constraint_compliant
        ):
            raise RuntimeError("openevolve_evaluation_reuse_v2_projection_conflict")


@dataclass(frozen=True)
class StandardNativeOpenEvolveRuntime:
    """Build EmbeddedOpenEvolveSearch only through standard runtime dependencies."""

    backend: Any
    component: Any
    metadata: Any
    contract: ResearchContract
    run_id: str
    output_root: Path
    evaluator: Any
    verifier: Any
    provenance_store: Any
    runtime_context: Any
    dataset_manifest: Any
    verification_policy: Any
    clock: Callable
    approved_models: Sequence[ApprovedModel] = ()
    approved_bridge: Any | None = None
    approved_bridge_model_name: str | None = None
    resource_broker: Any | None = None
    resource_request_factory: (
        Callable[[ScientificCandidateIdentity], Any | None] | None
    ) = None

    @staticmethod
    def _latest_checkpoint(output_dir: Path) -> Path | None:
        checkpoints = tuple((output_dir / "checkpoints").glob("checkpoint_*"))
        if not checkpoints:
            return None
        return max(
            checkpoints,
            key=EmbeddedOpenEvolveSearch._checkpoint_iteration,
        )

    def _models(self, search_request_id: str) -> tuple[ApprovedModel, ...]:
        if self.approved_models:
            return tuple(self.approved_models)
        if self.approved_bridge is None or self.approved_bridge_model_name is None:
            raise ValueError("native_openevolve_approved_model_bridge_required")
        return (
            ApprovedModel(
                name=self.approved_bridge_model_name,
                weight=1.0,
                adapter=ApprovedModelBridgeLLM(
                    self.approved_bridge,
                    model=self.approved_bridge_model_name,
                    component=self.component.component_spec(),
                    search_request_id=search_request_id,
                ),
            ),
        )

    def run_search(self, request: SearchRequest) -> NativeEvolutionResult:
        configuration = native_configuration_from_search_space(
            payload_hash(
                {
                    "run_id": self.run_id,
                    "search_request": request,
                    "contract_id": self.contract.contract_id,
                }
            ),
            request.search_space,
        )
        limits = native_limits_from_search_space(request.search_space)
        search_contract = self.backend.create_search_contract(request, self.contract)
        normalizer = TaskOwnedCandidateNormalizer(self.backend, search_contract)
        coordinator = NativeScientificEvaluationCoordinator(
            run_id=self.run_id,
            contract=self.contract,
            evaluator=self.evaluator,
            verifier=self.verifier,
            provenance_store=self.provenance_store,
            runtime_context=self.runtime_context,
            dataset_manifest=self.dataset_manifest,
            verification_policy=self.verification_policy,
            dataset_version=self.metadata.dataset_version,
            code_version=self.metadata.code_version,
            clock=self.clock,
        )
        scientific_evaluator = TaskOwnedScientificEvaluator(
            normalizer=normalizer,
            search_request=request,
            research_contract=self.contract,
            metadata=self.metadata,
            run_id=self.run_id,
            coordinator=coordinator,
        )
        evaluator_adapter = AutoResearcherEvaluatorAdapter(
            normalizer=normalizer,
            evaluator=scientific_evaluator,
            component_identity=(
                f"{self.component.component_spec().component_id}@"
                f"{self.component.component_spec().component_version}"
            ),
            evaluator_identity=(
                f"{self.metadata.evaluator_id}@{self.evaluator_version}"
            ),
            dataset_version=self.metadata.dataset_version,
            code_version=self.metadata.code_version,
            maximum_evaluations=limits.maximum_candidate_evaluations,
            resource_broker=self.resource_broker,
            resource_request_factory=self.resource_request_factory,
            worker_id=f"openevolve-{self.run_id}",
            reuse_validator=coordinator.validate_reuse,
        )
        output_dir = self.output_root / ("search-" + payload_hash(request)[:24])
        raw = request.search_space.get("openevolve", request.search_space)
        if not isinstance(raw, Mapping):
            raise ValueError("openevolve_native_controller_configuration_required")
        chunk = raw.get("standard_runtime_iterations_per_invocation")
        if chunk is not None and (type(chunk) is not int or chunk <= 0):
            raise ValueError("openevolve_standard_runtime_iteration_chunk_invalid")
        runtime = EmbeddedOpenEvolveSearch(
            output_dir=output_dir,
            initial_source=self.component.component_spec().seed_source,
            configuration=configuration,
            limits=limits,
            models=self._models(request.request_id),
            evaluator=evaluator_adapter,
        )
        return asyncio.run(
            runtime.run(
                checkpoint_path=self._latest_checkpoint(output_dir),
                iterations=chunk,
            )
        )

    @property
    def evaluator_version(self) -> str:
        return str(
            getattr(
                self.evaluator,
                "reuse_version",
                getattr(self.evaluator, "version", self.evaluator.evaluator_id),
            )
        )
