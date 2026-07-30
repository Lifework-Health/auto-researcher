"""Generic offline task plugin used by default CI and examples."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from pydantic import JsonValue

from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.contracts.models import SearchRequest
from auto_researcher.search.optuna.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    IntParameterSpec,
    OptimisationDirection,
    OptunaStudySpec,
)
from auto_researcher.search.optuna.narrowing import narrow_study_spec
from auto_researcher.tasks.models import (
    ArtefactPolicy,
    DatasetManifest,
    ExperimentMetadata,
    ReadinessCheck,
    ReadinessResult,
    TaskDescriptor,
    TaskRuntimeContext,
)
from auto_researcher.tasks.synthetic.configuration import SyntheticConfiguration
from auto_researcher.tasks.synthetic.evaluator import SyntheticEvaluator
from auto_researcher.tasks.synthetic.verification import SyntheticVerificationPolicy

SYNTHETIC_DATA_SEED = b"auto-researcher-synthetic-landscape-v2"
SYNTHETIC_DATA_HASH = hashlib.sha256(SYNTHETIC_DATA_SEED).hexdigest()


class SyntheticTask:
    task_id = "synthetic"
    task_version = "1.0"

    def descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="Deterministic synthetic landscape",
            domain="synthetic",
            description="Offline task proving domain-neutral orchestration.",
            supported_search_types=frozenset({SearchType.DIRECT, SearchType.OPTUNA}),
            evaluator_id="synthetic-evaluator",
            verification_policy_id="synthetic-policy-v1",
            configuration_schema_version="1.0",
        )

    def readiness(self, runtime_context: TaskRuntimeContext) -> ReadinessResult:
        return ReadinessResult(
            ready=True,
            checks=(
                ReadinessCheck(
                    code="synthetic_runtime",
                    passed=True,
                    message="Synthetic task requires no external data or services.",
                ),
            ),
        )

    def validate_contract(self, contract: ResearchContract) -> None:
        descriptor = self.descriptor()
        if (contract.task_id, contract.task_version) != (
            self.task_id,
            self.task_version,
        ):
            raise ValueError(
                f"contract targets {contract.task_id}@{contract.task_version}, "
                f"not {self.task_id}@{self.task_version}"
            )
        if contract.evaluator_id != descriptor.evaluator_id:
            raise ValueError("contract evaluator_id does not match the synthetic task")
        if contract.primary_metric != "objective_score":
            raise ValueError("synthetic primary_metric must be 'objective_score'")
        if contract.objective_version != "1":
            raise ValueError("synthetic objective_version must be '1'")
        if contract.task_constraints_version != "1.0":
            raise ValueError("synthetic task_constraints_version must be '1.0'")
        if not contract.allowed_search_types.issubset(
            descriptor.supported_search_types
        ):
            raise ValueError("contract requests unsupported synthetic search types")

    def normalise_configuration(
        self,
        configuration: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        return SyntheticConfiguration.model_validate(configuration).model_dump(
            mode="json"
        )

    def experiment_metadata(
        self,
        runtime_context: TaskRuntimeContext,
    ) -> ExperimentMetadata:
        manifest = self.dataset_manifest(runtime_context)
        return ExperimentMetadata(
            evaluator_id=self.descriptor().evaluator_id,
            code_version="synthetic-task-1.0",
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.SIMULATED,
        )

    def create_evaluator(self, runtime_context: TaskRuntimeContext) -> SyntheticEvaluator:
        return SyntheticEvaluator(
            runtime_context,
            self.experiment_metadata(runtime_context),
            self.dataset_manifest(runtime_context),
        )

    def create_verification_policy(
        self,
        contract: ResearchContract,
    ) -> SyntheticVerificationPolicy:
        self.validate_contract(contract)
        return SyntheticVerificationPolicy()

    def dataset_manifest(
        self,
        runtime_context: TaskRuntimeContext,
    ) -> DatasetManifest:
        created_at = runtime_context.manifest_created_at or datetime.now(UTC)
        return DatasetManifest(
            task_id=self.task_id,
            dataset_version=f"synthetic:{SYNTHETIC_DATA_HASH[:12]}",
            files=("synthetic-landscape-v2",),
            hashes={"synthetic-landscape-v2": SYNTHETIC_DATA_HASH},
            loader_version="synthetic-loader-v1",
            created_at=created_at,
            metadata={"generator": "deterministic", "contains_patient_data": False},
        )

    def artefact_policy(self) -> ArtefactPolicy:
        return ArtefactPolicy(
            allowed_artefact_types=frozenset(
                {
                    "experiment_spec",
                    "evaluation_result",
                    "dataset_manifest",
                    "evaluator_manifest",
                    "study_spec",
                    "study_summary",
                    "trials_summary",
                    "selected_trial",
                }
            ),
            prohibited_artefact_types=frozenset({"raw_input_data"}),
            contains_sensitive_data=False,
            retention_notes="Synthetic artefacts may use standard development retention.",
        )

    def create_optuna_study_spec(
        self,
        contract: ResearchContract,
        request: SearchRequest,
    ) -> OptunaStudySpec:
        self.validate_contract(contract)
        if request.search_type != SearchType.OPTUNA:
            raise ValueError("synthetic Optuna study requires an OPTUNA SearchRequest")
        registered = OptunaStudySpec(
            schema_version="1.0",
            task_id=self.task_id,
            task_version=self.task_version,
            search_space_version="synthetic-landscape-v1",
            direction=OptimisationDirection.MAXIMIZE,
            parameters=(
                CategoricalParameterSpec(
                    name="model_family",
                    choices=("linear", "tree", "neural"),
                ),
                IntParameterSpec(name="complexity", low=1, high=10),
                FloatParameterSpec(
                    name="learning_rate",
                    low=0.001,
                    high=1.0,
                    log=True,
                ),
            ),
            trial_budget=request.experiment_budget,
            seed=20260730,
            objective_metric=contract.primary_metric,
            study_metadata={"task_kind": "offline_reference"},
        )
        return narrow_study_spec(
            registered,
            dict(request.search_space),
            request_experiment_budget=request.experiment_budget,
        )


def default_synthetic_configuration() -> dict[str, JsonValue]:
    return {
        "model_family": "tree",
        "complexity": 4,
        "learning_rate": 0.05,
    }


def default_synthetic_contract(
    maximum_cycles: int = 1,
    *,
    search_types: frozenset[SearchType] | None = None,
    maximum_experiments: int | None = None,
) -> ResearchContract:
    return ResearchContract(
        contract_id="synthetic-demo-contract",
        schema_version="1.0",
        task_id="synthetic",
        task_version="1.0",
        objective_version="1",
        primary_metric="objective_score",
        task_constraints_version="1.0",
        question="Which bounded synthetic configuration maximises the objective?",
        objective="maximise the deterministic synthetic objective score",
        constraints={
            "support_threshold": 0.75,
            "refute_threshold": 0.4,
            "maximum_runtime": 3.0,
        },
        allowed_search_types=search_types or frozenset({SearchType.DIRECT}),
        evaluator_id="synthetic-evaluator",
        verifier_id="deterministic-verifier",
        maximum_cycles=maximum_cycles,
        maximum_experiments=(
            maximum_experiments
            if maximum_experiments is not None
            else maximum_cycles
        ),
        maximum_cost=1.0,
        requires_approval_for=frozenset(),
        provenance=ProvenanceKind.SIMULATED,
    )
