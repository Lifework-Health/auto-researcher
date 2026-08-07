"""Real-data, non-patient Iris weighted k-NN research task plugin."""

from __future__ import annotations

from typing import Literal

from pydantic import JsonValue

from auto_researcher.agents.models import TaskAgentContext
from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import ResearchContract, SearchRequest
from auto_researcher.search.optuna.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    OptimisationDirection,
    OptunaStudySpec,
)
from auto_researcher.search.optuna.narrowing import narrow_study_spec
from auto_researcher.search.protocols import SearchCapability
from auto_researcher.tasks.iris_knn.configuration import (
    DISTANCE_POWER_CHOICES,
    FEATURE_WEIGHT_HIGH,
    FEATURE_WEIGHT_LOW,
    K_CHOICES,
    OPTUNA_WEIGHT_NAMES,
    baseline_configuration,
    normalise_iris_configuration,
)
from auto_researcher.tasks.iris_knn.evaluator import (
    EVALUATOR_CODE_VERSION,
    EVALUATOR_ID,
    IrisKNNEvaluator,
)
from auto_researcher.tasks.iris_knn.manifests import (
    DATASET_VERSION,
    FOLD_VERSION,
    build_dataset_manifest,
    load_fold_assignments,
    load_iris_rows,
    verify_data_files,
)
from auto_researcher.tasks.iris_knn.verification import IrisKNNVerificationPolicy
from auto_researcher.tasks.models import (
    ArtefactPolicy,
    ExperimentMetadata,
    ReadinessCheck,
    ReadinessResult,
    TaskDescriptor,
    TaskRuntimeContext,
)


class IrisKNNTask:
    task_id = "iris_knn"
    task_version = "1.0"

    def live_mutation_dataset_class(self) -> Literal["public_benchmark"]:
        """Classify only this fixed, non-patient, host-evaluated benchmark."""

        return "public_benchmark"

    def descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="Iris Weighted k-NN Benchmark",
            domain="biology",
            description="Deterministic weighted k-NN classification on fixed real Iris measurements.",
            supported_search_types=frozenset(
                {SearchType.DIRECT, SearchType.OPTUNA, SearchType.OPENEVOLVE}
            ),
            evaluator_id=EVALUATOR_ID,
            verification_policy_id=IrisKNNVerificationPolicy.policy_id,
            configuration_schema_version="iris-knn-configuration-v1",
        )

    def readiness(self, runtime_context: TaskRuntimeContext) -> ReadinessResult:
        data_ok, folds_ok = verify_data_files(runtime_context)
        schema_ok = False
        if data_ok and folds_ok:
            try:
                schema_ok = (
                    len(load_iris_rows(runtime_context)) == 150
                    and len(load_fold_assignments(runtime_context)) == 150
                )
            except Exception:
                schema_ok = False
        checks = (
            ReadinessCheck(
                code="iris_dataset_hash",
                passed=data_ok,
                message="Vendored UCI Iris bytes match the registered SHA-256.",
            ),
            ReadinessCheck(
                code="iris_fold_hash",
                passed=folds_ok,
                message="Fixed fold assignment matches the registered SHA-256.",
            ),
            ReadinessCheck(
                code="iris_dataset_schema",
                passed=schema_ok,
                message="Dataset and five-fold stratification satisfy the registered shape.",
            ),
        )
        errors = tuple(check.code for check in checks if not check.passed)
        return ReadinessResult(ready=not errors, checks=checks, errors=errors)

    def validate_contract(self, contract: ResearchContract) -> None:
        descriptor = self.descriptor()
        if (contract.task_id, contract.task_version) != (
            self.task_id,
            self.task_version,
        ):
            raise ValueError("contract does not target iris_knn@1.0")
        if contract.evaluator_id != descriptor.evaluator_id:
            raise ValueError("contract evaluator does not match Iris task")
        if contract.primary_metric != "mean_balanced_accuracy":
            raise ValueError("Iris primary metric must be mean_balanced_accuracy")
        if contract.objective_version != "iris-balanced-accuracy-v1":
            raise ValueError("Iris objective version mismatch")
        if contract.task_constraints_version != "iris-knn-configuration-v1":
            raise ValueError("Iris constraint version mismatch")
        if not contract.allowed_search_types.issubset(
            descriptor.supported_search_types
        ):
            raise ValueError("contract requests unsupported Iris search types")

    def normalise_configuration(
        self, configuration: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return normalise_iris_configuration(configuration)

    def dataset_manifest(self, runtime_context: TaskRuntimeContext):
        return build_dataset_manifest(runtime_context)

    def experiment_metadata(
        self, runtime_context: TaskRuntimeContext
    ) -> ExperimentMetadata:
        return ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=EVALUATOR_CODE_VERSION,
            dataset_version=DATASET_VERSION,
            provenance=ProvenanceKind.REAL,
        )

    def create_evaluator(self, runtime_context: TaskRuntimeContext) -> IrisKNNEvaluator:
        return IrisKNNEvaluator(
            runtime_context,
            self.experiment_metadata(runtime_context),
            self.dataset_manifest(runtime_context),
        )

    def create_verification_policy(
        self, contract: ResearchContract
    ) -> IrisKNNVerificationPolicy:
        self.validate_contract(contract)
        return IrisKNNVerificationPolicy()

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
                    "openevolve_search",
                    "openevolve_candidate",
                    "openevolve_lineage",
                }
            ),
            prohibited_artefact_types=frozenset(
                {"raw_input_data", "row_predictions", "fold_rows"}
            ),
            contains_sensitive_data=False,
            retention_notes="Aggregate non-patient benchmark artefacts use normal development retention.",
        )

    def create_optuna_study_spec(
        self, contract: ResearchContract, request: SearchRequest
    ) -> OptunaStudySpec:
        self.validate_contract(contract)
        if request.search_type != SearchType.OPTUNA:
            raise ValueError("Iris Optuna study requires an OPTUNA SearchRequest")
        registered = OptunaStudySpec(
            schema_version="1.0",
            task_id=self.task_id,
            task_version=self.task_version,
            search_space_version="iris-knn-configuration-v1",
            direction=OptimisationDirection.MAXIMIZE,
            parameters=(
                *(
                    FloatParameterSpec(
                        name=name, low=FEATURE_WEIGHT_LOW, high=FEATURE_WEIGHT_HIGH
                    )
                    for name in OPTUNA_WEIGHT_NAMES
                ),
                CategoricalParameterSpec(name="k", choices=K_CHOICES),
                CategoricalParameterSpec(
                    name="distance_power", choices=DISTANCE_POWER_CHOICES
                ),
            ),
            trial_budget=request.experiment_budget,
            seed=20260807,
            n_startup_trials=8,
            objective_metric=contract.primary_metric,
            study_metadata={
                "dataset_version": DATASET_VERSION,
                "fold_version": FOLD_VERSION,
                "configuration_schema_version": "iris-knn-configuration-v1",
            },
        )
        return narrow_study_spec(
            registered,
            dict(request.search_space),
            request_experiment_budget=request.experiment_budget,
        )

    def create_evolvable_component(
        self, contract: ResearchContract, runtime_context: TaskRuntimeContext
    ):
        from auto_researcher.tasks.iris_knn.openevolve import IrisKNNEvolvableComponent

        self.validate_contract(contract)
        return IrisKNNEvolvableComponent()

    def create_agent_context(
        self,
        contract: ResearchContract,
        runtime_context: TaskRuntimeContext,
        search_capabilities: dict[SearchType, SearchCapability],
    ) -> TaskAgentContext:
        from auto_researcher.tasks.iris_knn.agents import create_iris_agent_context

        self.validate_contract(contract)
        return create_iris_agent_context(contract, search_capabilities)


def default_iris_contract(
    maximum_cycles: int = 1,
    *,
    search_types: frozenset[SearchType] | None = None,
    maximum_experiments: int | None = None,
) -> ResearchContract:
    return ResearchContract(
        contract_id="iris-knn-benchmark-contract",
        schema_version="1.0",
        task_id="iris_knn",
        task_version="1.0",
        objective_version="iris-balanced-accuracy-v1",
        primary_metric="mean_balanced_accuracy",
        task_constraints_version="iris-knn-configuration-v1",
        question="Which bounded weighted k-NN configuration maximises fixed-fold Iris balanced accuracy?",
        objective="maximise mean five-fold balanced accuracy on the fixed Iris folds",
        constraints={
            "dataset_version": DATASET_VERSION,
            "fold_version": FOLD_VERSION,
            "score_minimum": 0.0,
            "score_maximum": 1.0,
        },
        allowed_search_types=search_types or frozenset({SearchType.DIRECT}),
        evaluator_id=EVALUATOR_ID,
        verifier_id="deterministic-verifier",
        maximum_cycles=maximum_cycles,
        maximum_experiments=maximum_experiments
        if maximum_experiments is not None
        else maximum_cycles,
        maximum_cost=1.0,
        requires_approval_for=frozenset(),
        provenance=ProvenanceKind.REAL,
    )


def default_iris_configuration() -> dict:
    return baseline_configuration()
