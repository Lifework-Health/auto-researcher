"""iCCA Network Based Stratification task plugin."""

from __future__ import annotations

import importlib.util

from pydantic import JsonValue

from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.tasks.icca_nbs.bindings import (
    ICCABindings,
    load_installed_icca_bindings,
)
from auto_researcher.tasks.icca_nbs.configuration import ICCADirectConfiguration
from auto_researcher.tasks.icca_nbs.evaluator_adapter import ICCANBSEvaluatorAdapter
from auto_researcher.tasks.icca_nbs.manifests import (
    ICCA_DATA_FILES,
    build_icca_dataset_manifest,
)
from auto_researcher.tasks.icca_nbs.verification import ICCANBSVerificationPolicy
from auto_researcher.tasks.models import (
    ArtefactPolicy,
    DatasetManifest,
    ExperimentMetadata,
    ReadinessCheck,
    ReadinessResult,
    TaskDescriptor,
    TaskRuntimeContext,
)


class ICCANBSTask:
    task_id = "icca_nbs"
    task_version = "1.0"

    def __init__(self, bindings: ICCABindings | None = None) -> None:
        self._injected_bindings = bindings
        self._resolved_bindings: ICCABindings | None = bindings
        self._manifest_cache: dict[tuple[object, ...], DatasetManifest] = {}

    def _bindings(self) -> ICCABindings:
        if self._resolved_bindings is None:
            self._resolved_bindings = load_installed_icca_bindings()
        return self._resolved_bindings

    def descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="iCCA Network Based Stratification",
            domain="cancer-subtyping",
            description="Adapter over the auto_agent_v2 iCCA scientific evaluator.",
            supported_search_types=frozenset({SearchType.DIRECT}),
            evaluator_id="icca-nbs-v2-evaluator",
            verification_policy_id="icca-nbs-policy-v1",
            configuration_schema_version="1.0",
        )

    def readiness(self, runtime_context: TaskRuntimeContext) -> ReadinessResult:
        dependency_available = (
            self._injected_bindings is not None
            or importlib.util.find_spec("harness") is not None
        )
        data_checks = [
            bool(
                runtime_context.data_dir is not None
                and (runtime_context.data_dir / filename).is_file()
            )
            for filename in ICCA_DATA_FILES
        ]
        checks = (
            ReadinessCheck(
                code="auto_agent_v2_importable",
                passed=dependency_available,
                message=(
                    "auto_agent_v2 bindings are available."
                    if dependency_available
                    else "Install with `pip install -e ../auto_agent_v2`."
                ),
            ),
            ReadinessCheck(
                code="icca_dataset_files",
                passed=all(data_checks),
                message=(
                    "Required iCCA source files are present."
                    if all(data_checks)
                    else "data_dir must contain Combined_binary_matrix.csv and "
                    "Combined_clinical.csv."
                ),
            ),
        )
        errors = tuple(check.message for check in checks if not check.passed)
        return ReadinessResult(
            ready=not errors,
            checks=checks,
            errors=errors,
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
            raise ValueError("contract evaluator_id does not match iCCA NBS")
        if contract.primary_metric != "stability_objective":
            raise ValueError("iCCA primary_metric must be 'stability_objective'")
        if contract.objective_version != "0.9":
            raise ValueError("iCCA objective_version must be '0.9'")
        if contract.task_constraints_version != "0.9":
            raise ValueError("iCCA task_constraints_version must be '0.9'")
        if not contract.allowed_search_types.issubset(
            descriptor.supported_search_types
        ):
            raise ValueError("contract requests unsupported iCCA search types")

    def normalise_configuration(
        self,
        configuration: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        return ICCADirectConfiguration.normalise(
            configuration,
            self._bindings(),
        ).model_dump(mode="json")

    def experiment_metadata(
        self,
        runtime_context: TaskRuntimeContext,
    ) -> ExperimentMetadata:
        bindings = self._bindings()
        manifest = self.dataset_manifest(runtime_context)
        return ExperimentMetadata(
            evaluator_id=self.descriptor().evaluator_id,
            code_version=bindings.code_version,
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.REAL,
        )

    def create_evaluator(
        self,
        runtime_context: TaskRuntimeContext,
    ) -> ICCANBSEvaluatorAdapter:
        bindings = self._bindings()
        return ICCANBSEvaluatorAdapter(
            bindings,
            runtime_context,
            self.experiment_metadata(runtime_context),
            self.dataset_manifest(runtime_context),
        )

    def create_verification_policy(
        self,
        contract: ResearchContract,
    ) -> ICCANBSVerificationPolicy:
        self.validate_contract(contract)
        return ICCANBSVerificationPolicy()

    def dataset_manifest(
        self,
        runtime_context: TaskRuntimeContext,
    ) -> DatasetManifest:
        bindings = self._bindings()
        key = (
            runtime_context.data_dir,
            runtime_context.manifest_created_at,
            runtime_context.task_options.get("objective_version"),
            bindings.package_version,
        )
        if key not in self._manifest_cache:
            self._manifest_cache[key] = build_icca_dataset_manifest(
                runtime_context,
                loader_version=f"harness-{bindings.package_version}",
            )
        return self._manifest_cache[key]

    def artefact_policy(self) -> ArtefactPolicy:
        return ArtefactPolicy(
            allowed_artefact_types=frozenset(
                {
                    "experiment_spec",
                    "evaluation_result",
                    "dataset_manifest",
                    "evaluator_manifest",
                }
            ),
            prohibited_artefact_types=frozenset(
                {
                    "patient_cluster_assignments",
                    "patient_predictions",
                    "raw_clinical_table",
                    "mutation_matrix",
                }
            ),
            contains_sensitive_data=False,
            retention_notes=(
                "PR 2 writes only aggregate, non-patient-level manifests and results."
            ),
        )
