"""Adapter that delegates all iCCA scientific work to installed auto_agent_v2."""

from __future__ import annotations

from auto_researcher.contracts.enums import ProvenanceKind
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
)
from auto_researcher.tasks.artifacts import (
    artefact_references,
    json_safe,
    write_artefact_bundle,
)
from auto_researcher.tasks.icca_nbs.bindings import ICCABindings
from auto_researcher.tasks.icca_nbs.configuration import (
    ICCADirectConfiguration,
    resolve_enum_alias,
)
from auto_researcher.tasks.icca_nbs.diagnostics import (
    ICCAEvaluationFailureStage,
    classify_scientific_failure,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)


class ICCANBSEvaluatorAdapter:
    evaluator_id = "icca-nbs-v2-evaluator"
    version = "icca-adapter-v1.1"
    cost_per_experiment = 0.0

    def __init__(
        self,
        bindings: ICCABindings,
        context: TaskRuntimeContext,
        metadata: ExperimentMetadata,
        dataset_manifest: DatasetManifest,
    ) -> None:
        self.bindings = bindings
        self.context = context
        self.metadata = metadata
        self.dataset_manifest = dataset_manifest
        self._cohort = None
        self._paths = None
        self._propagation_cache = None

    def _evaluator_manifest(self) -> dict:
        return {
            "task_id": "icca_nbs",
            "task_version": "1.0",
            "evaluator_id": self.evaluator_id,
            "adapter_version": self.version,
            "v2_package_version": self.bindings.package_version,
            "v2_code_version": self.bindings.code_version,
            "dataset_version": self.metadata.dataset_version,
            "provenance": self.metadata.provenance.value,
        }

    def _dataset_fingerprint(self) -> str:
        fingerprint = self.dataset_manifest.metadata.get(
            "combined_dataset_fingerprint",
            self.dataset_manifest.dataset_version,
        )
        return str(json_safe(fingerprint))

    def _failure_diagnostics(
        self,
        experiment: ExperimentSpec,
        configuration: ICCADirectConfiguration,
        *,
        exception_class: str,
        stage: ICCAEvaluationFailureStage,
        dataset_loading_completed: bool,
        propagation_completed: bool,
        clustering_completed: bool,
        eligibility_evaluation_completed: bool,
    ) -> dict:
        return {
            "safe_exception_class": exception_class,
            "failure_stage": stage.value,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.version,
            "experiment_id": experiment.experiment_id,
            "canonical_configuration": configuration.model_dump(mode="json"),
            "dataset_fingerprint": self._dataset_fingerprint(),
            "dataset_loading_completed": dataset_loading_completed,
            "propagation_completed": propagation_completed,
            "clustering_completed": clustering_completed,
            "eligibility_evaluation_completed": eligibility_evaluation_completed,
        }

    def _failure(
        self,
        experiment: ExperimentSpec,
        message: str,
        *,
        diagnostics: dict | None = None,
        write_artefacts: bool = True,
    ) -> EvaluationResult:
        result = EvaluationResult(
            experiment_id=experiment.experiment_id,
            success=False,
            primary_score=None,
            metrics={"failure_diagnostics": diagnostics} if diagnostics else {},
            constraint_results={},
            artefact_references=artefact_references(
                self.context, experiment.experiment_id
            ),
            evaluator_version=self.version,
            provenance=self.metadata.provenance,
            error=message,
        )
        if write_artefacts:
            write_artefact_bundle(
                self.context,
                experiment,
                result,
                self.dataset_manifest,
                self._evaluator_manifest(),
            )
        return result

    def _staged_failure(
        self,
        experiment: ExperimentSpec,
        configuration: ICCADirectConfiguration,
        exc: Exception,
        stage: ICCAEvaluationFailureStage,
        *,
        dataset_loading_completed: bool,
        propagation_completed: bool,
        clustering_completed: bool,
        eligibility_evaluation_completed: bool,
        write_artefacts: bool = True,
    ) -> EvaluationResult:
        exception_class = type(exc).__name__
        diagnostics = self._failure_diagnostics(
            experiment,
            configuration,
            exception_class=exception_class,
            stage=stage,
            dataset_loading_completed=dataset_loading_completed,
            propagation_completed=propagation_completed,
            clustering_completed=clustering_completed,
            eligibility_evaluation_completed=eligibility_evaluation_completed,
        )
        try:
            return self._failure(
                experiment,
                f"icca_evaluation_failed: {stage.value}: {exception_class}",
                diagnostics=diagnostics,
                write_artefacts=write_artefacts,
            )
        except Exception as artefact_exc:
            artefact_diagnostics = self._failure_diagnostics(
                experiment,
                configuration,
                exception_class=type(artefact_exc).__name__,
                stage=ICCAEvaluationFailureStage.ARTEFACT_WRITING,
                dataset_loading_completed=dataset_loading_completed,
                propagation_completed=propagation_completed,
                clustering_completed=clustering_completed,
                eligibility_evaluation_completed=eligibility_evaluation_completed,
            )
            return self._failure(
                experiment,
                "icca_evaluation_failed: ARTEFACT_WRITING: "
                f"{type(artefact_exc).__name__}",
                diagnostics=artefact_diagnostics,
                write_artefacts=False,
            )

    def _map_evaluation_result(
        self,
        experiment: ExperimentSpec,
        configuration: ICCADirectConfiguration,
        v2_result,
        contract: ResearchContract,
        *,
        score: float,
    ) -> EvaluationResult:
        selection_inputs = json_safe(v2_result.selection_inputs)
        eligibility = json_safe(v2_result.eligibility)
        pac = float(json_safe(v2_result.selection_inputs["pac"]))
        provenance = ProvenanceKind(
            str(json_safe(getattr(v2_result, "provenance", "REAL")))
        )
        constraint_results = {
            gate: bool(v2_result.eligibility.get(gate, False))
            for gate in ("logrank_pass", "clinical_pass", "floors_pass")
        }
        metrics = {
            "primary_score": score,
            "stability": 1.0 - pac,
            "scientific": json_safe(v2_result.metrics),
            "selection_inputs": selection_inputs,
            "eligibility": eligibility,
            "per_cluster": json_safe(v2_result.per_cluster),
            "configuration": configuration.model_dump(mode="json"),
            "evaluation_settings": {
                "r": configuration.r,
                "objective_version": contract.objective_version,
            },
        }
        return EvaluationResult(
            experiment_id=experiment.experiment_id,
            success=True,
            primary_score=score,
            metrics=metrics,
            constraint_results=constraint_results,
            artefact_references=artefact_references(
                self.context, experiment.experiment_id
            ),
            evaluator_version=f"{self.version}:{self.bindings.code_version}",
            provenance=provenance,
            error=None,
        )

    def map_evaluation(
        self,
        experiment: ExperimentSpec,
        configuration: ICCADirectConfiguration,
        v2_result,
        contract: ResearchContract,
    ) -> EvaluationResult:
        score = float(self.bindings.stability_objective(v2_result))
        result = self._map_evaluation_result(
            experiment,
            configuration,
            v2_result,
            contract,
            score=score,
        )
        write_artefact_bundle(
            self.context,
            experiment,
            result,
            self.dataset_manifest,
            self._evaluator_manifest(),
        )
        return result

    def evaluate(
        self,
        experiment: ExperimentSpec,
        contract: ResearchContract,
    ) -> EvaluationResult:
        if (
            experiment.evaluator_id != self.metadata.evaluator_id
            or experiment.code_version != self.metadata.code_version
            or experiment.dataset_version != self.metadata.dataset_version
            or experiment.provenance != self.metadata.provenance
        ):
            return self._failure(experiment, "experiment_metadata_mismatch")
        dataset_loading_completed = False
        propagation_completed = False
        clustering_completed = False
        eligibility_evaluation_completed = False
        try:
            configuration = ICCADirectConfiguration.normalise(
                experiment.configuration,
                self.bindings,
            )
        except Exception as exc:
            # Configuration should normally have been rejected during planning. This
            # adapter guard protects direct callers and still performs no data access.
            return self._failure(
                experiment,
                "icca_evaluation_failed: CONFIGURATION_VALIDATION: "
                f"{type(exc).__name__}",
            )
        try:
            network = resolve_enum_alias(
                self.bindings.network_type, configuration.network
            )
            alignment = resolve_enum_alias(
                self.bindings.alignment_type, configuration.alignment
            )
            if self._cohort is None:
                self._cohort = self.bindings.load_cohort(
                    self.context.data_dir,
                    verbose=False,
                )
            dataset_loading_completed = True
            if self._paths is None:
                self._paths = self.bindings.harness_paths_factory(
                    self.context.workspace_dir
                )
            if self._propagation_cache is None:
                self._propagation_cache = self.bindings.propagation_cache_factory(
                    self._paths
                )
        except Exception as exc:
            return self._staged_failure(
                experiment,
                configuration,
                exc,
                ICCAEvaluationFailureStage.DATASET_LOADING,
                dataset_loading_completed=dataset_loading_completed,
                propagation_completed=propagation_completed,
                clustering_completed=clustering_completed,
                eligibility_evaluation_completed=eligibility_evaluation_completed,
            )
        try:
            propagated = self._propagation_cache.get(
                self._cohort.mutations,
                network,
                alignment,
                configuration.alpha,
            )
            propagation_completed = True
        except Exception as exc:
            return self._staged_failure(
                experiment,
                configuration,
                exc,
                ICCAEvaluationFailureStage.NETWORK_PROPAGATION,
                dataset_loading_completed=dataset_loading_completed,
                propagation_completed=propagation_completed,
                clustering_completed=clustering_completed,
                eligibility_evaluation_completed=eligibility_evaluation_completed,
            )
        try:
            config_evaluation = self.bindings.evaluate(
                propagated.matrix,
                propagated.patient_ids,
                self._cohort,
                k_values=[configuration.K],
                r=configuration.r,
                config={
                    "network": configuration.network,
                    "alignment": configuration.alignment,
                    "alpha": configuration.alpha,
                },
            )
        except Exception as exc:
            stage = classify_scientific_failure(exc)
            clustering_completed = (
                stage == ICCAEvaluationFailureStage.ELIGIBILITY_EVALUATION
            )
            return self._staged_failure(
                experiment,
                configuration,
                exc,
                stage,
                dataset_loading_completed=dataset_loading_completed,
                propagation_completed=propagation_completed,
                clustering_completed=clustering_completed,
                eligibility_evaluation_completed=eligibility_evaluation_completed,
            )
        clustering_completed = True
        try:
            if configuration.K not in config_evaluation.per_k:
                raise ValueError(
                    f"v2 evaluator did not return requested K={configuration.K}"
                )
            eligibility_evaluation_completed = True
        except Exception as exc:
            return self._staged_failure(
                experiment,
                configuration,
                exc,
                ICCAEvaluationFailureStage.ELIGIBILITY_EVALUATION,
                dataset_loading_completed=dataset_loading_completed,
                propagation_completed=propagation_completed,
                clustering_completed=clustering_completed,
                eligibility_evaluation_completed=eligibility_evaluation_completed,
            )
        try:
            score = float(
                self.bindings.stability_objective(
                    config_evaluation.per_k[configuration.K]
                )
            )
        except Exception as exc:
            return self._staged_failure(
                experiment,
                configuration,
                exc,
                ICCAEvaluationFailureStage.OBJECTIVE_CALCULATION,
                dataset_loading_completed=dataset_loading_completed,
                propagation_completed=propagation_completed,
                clustering_completed=clustering_completed,
                eligibility_evaluation_completed=eligibility_evaluation_completed,
            )
        try:
            result = self._map_evaluation_result(
                experiment,
                configuration,
                config_evaluation.per_k[configuration.K],
                contract,
                score=score,
            )
        except Exception as exc:
            return self._staged_failure(
                experiment,
                configuration,
                exc,
                ICCAEvaluationFailureStage.OBJECTIVE_CALCULATION,
                dataset_loading_completed=dataset_loading_completed,
                propagation_completed=propagation_completed,
                clustering_completed=clustering_completed,
                eligibility_evaluation_completed=eligibility_evaluation_completed,
            )
        try:
            write_artefact_bundle(
                self.context,
                experiment,
                result,
                self.dataset_manifest,
                self._evaluator_manifest(),
            )
        except Exception as exc:
            return self._staged_failure(
                experiment,
                configuration,
                exc,
                ICCAEvaluationFailureStage.ARTEFACT_WRITING,
                dataset_loading_completed=dataset_loading_completed,
                propagation_completed=propagation_completed,
                clustering_completed=clustering_completed,
                eligibility_evaluation_completed=eligibility_evaluation_completed,
                write_artefacts=False,
            )
        return result
