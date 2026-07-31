"""Adapter that delegates all iCCA scientific work to installed auto_agent_v2."""

from __future__ import annotations

import math

from auto_researcher.contracts.enums import ProvenanceKind
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
)
from auto_researcher.tasks.artifacts import (
    ARTEFACT_BUNDLE_SCHEMA_VERSION,
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
from auto_researcher.tasks.icca_nbs.metric_policy import (
    ICCA_SCIENTIFIC_JSON_POLICY,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)
from auto_researcher.tasks.scientific_json import (
    SCIENTIFIC_JSON_ENCODING_VERSION,
    ScientificJsonNormalisationError,
    ScientificJsonNormalisationResult,
    normalise_scientific_json,
    require_valid_scientific_json,
)


class NonFinitePrimaryScoreError(ValueError):
    """Safe marker for a non-finite task objective."""


class ICCANBSEvaluatorAdapter:
    evaluator_id = "icca-nbs-v2-evaluator"
    version = "icca-adapter-v1.2"
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
            "result_encoding_version": SCIENTIFIC_JSON_ENCODING_VERSION,
            "artefact_bundle_schema_version": ARTEFACT_BUNDLE_SCHEMA_VERSION,
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
        normalisation_reason_code: str | None = None,
        unavailable_paths: tuple[str, ...] = (),
        rejected_paths: tuple[str, ...] = (),
    ) -> dict:
        diagnostics = {
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
        if normalisation_reason_code is not None:
            diagnostics.update(
                {
                    "normalisation_reason_code": normalisation_reason_code,
                    "unavailable_count": len(unavailable_paths),
                    "rejected_count": len(rejected_paths),
                    "unavailable_paths": list(unavailable_paths),
                    "rejected_paths": list(rejected_paths),
                }
            )
        if stage == ICCAEvaluationFailureStage.ARTEFACT_WRITING:
            diagnostics["artefact_persistence_failure_code"] = (
                "bundle_publication_failed"
            )
        return diagnostics

    def _failure(
        self,
        experiment: ExperimentSpec,
        message: str,
        *,
        diagnostics: dict | None = None,
        write_artefacts: bool = True,
    ) -> EvaluationResult:
        references = (
            artefact_references(self.context, experiment.experiment_id)
            if write_artefacts
            else ()
        )
        result = EvaluationResult(
            experiment_id=experiment.experiment_id,
            success=False,
            primary_score=None,
            metrics={"failure_diagnostics": diagnostics} if diagnostics else {},
            constraint_results={},
            artefact_references=references,
            evaluator_version=self.version,
            provenance=self.metadata.provenance,
            error=message,
        )
        if write_artefacts:
            try:
                write_artefact_bundle(
                    self.context,
                    experiment,
                    result,
                    self.dataset_manifest,
                    self._evaluator_manifest(),
                )
            except Exception as exc:
                persistence_diagnostics = dict(diagnostics or {})
                persistence_diagnostics.update(
                    {
                        "safe_exception_class": type(exc).__name__,
                        "failure_stage": ICCAEvaluationFailureStage.ARTEFACT_WRITING.value,
                        "evaluator_id": self.evaluator_id,
                        "evaluator_version": self.version,
                        "experiment_id": experiment.experiment_id,
                        "dataset_fingerprint": self._dataset_fingerprint(),
                        "artefact_persistence_failure_code": "bundle_publication_failed",
                    }
                )
                return EvaluationResult(
                    experiment_id=experiment.experiment_id,
                    success=False,
                    primary_score=None,
                    metrics={"failure_diagnostics": persistence_diagnostics},
                    constraint_results={},
                    artefact_references=(),
                    evaluator_version=self.version,
                    provenance=self.metadata.provenance,
                    error=(
                        "icca_evaluation_failed: ARTEFACT_WRITING: "
                        f"{type(exc).__name__}"
                    ),
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
        normalisation_reason_code = None
        unavailable_paths: tuple[str, ...] = ()
        rejected_paths: tuple[str, ...] = ()
        if isinstance(exc, ScientificJsonNormalisationError):
            normalisation_reason_code = exc.reason_code
            unavailable_paths = exc.result.unavailable_paths
            rejected_paths = exc.result.rejected_paths
        elif isinstance(exc, NonFinitePrimaryScoreError):
            normalisation_reason_code = "non_finite_primary_score"
        diagnostics = self._failure_diagnostics(
            experiment,
            configuration,
            exception_class=exception_class,
            stage=stage,
            dataset_loading_completed=dataset_loading_completed,
            propagation_completed=propagation_completed,
            clustering_completed=clustering_completed,
            eligibility_evaluation_completed=eligibility_evaluation_completed,
            normalisation_reason_code=normalisation_reason_code,
            unavailable_paths=unavailable_paths,
            rejected_paths=rejected_paths,
        )
        return self._failure(
            experiment,
            f"icca_evaluation_failed: {stage.value}: {exception_class}",
            diagnostics=diagnostics,
            write_artefacts=write_artefacts,
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
        if not math.isfinite(score):
            raise NonFinitePrimaryScoreError
        eligibility_result = normalise_scientific_json(
            v2_result.eligibility,
            root_path="eligibility",
        )
        eligibility = require_valid_scientific_json(
            eligibility_result,
            reason_code="invalid_eligibility_value",
        )
        constraint_results: dict[str, bool] = {}
        for gate in ("logrank_pass", "clinical_pass", "floors_pass"):
            value = eligibility.get(gate)
            if type(value) is not bool:
                invalid = ScientificJsonNormalisationResult(
                    value=eligibility,
                    unavailable_paths=eligibility_result.unavailable_paths,
                    rejected_paths=(f"eligibility.{gate}",),
                    category_counts=eligibility_result.category_counts,
                )
                raise ScientificJsonNormalisationError(
                    invalid,
                    reason_code="invalid_constraint_value",
                )
            constraint_results[gate] = value

        pac = float(json_safe(v2_result.selection_inputs["pac"]))
        provenance = ProvenanceKind(
            str(json_safe(getattr(v2_result, "provenance", "REAL")))
        )
        raw_metrics = {
            "primary_score": score,
            "stability": 1.0 - pac,
            "scientific": v2_result.metrics,
            "selection_inputs": v2_result.selection_inputs,
            "eligibility": eligibility,
            "per_cluster": v2_result.per_cluster,
            "configuration": configuration.model_dump(mode="json"),
            "evaluation_settings": {
                "r": configuration.r,
                "objective_version": contract.objective_version,
            },
        }
        normalisation = normalise_scientific_json(
            raw_metrics,
            policy=ICCA_SCIENTIFIC_JSON_POLICY,
        )
        metrics = require_valid_scientific_json(normalisation)
        metrics["metric_availability"] = {
            "unavailable_paths": list(normalisation.unavailable_paths),
            "unavailable_count": len(normalisation.unavailable_paths),
            "encoding": ICCA_SCIENTIFIC_JSON_POLICY.unavailable_encoding,
            "result_encoding_version": SCIENTIFIC_JSON_ENCODING_VERSION,
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
        completed = {
            "dataset_loading_completed": True,
            "propagation_completed": True,
            "clustering_completed": True,
            "eligibility_evaluation_completed": True,
        }
        try:
            score = float(self.bindings.stability_objective(v2_result))
            if not math.isfinite(score):
                raise NonFinitePrimaryScoreError
        except Exception as exc:
            return self._staged_failure(
                experiment,
                configuration,
                exc,
                ICCAEvaluationFailureStage.OBJECTIVE_CALCULATION,
                **completed,
            )
        try:
            result = self._map_evaluation_result(
                experiment,
                configuration,
                v2_result,
                contract,
                score=score,
            )
        except Exception as exc:
            return self._staged_failure(
                experiment,
                configuration,
                exc,
                ICCAEvaluationFailureStage.RESULT_NORMALISATION,
                **completed,
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
                write_artefacts=False,
                **completed,
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
            if not math.isfinite(score):
                raise NonFinitePrimaryScoreError
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
                ICCAEvaluationFailureStage.RESULT_NORMALISATION,
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
