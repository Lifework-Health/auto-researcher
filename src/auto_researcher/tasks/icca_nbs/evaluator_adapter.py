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
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)


class ICCANBSEvaluatorAdapter:
    evaluator_id = "icca-nbs-v2-evaluator"
    version = "icca-adapter-v1"
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

    def _failure(self, experiment: ExperimentSpec, message: str) -> EvaluationResult:
        result = EvaluationResult(
            experiment_id=experiment.experiment_id,
            success=False,
            primary_score=None,
            metrics={},
            constraint_results={},
            artefact_references=artefact_references(
                self.context, experiment.experiment_id
            ),
            evaluator_version=self.version,
            provenance=self.metadata.provenance,
            error=message,
        )
        write_artefact_bundle(
            self.context,
            experiment,
            result,
            self.dataset_manifest,
            self._evaluator_manifest(),
        )
        return result

    def map_evaluation(
        self,
        experiment: ExperimentSpec,
        configuration: ICCADirectConfiguration,
        v2_result,
        contract: ResearchContract,
    ) -> EvaluationResult:
        score = float(self.bindings.stability_objective(v2_result))
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
        result = EvaluationResult(
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
        try:
            configuration = ICCADirectConfiguration.normalise(
                experiment.configuration,
                self.bindings,
            )
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
            if self._paths is None:
                self._paths = self.bindings.harness_paths_factory(
                    self.context.workspace_dir
                )
            if self._propagation_cache is None:
                self._propagation_cache = self.bindings.propagation_cache_factory(
                    self._paths
                )
            propagated = self._propagation_cache.get(
                self._cohort.mutations,
                network,
                alignment,
                configuration.alpha,
            )
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
            if configuration.K not in config_evaluation.per_k:
                raise ValueError(
                    f"v2 evaluator did not return requested K={configuration.K}"
                )
            return self.map_evaluation(
                experiment,
                configuration,
                config_evaluation.per_k[configuration.K],
                contract,
            )
        except Exception as exc:
            return self._failure(
                experiment,
                f"icca_evaluation_failed: {type(exc).__name__}",
            )
