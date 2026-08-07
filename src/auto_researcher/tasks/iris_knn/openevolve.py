"""Narrow data-free OpenEvolve component for Iris configuration proposals."""

from __future__ import annotations

import hashlib

from auto_researcher.contracts.models import (
    ExperimentSpec,
    ResearchContract,
    SearchRequest,
)
from auto_researcher.search.openevolve.models import (
    CandidatePreparationResult,
    EvolvableComponentSpec,
    OpenEvolveCandidate,
)
from auto_researcher.tasks.iris_knn.configuration import (
    baseline_configuration,
    configuration_schema,
    normalise_iris_configuration,
)
from auto_researcher.tasks.iris_knn.evaluator import EVALUATOR_CODE_VERSION
from auto_researcher.tasks.models import ExperimentMetadata

SEED_SOURCE = """def evolve(configuration):
    return {"feature_weights": [1.0, 1.0, 1.0, 1.0], "k": 3, "distance_power": 2}
"""
PETAL_WEIGHTED_SOURCE = """def evolve(configuration):
    return {"feature_weights": [0.2, 0.2, 4.0, 4.0], "k": 5, "distance_power": 2}
"""
TREE_LIKE_SOURCE = """def evolve(configuration):
    return {"feature_weights": [0.1, 0.1, 4.0, 4.0], "k": 7, "distance_power": 1}
"""


class IrisKNNEvolvableComponent:
    def component_spec(self) -> EvolvableComponentSpec:
        return EvolvableComponentSpec(
            component_id="iris-weighted-knn-configuration",
            component_version="1.0",
            mutable_file="candidate.py",
            allowed_files=("candidate.py",),
            entry_point="evolve",
            immutable_interface_contract="evolve(configuration: bounded JSON object) -> IrisKNNConfiguration JSON object",
            parameter_schema=configuration_schema(),
            output_schema={"model": "IrisKNNConfiguration@1.0"},
            seed_source=SEED_SOURCE,
            deterministic_mutation_sources=(PETAL_WEIGHTED_SOURCE, TREE_LIKE_SOURCE),
            maximum_source_bytes=4_096,
            task_mutation_context={
                "benchmark": "Iris species classification",
                "feature_names": [
                    "sepal_length_cm",
                    "sepal_width_cm",
                    "petal_length_cm",
                    "petal_width_cm",
                ],
                "objective": "maximise mean five-fold balanced accuracy",
                "baseline_configuration": baseline_configuration(),
                "data_boundary": "No observations, row labels, folds, predictions, or confusion matrices are supplied.",
            },
        )

    def seed_configuration(self) -> dict:
        return baseline_configuration()

    def candidate_to_experiment(
        self,
        candidate: OpenEvolveCandidate,
        preparation: CandidatePreparationResult,
        request: SearchRequest,
        contract: ResearchContract,
        metadata: ExperimentMetadata,
        *,
        run_id: str,
    ) -> ExperimentSpec:
        configuration = normalise_iris_configuration(
            dict(preparation.generated_configuration)
        )
        digest = hashlib.sha256(
            f"{run_id}\x1f{request.request_id}\x1f{candidate.candidate_id}".encode()
        ).hexdigest()[:16]
        return ExperimentSpec(
            experiment_id=f"experiment-{digest}",
            hypothesis_id=request.hypothesis_id,
            search_request_id=request.request_id,
            configuration=configuration,
            evaluator_id=metadata.evaluator_id,
            code_version=metadata.code_version,
            dataset_version=metadata.dataset_version,
            provenance=metadata.provenance,
        )


def default_iris_openevolve_configuration() -> dict:
    return {
        "openevolve": {
            "population_size": 1,
            "maximum_generations": 2,
            "maximum_wall_time_seconds": 30.0,
            "maximum_model_calls": 0,
            "maximum_failed_candidates": 2,
            "maximum_consecutive_failures": 2,
            "maximum_artefact_bytes": 2_000_000,
            "random_seed": 20260807,
            "objective_direction": "MAXIMIZE",
            "objective_threshold": None,
            "sandbox_policy_id": "openevolve-sandbox-v1",
            "evaluator_identity": f"iris-weighted-knn-evaluator@{EVALUATOR_CODE_VERSION}",
            "verifier_identity": "deterministic-verifier-v1@iris-knn-evidence-policy-v1",
            "candidate_cpu_time_seconds": 2,
            "candidate_wall_time_seconds": 3.0,
            "candidate_memory_bytes": 268_435_456,
            "candidate_process_limit": 1,
            "candidate_output_bytes": 64_000,
            "candidate_log_bytes": 8_000,
            "candidate_file_count_limit": 8,
            "candidate_workspace_bytes": 1_048_576,
            "candidate_file_size_bytes": 64_000,
        }
    }
