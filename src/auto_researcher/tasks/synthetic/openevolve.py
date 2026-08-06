"""Task-owned synthetic evolvable surface used by offline PR 6 demonstrations."""

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
from auto_researcher.tasks.models import ExperimentMetadata
from auto_researcher.tasks.synthetic.configuration import SyntheticConfiguration

SEED_SOURCE = """def evolve(configuration):
    return {"model_family": "linear", "complexity": 4, "learning_rate": 0.05}
"""
TREE_SOURCE = """def evolve(configuration):
    return {"model_family": "tree", "complexity": 4, "learning_rate": 0.05}
"""
NEURAL_SOURCE = """def evolve(configuration):
    return {"model_family": "neural", "complexity": 4, "learning_rate": 0.05}
"""


class SyntheticEvolvableComponent:
    def component_spec(self) -> EvolvableComponentSpec:
        return EvolvableComponentSpec(
            component_id="synthetic-scoring-transformation",
            component_version="1.0",
            mutable_file="candidate.py",
            allowed_files=("candidate.py",),
            entry_point="evolve",
            immutable_interface_contract="evolve(configuration: JSON object) -> SyntheticConfiguration JSON object",
            parameter_schema={
                "model_family": ["linear", "tree", "neural"],
                "complexity": {"minimum": 1, "maximum": 10},
                "learning_rate": {"exclusive_minimum": 0, "maximum": 1},
            },
            output_schema={"model": "SyntheticConfiguration@1.0"},
            seed_source=SEED_SOURCE,
            deterministic_mutation_sources=(TREE_SOURCE, NEURAL_SOURCE, NEURAL_SOURCE),
            maximum_source_bytes=4_096,
            task_mutation_context={"purpose": "offline known-improvement fixture"},
        )

    def seed_configuration(self) -> dict:
        return {"model_family": "linear", "complexity": 4, "learning_rate": 0.05}

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
        configuration = SyntheticConfiguration.model_validate(
            preparation.generated_configuration
        ).model_dump(mode="json")
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


def default_synthetic_openevolve_configuration() -> dict:
    return {
        "openevolve": {
            "population_size": 1,
            "maximum_generations": 3,
            "maximum_wall_time_seconds": 30.0,
            "maximum_model_calls": 0,
            "maximum_failed_candidates": 3,
            "maximum_consecutive_failures": 2,
            "maximum_artefact_bytes": 2_000_000,
            "random_seed": 20260805,
            "objective_direction": "MAXIMIZE",
            "objective_threshold": 0.88,
            "sandbox_policy_id": "openevolve-sandbox-v1",
            "evaluator_identity": "synthetic-evaluator@synthetic-task-1.0+scientific-json-v1+experiment-bundle-v2",
            "verifier_identity": "deterministic-verifier-v1@synthetic-policy-v1",
            "candidate_cpu_time_seconds": 2,
            "candidate_wall_time_seconds": 3.0,
            "candidate_memory_bytes": 268435456,
            "candidate_process_limit": 1,
            "candidate_output_bytes": 64000,
            "candidate_log_bytes": 8000,
            "candidate_file_count_limit": 8,
            "candidate_workspace_bytes": 1048576,
            "candidate_file_size_bytes": 64000,
        }
    }
