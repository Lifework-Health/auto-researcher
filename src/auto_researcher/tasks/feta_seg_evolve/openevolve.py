"""One-file, metadata-only OpenEvolve surface for FeTA TrainingPolicy v1."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

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
from auto_researcher.tasks.feta_seg_evolve.configuration import (
    CandidateProvenance,
    EvolveBaseConfiguration,
    build_evolve_configuration,
)
from auto_researcher.tasks.feta_seg_evolve.evaluator import (
    EVALUATOR_ID,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_seg_evolve.training_policy import (
    TrainingPolicy,
    default_training_policy,
)
from auto_researcher.tasks.feta_seg.manifests import (
    DATASET_RELEASE,
    EXPECTED_MANIFEST_HASH,
)
from auto_researcher.tasks.models import ExperimentMetadata

COMPONENT_ID = "feta-segresnet-training-policy"
COMPONENT_VERSION = "1.0"

SEED_SOURCE = """def evolve(configuration):
    return configuration["seed_training_policy"]
"""

COSINE_SOURCE = """def evolve(configuration):
    return {
        "policy_version": "feta-training-policy-v1",
        "learning_rate": {"family": "cosine", "warmup_fraction": 0.1, "end_multiplier": 0.1},
        "dice_weight": {"family": "linear", "start": 0.8, "end": 1.2},
        "augmentation": {"flip_probability": 0.1, "intensity_probability": 0.1, "scale_factor": 0.05, "shift_offset": 0.05},
        "positive_negative_ratio": "2:1",
    }
"""

LINEAR_SOURCE = """def evolve(configuration):
    return {
        "policy_version": "feta-training-policy-v1",
        "learning_rate": {"family": "linear", "warmup_fraction": 0.0, "end_multiplier": 0.2},
        "dice_weight": {"family": "linear", "start": 1.0, "end": 1.3},
        "augmentation": {"flip_probability": 0.15, "intensity_probability": 0.1, "scale_factor": 0.05, "shift_offset": 0.05},
        "positive_negative_ratio": "1:1",
    }
"""


def _safe_hpo_observations(options: dict[str, Any]) -> tuple[str, ...]:
    raw = options.get("hpo_observations", ())
    if not isinstance(raw, (list, tuple)) or len(raw) > 12:
        raise ValueError("feta_evolve_hpo_observations_invalid")
    observations = tuple(raw)
    prohibited_tokens = (
        "/",
        "\\",
        "case ",
        "checkpoint",
        "holdout",
        "mask",
        "mri",
        "path",
        "patient",
        "prediction",
        "scan",
        "sub-",
        "subject",
        "voxel",
    )
    if any(
        not isinstance(item, str)
        or not item.strip()
        or len(item) > 500
        or any(token in item.casefold() for token in prohibited_tokens)
        for item in observations
    ):
        raise ValueError("feta_evolve_hpo_observations_invalid")
    return observations


class FeTASegEvolvableComponent:
    def __init__(
        self,
        base_configuration: EvolveBaseConfiguration,
        seeding_mode: Literal["pure", "optuna"],
        *,
        task_options: dict[str, Any] | None = None,
    ) -> None:
        self.base_configuration = base_configuration
        self.seeding_mode = seeding_mode
        self.hpo_observations = _safe_hpo_observations(task_options or {})
        self.seed_policy = default_training_policy(
            dice_weight=base_configuration.dice_weight,
            augmentation_strength=base_configuration.augmentation_strength,
            ratio=base_configuration.positive_negative_ratio,
        )

    def component_spec(self) -> EvolvableComponentSpec:
        safe_base = self.base_configuration.model_dump(mode="json")
        safe_context = {
            "objective": "maximise fold-0 mean subject-level macro Dice",
            "immutable_architecture": "3D SegResNet 32 filters, blocks 1-2-2-4 / 1-1-1",
            "immutable_preprocessing": "RAS, 0.5 mm, foreground crop, nonzero z-score, 128^3 patches",
            "base_configuration": safe_base,
            "legal_training_policy_schema": TrainingPolicy.model_json_schema(),
            "aggregate_hpo_observations": list(self.hpo_observations),
            "domain_guidance": [
                "Strong augmentation underperformed in the preceding aggregate HPO screen.",
                "Reconstruction-method robustness is important.",
                "Grey matter is a difficult tissue.",
            ],
            "metric_names": [
                "mean_subject_macro_dice",
                "reconstruction_gap",
                "per_tissue_dice",
            ],
            "data_boundary": "No MRI voxels, masks, paths, subject rows, predictions, checkpoints, holdout information, or evaluator internals are exposed.",
        }
        return EvolvableComponentSpec(
            component_id=COMPONENT_ID,
            component_version=COMPONENT_VERSION,
            mutable_file="candidate.py",
            allowed_files=("candidate.py",),
            entry_point="evolve",
            immutable_interface_contract=(
                "evolve(configuration: bounded metadata-only FeTA policy seed) -> "
                "TrainingPolicy@feta-training-policy-v1 JSON object"
            ),
            parameter_schema={
                "model": "FeTATrainingPolicySeedInput@1.0",
                "base_configuration": safe_base,
                "seed_training_policy": self.seed_policy.model_dump(mode="json"),
                "mutation_context": safe_context,
            },
            output_schema=TrainingPolicy.model_json_schema(),
            seed_source=SEED_SOURCE,
            deterministic_mutation_sources=(COSINE_SOURCE, LINEAR_SOURCE),
            maximum_source_bytes=8_192,
            task_mutation_context=safe_context,
        )

    def seed_configuration(self) -> dict:
        return {
            "base_configuration": self.base_configuration.model_dump(mode="json"),
            "seed_training_policy": self.seed_policy.model_dump(mode="json"),
        }

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
        del contract
        policy = TrainingPolicy.model_validate(
            dict(preparation.generated_configuration)
        )
        configuration = build_evolve_configuration(
            self.base_configuration,
            policy,
            seeding_mode=self.seeding_mode,
            candidate_provenance=CandidateProvenance(
                candidate_id=candidate.candidate_id,
                source_hash=candidate.source_hash,
                generation=candidate.generation,
                parent_candidate_ids=candidate.parent_candidate_ids,
                creation_provenance=candidate.creation_provenance,
            ),
        )
        digest = hashlib.sha256(
            f"{run_id}\x1f{request.request_id}\x1f{candidate.candidate_id}".encode()
        ).hexdigest()[:16]
        return ExperimentSpec(
            experiment_id=f"experiment-{digest}",
            hypothesis_id=request.hypothesis_id,
            search_request_id=request.request_id,
            configuration=configuration.model_dump(mode="json"),
            evaluator_id=metadata.evaluator_id,
            code_version=metadata.code_version,
            dataset_version=metadata.dataset_version,
            provenance=metadata.provenance,
        )


def default_feta_evolve_openevolve_configuration() -> dict[str, Any]:
    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"
    return {
        "openevolve": {
            "population_size": 1,
            "maximum_generations": 2,
            "maximum_candidate_evaluations": 3,
            "maximum_wall_time_seconds": 86_400.0,
            "maximum_model_calls": 0,
            "maximum_failed_candidates": 2,
            "maximum_consecutive_failures": 2,
            "maximum_artefact_bytes": 20_000_000,
            "random_seed": 20260810,
            "objective_direction": "MAXIMIZE",
            "objective_threshold": None,
            "sandbox_policy_id": "openevolve-sandbox-v1",
            "evaluator_identity": f"{EVALUATOR_ID}@{evaluator_code_version(dataset_version)}",
            "verifier_identity": "deterministic-verifier-v1@feta-seg-evolve-evidence-policy-v1",
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
