"""Planner-ready BasicUNet fold-0 development campaign task."""

from __future__ import annotations

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
from auto_researcher.tasks.feta_unet_search.openevolve import (
    FeTAUNetEvolvableComponent,
    default_openevolve_configuration,
)
from auto_researcher.tasks.feta_seg.manifests import (
    DATASET_RELEASE,
    EXPECTED_MANIFEST_HASH,
)
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    FOLD_ID,
    SPLIT_ID,
)
from auto_researcher.tasks.feta_unet_direct.model import (
    ARCHITECTURE_ID,
    TRAINABLE_PARAMETER_COUNT,
)
from auto_researcher.tasks.feta_unet_direct.task import FeTAUNetDirectTask
from auto_researcher.tasks.feta_unet_search.configuration import (
    AUGMENTATION_STRENGTHS,
    CANDIDATE_CONFIGURATION_FIELDS,
    CONFIGURATION_SCHEMA_VERSION,
    DICE_WEIGHT_BOUNDS,
    DROPOUT_BOUNDS,
    FIDELITY_LEVELS,
    LEARNING_RATE_BOUNDS,
    POSITIVE_NEGATIVE_RATIOS,
    WEIGHT_DECAY_BOUNDS,
    baseline_search_configuration,
    normalise_search_configuration,
)
from auto_researcher.tasks.feta_unet_search.evaluator import (
    EVALUATOR_ID,
    OPTIMISER_ID,
    SCIENTIFIC_ID,
    FeTAUNetSearchEvaluator,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_unet_search.verification import (
    FeTAUNetSearchVerificationPolicy,
)
from auto_researcher.tasks.models import (
    ArtefactPolicy,
    ExperimentMetadata,
    TaskDescriptor,
    TaskRuntimeContext,
)


class FeTAUNetSearchTask(FeTAUNetDirectTask):
    task_id = "feta_unet_search"
    task_version = "1.0"

    def descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="FeTA BasicUNet Development Search",
            domain="fetal MRI segmentation",
            description=(
                "Planner-driven DIRECT, Optuna and OpenEvolve training-policy "
                "search on FeTA development fold 0 with the holdout sealed."
            ),
            supported_search_types=frozenset(
                {SearchType.DIRECT, SearchType.OPTUNA, SearchType.OPENEVOLVE}
            ),
            evaluator_id=EVALUATOR_ID,
            verification_policy_id=FeTAUNetSearchVerificationPolicy.policy_id,
            configuration_schema_version=CONFIGURATION_SCHEMA_VERSION,
        )

    def validate_contract(self, contract: ResearchContract) -> None:
        if (contract.task_id, contract.task_version) != (
            self.task_id,
            self.task_version,
        ):
            raise ValueError("contract does not target feta_unet_search@1.0")
        if (
            contract.evaluator_id != EVALUATOR_ID
            or contract.primary_metric != "mean_subject_macro_dice"
            or contract.objective_version != SCIENTIFIC_ID
            or contract.task_constraints_version != CONFIGURATION_SCHEMA_VERSION
        ):
            raise ValueError("feta_unet_search_contract_identity_mismatch")
        if not contract.allowed_search_types.issubset(
            {SearchType.DIRECT, SearchType.OPTUNA, SearchType.OPENEVOLVE}
        ):
            raise ValueError("feta_unet_search_type_unsupported")
        expected = {
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_hash": EXPECTED_FOLD_HASH,
            "architecture_identity": ARCHITECTURE_ID,
            "architecture_trainable_parameters": TRAINABLE_PARAMETER_COUNT,
            "holdout_policy": "sealed-no-evaluation",
            "search_scope": "development-fold-0-only",
        }
        if any(
            contract.constraints.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("feta_unet_search_scientific_identity_mismatch")

    def normalise_configuration(
        self, configuration: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return normalise_search_configuration(configuration)

    def experiment_metadata(self, context: TaskRuntimeContext) -> ExperimentMetadata:
        manifest = self.dataset_manifest(context)
        return ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(manifest.dataset_version),
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.REAL,
        )

    def create_evaluator(self, context: TaskRuntimeContext) -> FeTAUNetSearchEvaluator:
        manifest = self.dataset_manifest(context)
        metadata = self.experiment_metadata(context)
        return FeTAUNetSearchEvaluator(context, metadata, manifest)

    def create_verification_policy(
        self, contract: ResearchContract
    ) -> FeTAUNetSearchVerificationPolicy:
        self.validate_contract(contract)
        return FeTAUNetSearchVerificationPolicy()

    def create_optuna_study_spec(
        self, contract: ResearchContract, request: SearchRequest
    ) -> OptunaStudySpec:
        self.validate_contract(contract)
        if request.search_type != SearchType.OPTUNA:
            raise ValueError("feta_unet_search_requires_optuna_request")
        proposed = dict(request.search_space)
        proposed.pop("openevolve", None)
        raw_fixed = proposed.get("fixed", {})
        if not isinstance(raw_fixed, dict):
            raise ValueError("feta_unet_search_fixed_must_be_mapping")
        if set(raw_fixed) - set(CANDIDATE_CONFIGURATION_FIELDS):
            raise ValueError("feta_unet_search_fixed_contains_unknown_fields")
        fidelity = raw_fixed.get("maximum_epochs", 25)
        if isinstance(fidelity, bool) or not isinstance(fidelity, int):
            raise ValueError("feta_unet_search_fidelity_invalid")
        fixed = baseline_search_configuration(fidelity)
        registered = OptunaStudySpec(
            schema_version="1.0",
            task_id=self.task_id,
            task_version=self.task_version,
            search_space_version=CONFIGURATION_SCHEMA_VERSION,
            direction=OptimisationDirection.MAXIMIZE,
            parameters=(
                FloatParameterSpec(
                    name="learning_rate",
                    low=LEARNING_RATE_BOUNDS[0],
                    high=LEARNING_RATE_BOUNDS[1],
                    log=True,
                ),
                FloatParameterSpec(
                    name="weight_decay",
                    low=WEIGHT_DECAY_BOUNDS[0],
                    high=WEIGHT_DECAY_BOUNDS[1],
                    log=True,
                ),
                FloatParameterSpec(
                    name="dropout", low=DROPOUT_BOUNDS[0], high=DROPOUT_BOUNDS[1]
                ),
                FloatParameterSpec(
                    name="dice_weight",
                    low=DICE_WEIGHT_BOUNDS[0],
                    high=DICE_WEIGHT_BOUNDS[1],
                ),
                CategoricalParameterSpec(
                    name="positive_negative_ratio",
                    choices=POSITIVE_NEGATIVE_RATIOS,
                ),
                CategoricalParameterSpec(
                    name="augmentation_strength",
                    choices=AUGMENTATION_STRENGTHS,
                ),
            ),
            fixed_configuration={
                key: value
                for key, value in fixed.items()
                if key
                not in {
                    "learning_rate",
                    "weight_decay",
                    "dropout",
                    "dice_weight",
                    "positive_negative_ratio",
                    "augmentation_strength",
                }
            },
            trial_budget=request.experiment_budget,
            seed=20260807,
            sampler="TPE",
            n_startup_trials=min(6, request.experiment_budget),
            objective_metric=contract.primary_metric,
            study_metadata={
                "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
                "split_hash": EXPECTED_SPLIT_HASH,
                "fold_hash": EXPECTED_FOLD_HASH,
                "search_scope": "development-fold-0-only",
            },
        )
        return narrow_study_spec(
            registered,
            proposed,
            request_experiment_budget=request.experiment_budget,
        )

    def create_evolvable_component(
        self, contract: ResearchContract, runtime_context: TaskRuntimeContext
    ) -> FeTAUNetEvolvableComponent:
        self.validate_contract(contract)
        fidelity = runtime_context.task_options.get("openevolve_fidelity", 25)
        if isinstance(fidelity, bool) or not isinstance(fidelity, int):
            raise ValueError("feta_unet_openevolve_fidelity_invalid")
        return FeTAUNetEvolvableComponent(maximum_epochs=fidelity)

    def estimate_search_duration_seconds(
        self,
        request: SearchRequest,
        runtime_context: TaskRuntimeContext,
    ) -> float:
        raw_seconds_per_epoch = runtime_context.task_options.get(
            "campaign_seconds_per_epoch", 120.0
        )
        if (
            isinstance(raw_seconds_per_epoch, bool)
            or not isinstance(raw_seconds_per_epoch, (int, float))
            or raw_seconds_per_epoch <= 0
        ):
            raise ValueError("feta_unet_campaign_epoch_duration_invalid")
        if request.search_type == SearchType.OPENEVOLVE:
            fidelity = runtime_context.task_options.get("openevolve_fidelity", 25)
            candidates = request.experiment_budget
        elif request.search_type == SearchType.OPTUNA:
            fixed = request.search_space.get("fixed", {})
            fidelity = (
                fixed.get("maximum_epochs", 25) if isinstance(fixed, dict) else 25
            )
            candidates = request.experiment_budget
        else:
            fidelity = request.search_space.get("maximum_epochs", 25)
            candidates = 1
        if (
            isinstance(fidelity, bool)
            or not isinstance(fidelity, int)
            or fidelity not in FIDELITY_LEVELS
        ):
            raise ValueError("feta_unet_campaign_fidelity_invalid")
        return float(raw_seconds_per_epoch) * fidelity * candidates

    def artefact_policy(self) -> ArtefactPolicy:
        baseline = super().artefact_policy()
        return baseline.model_copy(
            update={
                "allowed_artefact_types": baseline.allowed_artefact_types
                | {
                    "training_summary",
                    "checkpoint_reference",
                    "study_spec",
                    "study_summary",
                    "trials_summary",
                    "selected_trial",
                }
            }
        )

    def create_agent_context(
        self,
        contract: ResearchContract,
        runtime_context: TaskRuntimeContext,
        search_capabilities: dict[SearchType, SearchCapability],
    ) -> TaskAgentContext:
        del search_capabilities
        self.validate_contract(contract)
        return TaskAgentContext(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="FeTA BasicUNet Development Search",
            domain="fetal MRI segmentation",
            task_description=(
                "Improve the fixed MONAI BasicUNet through bounded fold-0 "
                "training-policy experiments."
            ),
            safe_scientific_vocabulary=(
                "macro Dice",
                "BasicUNet",
                "learning rate",
                "augmentation strength",
                "training fidelity",
            ),
            primary_metric_description=(
                "Mean subject-level macro Dice over labels 1-7 on 14 fold-0 "
                "validation subjects."
            ),
            scientific_constraint_summary=(
                "fold 0 only",
                "holdout sealed",
                "fixed BasicUNet architecture and preprocessing",
                "bounded training-policy search",
            ),
            dataset_summary={
                "dataset_release": DATASET_RELEASE,
                "training_subjects": 54,
                "validation_subjects": 14,
                "holdout_subjects_evaluated": 0,
                "contains_medical_images": True,
            },
            available_search_types=(
                SearchType.DIRECT,
                SearchType.OPTUNA,
                SearchType.OPENEVOLVE,
            ),
            direct_configuration_schema={
                "fidelity_levels": list(FIDELITY_LEVELS),
                "baseline": baseline_search_configuration(),
            },
            optuna_space_summary={
                "learning_rate": list(LEARNING_RATE_BOUNDS),
                "weight_decay": list(WEIGHT_DECAY_BOUNDS),
                "dropout": list(DROPOUT_BOUNDS),
                "dice_weight": list(DICE_WEIGHT_BOUNDS),
                "positive_negative_ratio": list(POSITIVE_NEGATIVE_RATIOS),
                "augmentation_strength": list(AUGMENTATION_STRENGTHS),
                "fidelity_levels": list(FIDELITY_LEVELS),
            },
            openevolve_space_summary={
                **default_openevolve_configuration(candidate_evaluations=3),
                "mutable_policy": {
                    "learning_rate": list(LEARNING_RATE_BOUNDS),
                    "weight_decay": list(WEIGHT_DECAY_BOUNDS),
                    "dropout": list(DROPOUT_BOUNDS),
                    "dice_weight": list(DICE_WEIGHT_BOUNDS),
                    "positive_negative_ratio": list(POSITIVE_NEGATIVE_RATIOS),
                    "augmentation_strength": list(AUGMENTATION_STRENGTHS),
                },
                "fidelity": runtime_context.task_options.get("openevolve_fidelity", 25),
            },
            fixed_scientific_context={
                "architecture_identity": ARCHITECTURE_ID,
                "split_identity": SPLIT_ID,
                "split_hash": EXPECTED_SPLIT_HASH,
                "fold_identity": FOLD_ID,
                "fold_hash": EXPECTED_FOLD_HASH,
                "fold": 0,
                "optimiser_family": OPTIMISER_ID,
                "campaign_seconds_per_epoch": runtime_context.task_options.get(
                    "campaign_seconds_per_epoch", 120.0
                ),
                "campaign_finalisation_reserve_seconds": runtime_context.task_options.get(
                    "campaign_finalisation_reserve_seconds", 1800.0
                ),
            },
            task_limitations=(
                "No holdout evaluation, architecture mutation, multiple folds or multiple GPUs.",
            ),
            safety_notes=(
                "Raw MRI, masks, subject rows, predictions and checkpoint bytes are excluded from model context.",
            ),
        )


def default_feta_unet_search_contract(
    *, maximum_cycles: int = 12, maximum_experiments: int = 30
) -> ResearchContract:
    return ResearchContract(
        contract_id="feta-basic-unet-fold0-campaign-contract",
        schema_version="1.0",
        task_id="feta_unet_search",
        task_version="1.0",
        objective_version=SCIENTIFIC_ID,
        primary_metric="mean_subject_macro_dice",
        task_constraints_version=CONFIGURATION_SCHEMA_VERSION,
        question="Which bounded BasicUNet training policy improves fold-0 development macro Dice?",
        objective="maximise BasicUNet fold-0 development mean subject-level macro Dice",
        constraints={
            "dataset_release": DATASET_RELEASE,
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_identity": SPLIT_ID,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_identity": FOLD_ID,
            "fold_hash": EXPECTED_FOLD_HASH,
            "holdout_policy": "sealed-no-evaluation",
            "search_scope": "development-fold-0-only",
            "architecture_identity": ARCHITECTURE_ID,
            "architecture_trainable_parameters": TRAINABLE_PARAMETER_COUNT,
            "score_minimum": 0.0,
            "score_maximum": 1.0,
            "campaign_duration_seconds": 72_000,
            "campaign_finalisation_reserve_seconds": 1_800,
        },
        allowed_search_types=frozenset(
            {SearchType.DIRECT, SearchType.OPTUNA, SearchType.OPENEVOLVE}
        ),
        evaluator_id=EVALUATOR_ID,
        verifier_id="deterministic-verifier",
        maximum_cycles=maximum_cycles,
        maximum_experiments=maximum_experiments,
        maximum_cost=20.0,
        requires_approval_for=frozenset(),
        provenance=ProvenanceKind.REAL,
    )


def default_feta_unet_search_configuration(
    maximum_epochs: int = 25,
) -> dict[str, JsonValue]:
    return baseline_search_configuration(maximum_epochs)
