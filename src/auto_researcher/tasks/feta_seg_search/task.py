"""FeTA SegResNet bounded fold-0 DIRECT and Optuna search task."""

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
from auto_researcher.tasks.feta_seg.manifests import (
    DATASET_RELEASE,
    EXPECTED_MANIFEST_HASH,
    build_dataset_manifest,
    discover_pairs,
    inspect_subjects,
    manifest_hash,
)
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    FOLD_ID,
    SPLIT_ID,
)
from auto_researcher.tasks.feta_seg_search.configuration import (
    AUGMENTATION_STRENGTHS,
    CONFIGURATION_SCHEMA_VERSION,
    DICE_WEIGHT_BOUNDS,
    DROPOUT_BOUNDS,
    FIDELITY_LEVELS,
    LEARNING_RATE_BOUNDS,
    POSITIVE_NEGATIVE_RATIOS,
    WEIGHT_DECAY_BOUNDS,
    FeTASegSearchConfiguration,
    baseline_search_configuration,
    normalise_search_configuration,
)
from auto_researcher.tasks.feta_seg_search.evaluator import (
    EVALUATOR_ID,
    FeTASegSearchEvaluator,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_seg_search.verification import (
    FeTASegSearchVerificationPolicy,
)
from auto_researcher.tasks.models import (
    ArtefactPolicy,
    DatasetManifest,
    ExperimentMetadata,
    ReadinessCheck,
    ReadinessResult,
    TaskDescriptor,
    TaskRuntimeContext,
)


class FeTASegSearchTask:
    task_id = "feta_seg_search"
    task_version = "1.0"

    def descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="FeTA SegResNet Development Search",
            domain="fetal MRI segmentation",
            description=(
                "Bounded DIRECT and Optuna search on FeTA development fold 0 with "
                "the sealed hold-out excluded."
            ),
            supported_search_types=frozenset(
                {SearchType.DIRECT, SearchType.OPTUNA}
            ),
            evaluator_id=EVALUATOR_ID,
            verification_policy_id=FeTASegSearchVerificationPolicy.policy_id,
            configuration_schema_version=CONFIGURATION_SCHEMA_VERSION,
        )

    def readiness(self, context: TaskRuntimeContext) -> ReadinessResult:
        available = context.data_dir is not None and context.data_dir.exists()
        inventory = identity = dependencies = cuda = False
        try:
            import monai
            import nibabel
            import numpy
            import scipy  # type: ignore[import-untyped]
            import torch

            dependencies = all(
                module is not None for module in (monai, nibabel, numpy, scipy, torch)
            )
            cuda = bool(torch.cuda.is_available())
        except ImportError:
            pass
        if available:
            try:
                assert context.data_dir is not None
                pairs, _ = discover_pairs(context.data_dir)
                inventory = (
                    len(pairs) == 80
                    and sum(item[2] == "mial" for item in pairs.values()) == 40
                    and sum(item[2] == "irtk" for item in pairs.values()) == 40
                )
                subjects = inspect_subjects(context.data_dir, inspect_labels=False)
                identity = (
                    len(subjects) == 80
                    and manifest_hash(subjects) == EXPECTED_MANIFEST_HASH
                )
            except Exception:
                pass
        checks = (
            ReadinessCheck(
                code="feta_search_data_available",
                passed=available,
                message="A local FeTA root is required through TaskRuntimeContext.data_dir.",
            ),
            ReadinessCheck(
                code="feta_search_inventory_80_40_40",
                passed=inventory,
                message="Inventory must contain 80 paired subjects split 40 MIAL/40 IRTK.",
            ),
            ReadinessCheck(
                code="feta_search_dataset_identity_exact",
                passed=identity,
                message="The path-free manifest hash must match the audited FeTA export.",
            ),
            ReadinessCheck(
                code="feta_search_ml_dependencies_available",
                passed=dependencies,
                message="The pinned feta optional dependency set must be installed.",
            ),
            ReadinessCheck(
                code="feta_search_cuda_available",
                passed=cuda,
                message="FeTA candidate training requires a CUDA-capable PyTorch runtime.",
            ),
        )
        errors = tuple(item.code for item in checks if not item.passed)
        return ReadinessResult(ready=not errors, checks=checks, errors=errors)

    def validate_contract(self, contract: ResearchContract) -> None:
        if (contract.task_id, contract.task_version) != (
            self.task_id,
            self.task_version,
        ):
            raise ValueError("contract does not target feta_seg_search@1.0")
        if (
            contract.evaluator_id != EVALUATOR_ID
            or contract.primary_metric != "mean_subject_macro_dice"
            or contract.objective_version != "feta-fold0-search-macro-dice-v1"
            or contract.task_constraints_version != CONFIGURATION_SCHEMA_VERSION
        ):
            raise ValueError("feta_search_contract_identity_mismatch")
        if not contract.allowed_search_types.issubset(
            {SearchType.DIRECT, SearchType.OPTUNA}
        ):
            raise ValueError("feta_search_type_unsupported")
        expected = {
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_hash": EXPECTED_FOLD_HASH,
            "holdout_policy": "sealed-no-evaluation",
            "search_scope": "development-fold-0-only",
        }
        if any(contract.constraints.get(key) != value for key, value in expected.items()):
            raise ValueError("feta_search_contract_scientific_identity_mismatch")

    def normalise_configuration(
        self, configuration: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return normalise_search_configuration(configuration)

    def dataset_manifest(self, context: TaskRuntimeContext) -> DatasetManifest:
        manifest = build_dataset_manifest(context)
        return manifest.model_copy(update={"task_id": self.task_id})

    def experiment_metadata(self, context: TaskRuntimeContext) -> ExperimentMetadata:
        manifest = self.dataset_manifest(context)
        return ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(manifest.dataset_version),
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.REAL,
        )

    def create_evaluator(self, context: TaskRuntimeContext) -> FeTASegSearchEvaluator:
        manifest = self.dataset_manifest(context)
        metadata = ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(manifest.dataset_version),
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.REAL,
        )
        return FeTASegSearchEvaluator(context, metadata, manifest)

    def create_verification_policy(
        self, contract: ResearchContract
    ) -> FeTASegSearchVerificationPolicy:
        self.validate_contract(contract)
        return FeTASegSearchVerificationPolicy()

    def create_optuna_study_spec(
        self, contract: ResearchContract, request: SearchRequest
    ) -> OptunaStudySpec:
        self.validate_contract(contract)
        if request.search_type != SearchType.OPTUNA:
            raise ValueError("FeTA search Optuna study requires an OPTUNA request")
        proposed = dict(request.search_space)
        raw_fixed = proposed.get("fixed", {})
        if not isinstance(raw_fixed, dict):
            raise ValueError("FeTA search Optuna fixed section must be a mapping")
        hpo_names = {
            "learning_rate",
            "weight_decay",
            "dropout",
            "dice_weight",
            "positive_negative_ratio",
            "augmentation_strength",
        }
        if set(raw_fixed) - ({"fold", "maximum_epochs"} | hpo_names):
            raise ValueError("FeTA search Optuna fixed section contains unknown fields")
        if raw_fixed.get("fold", 0) != 0:
            raise ValueError("feta_search_fold_must_be_zero")
        fidelity = raw_fixed.get("maximum_epochs", 50)
        fixed_configuration = FeTASegSearchConfiguration(
            maximum_epochs=fidelity  # type: ignore[arg-type]
        ).model_dump(mode="json")
        fixed = {
            name: fixed_configuration[name]
            for name in (
                "fold",
                "maximum_epochs",
                "spatial_dims",
                "in_channels",
                "out_channels",
                "init_filters",
                "blocks_down",
                "blocks_up",
                "norm",
                "activation",
                "upsample_mode",
                "spacing_mm",
                "patch_size",
                "batch_size",
                "samples_per_volume",
                "ce_weight",
                "inference_overlap",
                "inference_blending",
                "sliding_window_batch_size",
                "seed",
            )
        }
        proposed["fixed"] = {
            **fixed,
            **{name: raw_fixed[name] for name in hpo_names if name in raw_fixed},
        }
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
                    name="augmentation_strength", choices=AUGMENTATION_STRENGTHS
                ),
            ),
            fixed_configuration=fixed,
            trial_budget=request.experiment_budget,
            seed=20260807,
            sampler="TPE",
            n_startup_trials=12,
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

    def artefact_policy(self) -> ArtefactPolicy:
        return ArtefactPolicy(
            allowed_artefact_types=frozenset(
                {
                    "experiment_spec",
                    "evaluation_result",
                    "dataset_manifest",
                    "evaluator_manifest",
                    "training_summary",
                    "checkpoint_reference",
                    "environment_manifest",
                    "study_spec",
                    "study_summary",
                    "trials_summary",
                    "selected_trial",
                }
            ),
            prohibited_artefact_types=frozenset(
                {
                    "raw_mri",
                    "raw_mask",
                    "raw_segmentation",
                    "prediction",
                    "holdout_data",
                    "holdout_prediction",
                    "holdout_metric",
                }
            ),
            contains_sensitive_data=True,
            retention_notes=(
                "Only aggregate and safe-ID fold-0 development metrics are published; "
                "MRI, masks, predictions, checkpoints and hold-out data stay outside git."
            ),
        )

    def create_agent_context(
        self,
        contract: ResearchContract,
        runtime_context: TaskRuntimeContext,
        search_capabilities: dict[SearchType, SearchCapability],
    ) -> TaskAgentContext:
        self.validate_contract(contract)
        return TaskAgentContext(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="FeTA SegResNet Development Search",
            domain="fetal MRI segmentation",
            task_description="Search six bounded SegResNet training axes on development fold 0.",
            safe_scientific_vocabulary=(
                "macro Dice",
                "SegResNet",
                "MIAL",
                "IRTK",
                "augmentation strength",
            ),
            primary_metric_description="Mean subject-level macro Dice over labels 1–7 on 14 fold-0 validation subjects.",
            scientific_constraint_summary=(
                "fold 0 only",
                "hold-out sealed",
                "fixed architecture and preprocessing",
                "six bounded HPO axes",
            ),
            dataset_summary={
                "dataset_release": DATASET_RELEASE,
                "training_subjects": 54,
                "validation_subjects": 14,
                "holdout_subjects_evaluated": 0,
                "contains_medical_images": True,
            },
            available_search_types=(SearchType.DIRECT, SearchType.OPTUNA),
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
            },
            fixed_scientific_context={
                "split_identity": SPLIT_ID,
                "split_hash": EXPECTED_SPLIT_HASH,
                "fold_identity": FOLD_ID,
                "fold_hash": EXPECTED_FOLD_HASH,
                "fold": 0,
            },
            task_limitations=(
                "No hold-out evaluation, architecture search or rung continuation.",
            ),
            safety_notes=(
                "Raw MRI, masks, subject rows and predictions are excluded from model context.",
            ),
        )


def default_feta_search_contract(
    *, maximum_experiments: int = 64
) -> ResearchContract:
    return ResearchContract(
        contract_id="feta-segresnet-fold0-search-contract",
        schema_version="1.0",
        task_id="feta_seg_search",
        task_version="1.0",
        objective_version="feta-fold0-search-macro-dice-v1",
        primary_metric="mean_subject_macro_dice",
        task_constraints_version=CONFIGURATION_SCHEMA_VERSION,
        question="Which bounded training configuration maximises FeTA fold-0 development macro Dice?",
        objective="maximise fold-0 development mean subject-level macro Dice",
        constraints={
            "dataset_release": DATASET_RELEASE,
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_identity": SPLIT_ID,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_identity": FOLD_ID,
            "fold_hash": EXPECTED_FOLD_HASH,
            "holdout_policy": "sealed-no-evaluation",
            "search_scope": "development-fold-0-only",
            "score_minimum": 0.0,
            "score_maximum": 1.0,
        },
        allowed_search_types=frozenset({SearchType.DIRECT, SearchType.OPTUNA}),
        evaluator_id=EVALUATOR_ID,
        verifier_id="deterministic-verifier",
        maximum_cycles=1,
        maximum_experiments=maximum_experiments,
        maximum_cost=1.0,
        requires_approval_for=frozenset(),
        provenance=ProvenanceKind.REAL,
    )


def default_feta_search_configuration(maximum_epochs: int = 50) -> dict[str, JsonValue]:
    return baseline_search_configuration(maximum_epochs)
