"""Separately identified frozen FeTA BasicUNet DIRECT task."""

from __future__ import annotations

from pydantic import JsonValue

from auto_researcher.agents.models import TaskAgentContext
from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import ResearchContract
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
from auto_researcher.tasks.feta_unet_direct.configuration import (
    FeTAUNetDirectConfiguration,
    baseline_configuration,
)
from auto_researcher.tasks.feta_unet_direct.evaluator import (
    EVALUATOR_ID,
    INFERENCE_ID,
    LOSS_ID,
    OPTIMISER_ID,
    RESULT_ID,
    SCIENTIFIC_ID,
    FeTAUNetDirectEvaluator,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_unet_direct.identities import (
    BASELINE_RUNNER_ID,
    ENGINEERING_SMOKE_RUNNER_ID,
)
from auto_researcher.tasks.feta_unet_direct.model import (
    ARCHITECTURE_ID,
    TRAINABLE_PARAMETER_COUNT,
    create_basic_unet,
    trainable_parameter_count,
)
from auto_researcher.tasks.feta_unet_direct.verification import (
    FeTAUNetDirectVerificationPolicy,
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


class FeTAUNetDirectTask:
    task_id = "feta_unet_direct"
    task_version = "1.0"

    def descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="FeTA Frozen BasicUNet DIRECT Baseline",
            domain="fetal MRI segmentation",
            description=(
                "Real-data engineering smoke or development-only five-fold FeTA 2.1 "
                "BasicUNet evaluation with a sealed holdout."
            ),
            supported_search_types=frozenset({SearchType.DIRECT}),
            evaluator_id=EVALUATOR_ID,
            verification_policy_id=FeTAUNetDirectVerificationPolicy.policy_id,
            configuration_schema_version="feta-basic-unet-direct-configuration-v1",
        )

    def readiness(self, context: TaskRuntimeContext) -> ReadinessResult:
        explicit_paths = all(
            path is not None and path.is_absolute()
            for path in (context.data_dir, context.workspace_dir, context.output_dir)
        )
        protected_acknowledged = (
            context.task_options.get("protected_storage_acknowledged") is True
        )
        separated = False
        if context.workspace_dir is not None and context.output_dir is not None:
            workspace = context.workspace_dir.resolve()
            output = context.output_dir.resolve()
            separated = not (
                workspace == output
                or workspace.is_relative_to(output)
                or output.is_relative_to(workspace)
            )
        available = context.data_dir is not None and context.data_dir.exists()
        inventory = identity = dependencies = architecture = cuda = False
        try:
            import monai
            import nibabel
            import numpy
            import scipy  # type: ignore[import-untyped]
            import torch

            dependencies = all(
                module is not None for module in (monai, nibabel, numpy, scipy, torch)
            )
            model = create_basic_unet(FeTAUNetDirectConfiguration())
            architecture = trainable_parameter_count(model) == TRAINABLE_PARAMETER_COUNT
            del model
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
                code="feta_unet_runtime_paths_explicit",
                passed=explicit_paths,
                message="Data, protected workspace/cache and output paths must be absolute runtime values.",
            ),
            ReadinessCheck(
                code="feta_unet_protected_storage_acknowledged",
                passed=protected_acknowledged,
                message="Runtime options must explicitly acknowledge protected storage.",
            ),
            ReadinessCheck(
                code="feta_unet_workspace_output_separated",
                passed=separated,
                message="Protected training state and protected output must use non-overlapping roots.",
            ),
            ReadinessCheck(
                code="feta_unet_data_available",
                passed=available,
                message="The FeTA root must be supplied through runtime.data_dir.",
            ),
            ReadinessCheck(
                code="feta_unet_inventory_80_40_40",
                passed=inventory,
                message="Inventory must contain 80 paired subjects split 40 MIAL/40 IRTK.",
            ),
            ReadinessCheck(
                code="feta_unet_dataset_identity_exact",
                passed=identity,
                message="The path-free manifest hash must match the audited FeTA export.",
            ),
            ReadinessCheck(
                code="feta_unet_ml_dependencies_available",
                passed=dependencies,
                message="The pinned feta optional dependency set must be installed.",
            ),
            ReadinessCheck(
                code="feta_unet_architecture_exact",
                passed=architecture,
                message="The frozen BasicUNet must have exactly 5,749,608 trainable parameters.",
            ),
            ReadinessCheck(
                code="feta_unet_cuda_available",
                passed=cuda,
                message="Both real-data profiles require a CUDA-capable PyTorch runtime.",
            ),
        )
        errors = tuple(check.code for check in checks if not check.passed)
        return ReadinessResult(ready=not errors, checks=checks, errors=errors)

    def validate_contract(self, contract: ResearchContract) -> None:
        if (contract.task_id, contract.task_version) != (
            self.task_id,
            self.task_version,
        ):
            raise ValueError("contract does not target feta_unet_direct@1.0")
        if (
            contract.evaluator_id != EVALUATOR_ID
            or contract.primary_metric != "mean_subject_macro_dice"
        ):
            raise ValueError("feta_unet_contract_identity_mismatch")
        if contract.allowed_search_types != frozenset({SearchType.DIRECT}):
            raise ValueError("feta_unet_direct_only")
        expected = {
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_hash": EXPECTED_FOLD_HASH,
            "architecture_identity": ARCHITECTURE_ID,
            "architecture_trainable_parameters": TRAINABLE_PARAMETER_COUNT,
            "holdout_policy": "sealed-no-evaluation",
        }
        if any(
            contract.constraints.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("feta_unet_contract_scientific_identity_mismatch")

    def normalise_configuration(
        self, configuration: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        validated = FeTAUNetDirectConfiguration.model_validate(configuration)
        return {
            "profile": validated.profile,
            "maximum_epochs": validated.maximum_epochs,
            "validation_every": validated.validation_every,
            "fold_count": validated.fold_count,
        }

    def dataset_manifest(self, context: TaskRuntimeContext) -> DatasetManifest:
        source = build_dataset_manifest(context)
        metadata = source.metadata
        # Preserve the exact audited manifest hash and release while keeping
        # subject IDs and filenames inside protected storage only.
        safe_metadata = {
            key: metadata[key]
            for key in (
                "manifest_version",
                "dataset_release",
                "loader_version",
                "subject_count",
                "reconstruction_counts",
                "label_schema",
                "labels",
                "manifest_hash",
                "absolute_paths_in_identity",
            )
        }
        safe_metadata.update(
            {
                "subject_details_withheld": True,
                "contains_subject_identifiers": False,
            }
        )
        return DatasetManifest(
            task_id=self.task_id,
            dataset_version=source.dataset_version,
            files=(),
            hashes={},
            loader_version=source.loader_version,
            created_at=source.created_at,
            metadata=safe_metadata,
        )

    def experiment_metadata(self, context: TaskRuntimeContext) -> ExperimentMetadata:
        manifest = self.dataset_manifest(context)
        return ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(manifest.dataset_version),
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.REAL,
        )

    def create_evaluator(self, context: TaskRuntimeContext) -> FeTAUNetDirectEvaluator:
        manifest = self.dataset_manifest(context)
        metadata = ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(manifest.dataset_version),
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.REAL,
        )
        return FeTAUNetDirectEvaluator(context, metadata, manifest)

    def create_verification_policy(
        self, contract: ResearchContract
    ) -> FeTAUNetDirectVerificationPolicy:
        self.validate_contract(contract)
        return FeTAUNetDirectVerificationPolicy()

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
                    "raw_mri",
                    "raw_segmentation",
                    "patient_identifier",
                    "subject_identifier",
                    "subject_metric_row",
                    "prediction",
                    "checkpoint",
                    "holdout_prediction",
                    "holdout_metric",
                }
            ),
            contains_sensitive_data=False,
            retention_notes=(
                "Only aggregate, identifier-free development metrics and hashed relative "
                "checkpoint references are shareable; protected storage retains training state."
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
            display_name="FeTA Frozen BasicUNet DIRECT Baseline",
            domain="fetal MRI segmentation",
            task_description="Execute one frozen MONAI BasicUNet profile on FeTA development data.",
            safe_scientific_vocabulary=("macro Dice", "BasicUNet", "MIAL", "IRTK"),
            primary_metric_description=(
                "Mean subject-level macro Dice over tissue labels 1–7; the baseline "
                "aggregates 68 out-of-fold development predictions."
            ),
            scientific_constraint_summary=(
                "DIRECT only",
                "holdout sealed",
                "architecture and training configuration frozen",
            ),
            dataset_summary={
                "dataset_release": DATASET_RELEASE,
                "development_subjects": 68,
                "holdout_subjects": 12,
                "contains_medical_images": True,
            },
            available_search_types=(SearchType.DIRECT,),
            direct_configuration_schema={
                "profiles": ["engineering_smoke", "frozen_baseline"]
            },
            optuna_space_summary={},
            fixed_scientific_context={
                "scientific_identity": SCIENTIFIC_ID,
                "architecture_identity": ARCHITECTURE_ID,
                "split_identity": SPLIT_ID,
                "split_hash": EXPECTED_SPLIT_HASH,
                "fold_identity": FOLD_ID,
                "fold_hash": EXPECTED_FOLD_HASH,
                "loss_identity": LOSS_ID,
                "optimiser_identity": OPTIMISER_ID,
                "inference_identity": INFERENCE_ID,
                "smoke_runner_identity": ENGINEERING_SMOKE_RUNNER_ID,
                "baseline_runner_identity": BASELINE_RUNNER_ID,
                "result_identity": RESULT_ID,
            },
            task_limitations=(
                "No holdout evaluation; no clinical claim; both profiles require CUDA.",
            ),
            safety_notes=(
                "No MRI, masks, subject identifiers, predictions or checkpoints enter model context or shareable evidence.",
            ),
        )


def default_feta_unet_direct_contract() -> ResearchContract:
    return ResearchContract(
        contract_id="feta-basic-unet-direct-development-contract",
        schema_version="1.0",
        task_id="feta_unet_direct",
        task_version="1.0",
        objective_version=SCIENTIFIC_ID,
        primary_metric="mean_subject_macro_dice",
        task_constraints_version="feta-basic-unet-direct-configuration-v1",
        question="What is the frozen BasicUNet performance on the FeTA development partition?",
        objective="measure frozen-development mean subject-level macro Dice",
        constraints={
            "dataset_release": DATASET_RELEASE,
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_identity": SPLIT_ID,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_identity": FOLD_ID,
            "fold_hash": EXPECTED_FOLD_HASH,
            "holdout_policy": "sealed-no-evaluation",
            "architecture_identity": ARCHITECTURE_ID,
            "architecture_trainable_parameters": TRAINABLE_PARAMETER_COUNT,
            "score_minimum": 0.0,
            "score_maximum": 1.0,
        },
        allowed_search_types=frozenset({SearchType.DIRECT}),
        evaluator_id=EVALUATOR_ID,
        verifier_id="deterministic-verifier",
        maximum_cycles=1,
        maximum_experiments=1,
        maximum_cost=0.01,
        requires_approval_for=frozenset(),
        provenance=ProvenanceKind.REAL,
    )


def default_feta_unet_direct_configuration() -> dict:
    return baseline_configuration()
