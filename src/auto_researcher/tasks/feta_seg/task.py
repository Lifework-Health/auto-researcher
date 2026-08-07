"""FeTA 2.1 locked-development SegResNet DIRECT task."""

from __future__ import annotations

from pydantic import JsonValue

from auto_researcher.agents.models import TaskAgentContext
from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.tasks.feta_seg.configuration import (
    FeTASegConfiguration,
    baseline_configuration,
)
from auto_researcher.tasks.feta_seg.evaluator import (
    EVALUATOR_ID,
    FeTASegEvaluator,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_seg.manifests import (
    DATASET_RELEASE,
    EXPECTED_MANIFEST_HASH,
    build_dataset_manifest,
    discover_pairs,
)
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
)
from auto_researcher.tasks.feta_seg.verification import FeTASegVerificationPolicy
from auto_researcher.tasks.models import (
    ArtefactPolicy,
    ExperimentMetadata,
    ReadinessCheck,
    ReadinessResult,
    TaskDescriptor,
    TaskRuntimeContext,
)
from auto_researcher.search.protocols import SearchCapability


class FeTASegTask:
    task_id = "feta_seg"
    task_version = "1.0"

    def descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="FeTA Locked SegResNet Baseline",
            domain="fetal MRI segmentation",
            description="Development-only five-fold FeTA 2.1 SegResNet evaluation with a sealed hold-out.",
            supported_search_types=frozenset({SearchType.DIRECT}),
            evaluator_id=EVALUATOR_ID,
            verification_policy_id=FeTASegVerificationPolicy.policy_id,
            configuration_schema_version="feta-segresnet-baseline-configuration-v1",
        )

    def readiness(self, context: TaskRuntimeContext) -> ReadinessResult:
        if context.task_options.get("mode") == "smoke" and context.data_dir is None:
            smoke_checks = (
                ReadinessCheck(
                    code="feta_generated_smoke",
                    passed=True,
                    message="Generated smoke mode is explicitly non-scientific.",
                ),
            )
            return ReadinessResult(
                ready=True, checks=smoke_checks, warnings=("not_scientific_baseline",)
            )
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
                data_dir = context.data_dir
                pairs, _ = discover_pairs(data_dir)
                inventory = (
                    len(pairs) == 80
                    and sum(item[2] == "mial" for item in pairs.values()) == 40
                    and sum(item[2] == "irtk" for item in pairs.values()) == 40
                )
                from auto_researcher.tasks.feta_seg.manifests import (
                    inspect_subjects,
                    manifest_hash,
                )

                subjects = inspect_subjects(data_dir, inspect_labels=False)
                identity = (
                    len(subjects) == 80
                    and manifest_hash(subjects) == EXPECTED_MANIFEST_HASH
                )
            except Exception:
                pass
        checks = (
            ReadinessCheck(
                code="feta_data_available",
                passed=available,
                message="A local FeTA root is required through TaskRuntimeContext.data_dir.",
            ),
            ReadinessCheck(
                code="feta_inventory_80_40_40",
                passed=inventory,
                message="Inventory must contain 80 paired subjects split 40 MIAL/40 IRTK.",
            ),
            ReadinessCheck(
                code="feta_dataset_identity_exact",
                passed=identity,
                message="The path-free manifest hash must match the audited FeTA export.",
            ),
            ReadinessCheck(
                code="feta_ml_dependencies_available",
                passed=dependencies,
                message="The pinned feta optional dependency set must be installed.",
            ),
            ReadinessCheck(
                code="feta_cuda_available",
                passed=cuda,
                message="The locked full baseline requires a CUDA-capable PyTorch runtime.",
            ),
        )
        errors = tuple(item.code for item in checks if not item.passed)
        return ReadinessResult(ready=not errors, checks=checks, errors=errors)

    def validate_contract(self, contract: ResearchContract) -> None:
        if (contract.task_id, contract.task_version) != (
            self.task_id,
            self.task_version,
        ):
            raise ValueError("contract does not target feta_seg@1.0")
        if (
            contract.evaluator_id != EVALUATOR_ID
            or contract.primary_metric != "mean_subject_macro_dice"
        ):
            raise ValueError("feta_contract_identity_mismatch")
        if contract.allowed_search_types != frozenset({SearchType.DIRECT}):
            raise ValueError("feta_direct_only")
        if contract.constraints.get("holdout_policy") != "sealed-no-evaluation":
            raise ValueError("feta_holdout_policy_missing")
        expected_constraints = {
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_hash": EXPECTED_FOLD_HASH,
        }
        if any(
            contract.constraints.get(key) != value
            for key, value in expected_constraints.items()
        ):
            raise ValueError("feta_contract_scientific_identity_mismatch")

    def normalise_configuration(
        self, configuration: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return FeTASegConfiguration.model_validate(
            configuration
        ).scientific_configuration()

    def dataset_manifest(self, context: TaskRuntimeContext):
        return build_dataset_manifest(context)

    def experiment_metadata(self, context: TaskRuntimeContext) -> ExperimentMetadata:
        manifest = self.dataset_manifest(context)
        return ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(manifest.dataset_version),
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.REAL,
        )

    def create_evaluator(self, context: TaskRuntimeContext) -> FeTASegEvaluator:
        manifest = self.dataset_manifest(context)
        metadata = ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(manifest.dataset_version),
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.REAL,
        )
        return FeTASegEvaluator(context, metadata, manifest)

    def create_verification_policy(
        self, contract: ResearchContract
    ) -> FeTASegVerificationPolicy:
        self.validate_contract(contract)
        return FeTASegVerificationPolicy()

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
                }
            ),
            prohibited_artefact_types=frozenset(
                {"raw_mri", "raw_segmentation", "holdout_prediction", "holdout_metric"}
            ),
            contains_sensitive_data=True,
            retention_notes=(
                "Aggregate and safe-ID per-development-subject metrics only; "
                "MRI, masks, predictions and checkpoints remain outside git."
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
            display_name="FeTA Locked SegResNet Baseline",
            domain="fetal MRI segmentation",
            task_description="Evaluate one fixed SegResNet on sealed-holdout FeTA development folds.",
            safe_scientific_vocabulary=("macro Dice", "SegResNet", "MIAL", "IRTK"),
            primary_metric_description="Mean subject-level macro Dice over tissue labels 1–7 across 68 OOF development predictions.",
            scientific_constraint_summary=(
                "DIRECT only",
                "hold-out sealed",
                "fixed architecture and preprocessing",
            ),
            dataset_summary={
                "dataset_release": DATASET_RELEASE,
                "development_subjects": 68,
                "holdout_subjects": 12,
                "contains_medical_images": True,
            },
            available_search_types=(SearchType.DIRECT,),
            direct_configuration_schema={"fixed_configuration": True},
            optuna_space_summary={},
            fixed_scientific_context={
                "split_identity": "feta-development-holdout-v1",
                "split_hash": EXPECTED_SPLIT_HASH,
                "fold_identity": "feta-dev-5fold-v1",
                "fold_hash": EXPECTED_FOLD_HASH,
            },
            task_limitations=(
                "No hold-out evaluation; no clinical claim; full result requires CUDA.",
            ),
            safety_notes=(
                "Raw MRI, masks, subject rows and predictions are excluded from model context.",
            ),
        )


def default_feta_contract() -> ResearchContract:
    return ResearchContract(
        contract_id="feta-segresnet-development-contract",
        schema_version="1.0",
        task_id="feta_seg",
        task_version="1.0",
        objective_version="feta-development-macro-dice-v1",
        primary_metric="mean_subject_macro_dice",
        task_constraints_version="feta-segresnet-baseline-configuration-v1",
        question="What is the locked SegResNet five-fold performance on the FeTA development partition?",
        objective="maximise locked-development five-fold mean subject-level macro Dice",
        constraints={
            "dataset_release": DATASET_RELEASE,
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_identity": "feta-development-holdout-v1",
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_identity": "feta-dev-5fold-v1",
            "fold_hash": EXPECTED_FOLD_HASH,
            "holdout_policy": "sealed-no-evaluation",
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


def default_feta_configuration() -> dict:
    return baseline_configuration()
