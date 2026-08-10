"""Sibling task for bounded FeTA OpenEvolve TrainingPolicy search."""

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
)
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    FOLD_ID,
    SPLIT_ID,
)
from auto_researcher.tasks.feta_seg_evolve.configuration import (
    EVOLVE_CONFIGURATION_VERSION,
    FeTASegEvolveConfiguration,
    base_configuration_from_runtime,
)
from auto_researcher.tasks.feta_seg_evolve.evaluator import (
    EVALUATOR_ID,
    FeTASegEvolveEvaluator,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_seg_evolve.openevolve import FeTASegEvolvableComponent
from auto_researcher.tasks.feta_seg_evolve.verification import (
    FeTASegEvolveVerificationPolicy,
)
from auto_researcher.tasks.feta_seg_search.task import FeTASegSearchTask
from auto_researcher.tasks.models import (
    ArtefactPolicy,
    DatasetManifest,
    ExperimentMetadata,
    TaskDescriptor,
    TaskRuntimeContext,
)


class FeTASegEvolveTask:
    task_id = "feta_seg_evolve"
    task_version = "1.0"

    def descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="FeTA SegResNet TrainingPolicy Evolution",
            domain="fetal MRI segmentation",
            description="Bounded metadata-only TrainingPolicy evolution on FeTA development fold 0.",
            supported_search_types=frozenset(
                {SearchType.DIRECT, SearchType.OPENEVOLVE}
            ),
            evaluator_id=EVALUATOR_ID,
            verification_policy_id=FeTASegEvolveVerificationPolicy.policy_id,
            configuration_schema_version=EVOLVE_CONFIGURATION_VERSION,
        )

    def readiness(self, context: TaskRuntimeContext):
        return FeTASegSearchTask().readiness(context)

    def validate_contract(self, contract: ResearchContract) -> None:
        if (contract.task_id, contract.task_version) != (
            self.task_id,
            self.task_version,
        ):
            raise ValueError("contract does not target feta_seg_evolve@1.0")
        if (
            contract.evaluator_id != EVALUATOR_ID
            or contract.primary_metric != "mean_subject_macro_dice"
            or contract.objective_version != "feta-fold0-training-policy-macro-dice-v1"
            or contract.task_constraints_version != EVOLVE_CONFIGURATION_VERSION
        ):
            raise ValueError("feta_evolve_contract_identity_mismatch")
        if not contract.allowed_search_types.issubset(
            {SearchType.DIRECT, SearchType.OPENEVOLVE}
        ):
            raise ValueError("feta_evolve_search_type_unsupported")
        expected = {
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_hash": EXPECTED_FOLD_HASH,
            "holdout_policy": "sealed-no-evaluation",
            "search_scope": "development-fold-0-only",
            "mutable_surface": "TrainingPolicy@feta-training-policy-v1-only",
        }
        if any(
            contract.constraints.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("feta_evolve_contract_scientific_identity_mismatch")

    def normalise_configuration(
        self, configuration: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return FeTASegEvolveConfiguration.model_validate(configuration).model_dump(
            mode="json"
        )

    def dataset_manifest(self, context: TaskRuntimeContext) -> DatasetManifest:
        return build_dataset_manifest(context).model_copy(
            update={"task_id": self.task_id}
        )

    def experiment_metadata(self, context: TaskRuntimeContext) -> ExperimentMetadata:
        manifest = self.dataset_manifest(context)
        return ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(manifest.dataset_version),
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.REAL,
        )

    def create_evaluator(self, context: TaskRuntimeContext) -> FeTASegEvolveEvaluator:
        manifest = self.dataset_manifest(context)
        metadata = ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(manifest.dataset_version),
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.REAL,
        )
        return FeTASegEvolveEvaluator(context, metadata, manifest)

    def create_verification_policy(
        self, contract: ResearchContract
    ) -> FeTASegEvolveVerificationPolicy:
        self.validate_contract(contract)
        return FeTASegEvolveVerificationPolicy()

    def create_evolvable_component(
        self, contract: ResearchContract, runtime_context: TaskRuntimeContext
    ) -> FeTASegEvolvableComponent:
        self.validate_contract(contract)
        base, mode = base_configuration_from_runtime(dict(runtime_context.task_options))
        return FeTASegEvolvableComponent(
            base,
            mode,
            task_options=dict(runtime_context.task_options),
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
                    "openevolve_search",
                    "openevolve_candidate",
                    "openevolve_lineage",
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
            retention_notes="Only bounded source, policy identities, lineage and aggregate fold-0 metrics are publishable.",
        )

    def create_agent_context(
        self,
        contract: ResearchContract,
        runtime_context: TaskRuntimeContext,
        search_capabilities: dict[SearchType, SearchCapability],
    ) -> TaskAgentContext:
        del runtime_context, search_capabilities
        self.validate_contract(contract)
        return TaskAgentContext(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="FeTA SegResNet TrainingPolicy Evolution",
            domain="fetal MRI segmentation",
            task_description="Evolve one bounded host-interpreted TrainingPolicy without data access.",
            safe_scientific_vocabulary=(
                "macro Dice",
                "TrainingPolicy",
                "SegResNet",
                "MIAL",
                "IRTK",
            ),
            primary_metric_description="Mean subject-level macro Dice on 14 fold-0 validation subjects.",
            scientific_constraint_summary=(
                "fold 0 only",
                "holdout sealed",
                "fixed architecture",
                "TrainingPolicy v1 only",
            ),
            dataset_summary={
                "dataset_release": DATASET_RELEASE,
                "training_subjects": 54,
                "validation_subjects": 14,
                "holdout_subjects_evaluated": 0,
                "contains_medical_images": True,
            },
            available_search_types=(SearchType.DIRECT, SearchType.OPENEVOLVE),
            direct_configuration_schema={
                "configuration_version": EVOLVE_CONFIGURATION_VERSION
            },
            optuna_space_summary={},
            fixed_scientific_context={
                "split_identity": SPLIT_ID,
                "split_hash": EXPECTED_SPLIT_HASH,
                "fold_identity": FOLD_ID,
                "fold_hash": EXPECTED_FOLD_HASH,
                "fold": 0,
            },
            task_limitations=("No live-model mutation approval for MRI-backed tasks.",),
            safety_notes=(
                "No MRI, masks, paths, subject rows, predictions, checkpoints or holdout information enter mutation context.",
            ),
        )


def default_feta_evolve_contract(*, maximum_experiments: int = 3) -> ResearchContract:
    return ResearchContract(
        contract_id="feta-segresnet-fold0-training-policy-contract",
        schema_version="1.0",
        task_id="feta_seg_evolve",
        task_version="1.0",
        objective_version="feta-fold0-training-policy-macro-dice-v1",
        primary_metric="mean_subject_macro_dice",
        task_constraints_version=EVOLVE_CONFIGURATION_VERSION,
        question="Which bounded TrainingPolicy improves FeTA fold-0 macro Dice?",
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
            "mutable_surface": "TrainingPolicy@feta-training-policy-v1-only",
            "score_minimum": 0.0,
            "score_maximum": 1.0,
        },
        allowed_search_types=frozenset({SearchType.DIRECT, SearchType.OPENEVOLVE}),
        evaluator_id=EVALUATOR_ID,
        verifier_id="deterministic-verifier",
        maximum_cycles=1,
        maximum_experiments=maximum_experiments,
        maximum_cost=1.0,
        requires_approval_for=frozenset(),
        provenance=ProvenanceKind.REAL,
    )
