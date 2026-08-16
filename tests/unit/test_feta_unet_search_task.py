from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from auto_researcher.agents.context import AgentContextAssembler
from auto_researcher.agents.models import PriorResearchSummary
from auto_researcher.contracts.enums import (
    EvidenceStatus,
    EventType,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import (
    BudgetState,
    DecisionEvent,
    ResearchContract,
    SearchRequest,
)
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.mutation import DeterministicMutationOperator
from auto_researcher.search.openevolve.models import CandidateOutcome, CandidateStatus
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
from auto_researcher.tasks.feta_seg.manifests import (
    DATASET_RELEASE,
    EXPECTED_MANIFEST_HASH,
)
from auto_researcher.tasks.feta_unet_search import (
    FeTAUNetSearchConfiguration,
    FeTAUNetSearchTask,
    default_feta_unet_search_contract,
)
from auto_researcher.tasks.feta_unet_search.evaluator import (
    AUGMENTATION_ID,
    EVALUATOR_ID,
    LOSS_ID,
    OPTIMISER_ID,
    FeTAUNetSearchEvaluator,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_unet_search.openevolve import (
    default_openevolve_configuration,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    ReadinessResult,
    TaskRuntimeContext,
)
from auto_researcher.tasks.protocols import (
    CampaignDurationCapableTask,
    OpenEvolveCapableTask,
    OptunaCapableTask,
)
from auto_researcher.tasks.registry import default_task_registry


def _request(search_type: SearchType, search_space: dict, budget: int = 3):
    return SearchRequest(
        request_id=f"unet-{search_type.value.lower()}",
        hypothesis_id="hypothesis",
        search_type=search_type,
        target="mean_subject_macro_dice",
        search_space=search_space,
        experiment_budget=budget,
        rationale="Exercise the bounded BasicUNet campaign surface.",
    )


def _runtime(**options) -> TaskRuntimeContext:
    return TaskRuntimeContext(
        run_id="run",
        data_dir=Path("/tmp/data"),
        workspace_dir=Path("/tmp/workspace"),
        output_dir=Path("/tmp/output"),
        task_options=options,
    )


def test_registry_exposes_three_method_unet_campaign_task():
    task = default_task_registry().get("feta_unet_search", "1.0")
    assert isinstance(task, OptunaCapableTask)
    assert isinstance(task, OpenEvolveCapableTask)
    assert isinstance(task, CampaignDurationCapableTask)
    assert task.descriptor().supported_search_types == frozenset(SearchType)


def test_agent_context_exposes_direct_executable_parameter_names():
    task = FeTAUNetSearchTask()
    context = task.create_agent_context(
        default_feta_unet_search_contract(),
        _runtime(),
        {},
    )

    assert set(context.direct_configuration_schema) == {
        "maximum_epochs",
        "learning_rate",
        "weight_decay",
        "dropout",
        "dice_weight",
        "positive_negative_ratio",
        "augmentation_strength",
    }


def test_one_runtime_assembly_exposes_all_three_backends():
    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"

    class AssemblyTask(FeTAUNetSearchTask):
        def readiness(self, context):
            return ReadinessResult(ready=True, checks=())

        def dataset_manifest(self, context):
            return DatasetManifest(
                task_id=self.task_id,
                dataset_version=dataset_version,
                files=(),
                hashes={},
                loader_version="test",
                created_at=datetime(2026, 8, 15, tzinfo=UTC),
                metadata={"manifest_hash": EXPECTED_MANIFEST_HASH},
            )

        def create_evaluator(self, context):
            class Evaluator:
                evaluator_id = EVALUATOR_ID
                cost_per_experiment = 0.0

                def evaluate(self, experiment, contract):  # pragma: no cover
                    raise AssertionError("assembly test must not evaluate")

            return Evaluator()

    configuration = {
        "trial_budget": 2,
        "fixed": {"maximum_epochs": 5},
        **default_openevolve_configuration(candidate_evaluations=2),
    }
    dependencies = task_memory_dependencies(
        AssemblyTask(),
        _runtime(openevolve_fidelity=5),
        default_feta_unet_search_contract(),
        configuration,
        search_type=SearchType.OPTUNA,
    )
    assert {
        search_type
        for search_type, capability in dependencies.search_capabilities.items()
        if capability.available
    } == set(SearchType)


def test_search_configuration_keeps_architecture_fixed_and_training_bounded():
    candidate = FeTAUNetSearchConfiguration(
        maximum_epochs=25,
        learning_rate=2e-4,
        weight_decay=5e-5,
        dropout=0.2,
        dice_weight=1.2,
        positive_negative_ratio="2:1",
        augmentation_strength="strong",
    )
    assert candidate.features == (32, 32, 64, 128, 256, 32)
    with pytest.raises(ValidationError, match="learning_rate"):
        FeTAUNetSearchConfiguration(learning_rate=0.1)
    with pytest.raises(ValidationError, match="fixed_context"):
        FeTAUNetSearchConfiguration(patch_size=(64, 64, 64))


def test_search_evaluator_binds_variable_training_policy_identities():
    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"
    metadata = ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version=evaluator_code_version(dataset_version),
        dataset_version=dataset_version,
        provenance=ProvenanceKind.REAL,
    )
    manifest = DatasetManifest(
        task_id="feta_unet_search",
        dataset_version=dataset_version,
        files=(),
        hashes={},
        loader_version="test",
        created_at=datetime(2026, 8, 15, tzinfo=UTC),
        metadata={"manifest_hash": EXPECTED_MANIFEST_HASH},
    )
    evaluator = FeTAUNetSearchEvaluator(_runtime(), metadata, manifest)
    assert evaluator.augmentation_identity == AUGMENTATION_ID
    assert evaluator.loss_identity == LOSS_ID
    assert evaluator.optimiser_identity == OPTIMISER_ID


def test_optuna_space_has_six_axes_and_fixed_fidelity():
    task = FeTAUNetSearchTask()
    specification = task.create_optuna_study_spec(
        default_feta_unet_search_contract(),
        _request(
            SearchType.OPTUNA,
            {"fixed": {"maximum_epochs": 5}},
            budget=2,
        ),
    )
    assert specification.trial_budget == 2
    assert specification.fixed_configuration["maximum_epochs"] == 5
    assert {item.name for item in specification.parameters} == {
        "learning_rate",
        "weight_decay",
        "dropout",
        "dice_weight",
        "positive_negative_ratio",
        "augmentation_strength",
    }


def test_openevolve_seed_executes_to_a_bounded_unet_experiment():
    task = FeTAUNetSearchTask()
    contract = default_feta_unet_search_contract()
    component = task.create_evolvable_component(
        contract, _runtime(openevolve_fidelity=5)
    )
    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"
    metadata = ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version=evaluator_code_version(dataset_version),
        dataset_version=dataset_version,
        provenance=ProvenanceKind.REAL,
    )
    backend = OpenEvolveBackend(
        component,
        metadata,
        "deterministic-verifier-v1@feta-basic-unet-search-evidence-policy-v1",
        DeterministicMutationOperator(),
        LocalSandboxRunner(),
    )
    request = _request(
        SearchType.OPENEVOLVE,
        default_openevolve_configuration(candidate_evaluations=3),
    )
    search_contract = backend.create_search_contract(request, contract)
    seed = backend.seed_candidate(search_contract)
    assert backend.validate(seed).status.value == "VALID"
    preparation = backend.prepare(seed, search_contract)
    experiment = component.candidate_to_experiment(
        seed,
        preparation,
        request,
        contract,
        metadata,
        run_id="run",
    )
    assert experiment.configuration["maximum_epochs"] == 5
    assert experiment.configuration["profile"] == "development_baseline"


def test_openevolve_uses_verified_initial_incumbent_and_observations():
    task = FeTAUNetSearchTask()
    component = task.create_evolvable_component(
        default_feta_unet_search_contract(),
        _runtime(
            openevolve_fidelity=5,
            initial_incumbent_configuration={
                "maximum_epochs": 150,
                "learning_rate": 2e-4,
                "weight_decay": 6e-6,
                "dropout": 0.05,
                "dice_weight": 1.2,
                "positive_negative_ratio": "2:1",
                "augmentation_strength": "strong",
            },
            initial_campaign_observations=[
                "Verified fold-0 baseline mean macro Dice was 0.807986."
            ],
        ),
    )
    assert component.seed_configuration()["seed_training_policy"] == {
        "policy_version": "feta-basic-unet-training-policy-v1",
        "learning_rate": 2e-4,
        "weight_decay": 6e-6,
        "dropout": 0.05,
        "dice_weight": 1.2,
        "positive_negative_ratio": "2:1",
        "augmentation_strength": "strong",
    }
    assert component.component_spec().task_mutation_context[
        "aggregate_campaign_observations"
    ] == ["Verified fold-0 baseline mean macro Dice was 0.807986."]


def test_openevolve_rejects_nonaggregate_initial_observation():
    with pytest.raises(ValueError, match="feta_unet_campaign_observations_invalid"):
        FeTAUNetSearchTask().create_evolvable_component(
            default_feta_unet_search_contract(),
            _runtime(
                openevolve_fidelity=5,
                initial_campaign_observations=["subject 1 prediction was poor"],
            ),
        )


def test_verified_optuna_incumbent_seeds_openevolve_and_parent_feedback():
    task = FeTAUNetSearchTask()
    contract = default_feta_unet_search_contract()
    winning = FeTAUNetSearchConfiguration(
        maximum_epochs=25,
        learning_rate=2e-4,
        weight_decay=6e-6,
        dropout=0.05,
        dice_weight=1.2,
        positive_negative_ratio="2:1",
        augmentation_strength="strong",
    ).model_dump(mode="json")
    prior = PriorResearchSummary(
        hypothesis_reference="hypothesis-optuna",
        experiment_reference="experiment-optuna-winner",
        search_type=SearchType.OPTUNA,
        primary_score=0.81,
        evidence_status=EvidenceStatus.SUPPORTED,
        constraint_compliant=True,
        concise_verified_finding="Verified aggregate Optuna winner.",
        safe_configuration=winning,
    )
    request = task.enrich_search_request(
        _request(
            SearchType.OPENEVOLVE,
            default_openevolve_configuration(candidate_evaluations=2),
            budget=2,
        ),
        (prior,),
    )
    context = request.search_space["campaign_context"]
    assert context["incumbent_primary_score"] == 0.81
    assert context["incumbent_training_policy"]["learning_rate"] == 2e-4

    component = task.create_evolvable_component(
        contract,
        _runtime(openevolve_fidelity=5),
    )
    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"
    metadata = ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version=evaluator_code_version(dataset_version),
        dataset_version=dataset_version,
        provenance=ProvenanceKind.REAL,
    )
    backend = OpenEvolveBackend(
        component,
        metadata,
        "deterministic-verifier-v1@feta-basic-unet-search-evidence-policy-v1",
        DeterministicMutationOperator(),
        LocalSandboxRunner(),
    )
    search_contract = backend.create_search_contract(request, contract)
    seed = backend.seed_candidate(search_contract)
    preparation = backend.prepare(seed, search_contract)
    assert preparation.generated_configuration["learning_rate"] == 2e-4
    assert preparation.generated_configuration["augmentation_strength"] == "strong"

    outcome = CandidateOutcome(
        candidate_id=seed.candidate_id,
        source_hash=seed.source_hash,
        status=CandidateStatus.VERIFIED,
        objective_value=0.81,
        constraint_compliant=True,
        verified=True,
        evidence_status=EvidenceStatus.SUPPORTED,
        selection_outcome="selected",
        replacement_outcome="active",
    )
    population = backend.initialise_population(search_contract).model_copy(
        update={
            "outcomes": (outcome,),
            "active_population_candidate_ids": (seed.candidate_id,),
            "best_known_candidate_ids": (seed.candidate_id,),
        }
    )
    reservation = backend.reserve_mutation(
        search_contract,
        population,
        seed,
    )
    assert reservation.campaign_context["incumbent_primary_score"] == 0.81
    assert reservation.parent_feedback == {
        "objective_value": 0.81,
        "constraint_compliant": True,
        "verified": True,
        "evidence_status": "SUPPORTED",
        "aggregate_metrics": {},
    }


def test_campaign_duration_estimate_counts_candidate_epochs():
    task = FeTAUNetSearchTask()
    runtime = _runtime(campaign_seconds_per_epoch=10.0, openevolve_fidelity=25)
    optuna = _request(
        SearchType.OPTUNA,
        {"fixed": {"maximum_epochs": 5}},
        budget=3,
    )
    evolve = _request(
        SearchType.OPENEVOLVE,
        default_openevolve_configuration(candidate_evaluations=2),
        budget=2,
    )
    assert task.estimate_search_duration_seconds(optuna, runtime) == 150.0
    assert task.estimate_search_duration_seconds(evolve, runtime) == 500.0


def test_budget_deadline_survives_cycles_and_exhausts():
    started = datetime(2026, 8, 16, 8, tzinfo=UTC)
    deadline = started + timedelta(hours=20)
    budget = BudgetState(
        maximum_cycles=12,
        maximum_experiments=30,
        maximum_cost=20,
        started_at=started,
        deadline_at=deadline,
    )
    assert budget.before_cycle(started).exhausted is False
    exhausted = budget.before_cycle(deadline)
    assert exhausted.exhausted is True
    assert exhausted.exhaustion_reason == "campaign_deadline_reached"


def test_campaign_contract_template_is_exactly_twenty_hours():
    root = Path(__file__).resolve().parents[2] / "examples/tasks/feta_unet_search"
    contract = ResearchContract.model_validate(
        yaml.safe_load((root / "contract-20h.yaml").read_text())
    )
    assert contract.constraints["campaign_duration_seconds"] == 20 * 60 * 60
    assert contract.allowed_search_types == frozenset(SearchType)


def test_prior_result_context_retains_safe_configuration_and_learning_curve():
    store = SQLiteProvenanceStore()
    store.append_event(
        DecisionEvent(
            event_id="event-unet-result",
            run_id="run",
            cycle=1,
            event_type=EventType.EVIDENCE_VERIFIED,
            actor="verifier",
            input_references=("experiment-1",),
            output_references=(
                "evidence:SUPPORTED",
                "verified:true",
                "constraints:true",
                "score:0.51",
                "search_type:OPTUNA",
                "hypothesis:hypothesis-1",
            ),
            rationale="Verified bounded fold-0 result.",
            timestamp=datetime(2026, 8, 15, tzinfo=UTC),
            code_version="test",
            provenance=ProvenanceKind.REAL,
            safe_payload={
                "configuration": {"maximum_epochs": 25, "learning_rate": 0.0002},
                "aggregate_metrics": {
                    "validation_history": [
                        {"epoch": 5, "validation_score": 0.3},
                        {"epoch": 25, "validation_score": 0.51},
                    ]
                },
            },
        )
    )
    _, prior = AgentContextAssembler(store)._prior("run", 2)
    assert prior[0].safe_configuration["learning_rate"] == 0.0002
    assert prior[0].aggregate_metrics["validation_history"][-1]["epoch"] == 25
