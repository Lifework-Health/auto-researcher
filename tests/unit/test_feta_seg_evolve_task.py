from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_researcher.cli import _load_task_configuration
from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import SearchRequest
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.identity import source_hash
from auto_researcher.search.openevolve.mutation import (
    DeterministicMutationOperator,
    FakeModelMutationOperator,
)
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
from auto_researcher.search.openevolve.upstream import mutation_constraints
from auto_researcher.search.openevolve.validation import validate_candidate
from auto_researcher.tasks.feta_seg_evolve import (
    EvolveBaseConfiguration,
    FeTASegEvolveTask,
    FeTASegEvolvableComponent,
    TrainingPolicy,
    default_feta_evolve_contract,
)
from auto_researcher.tasks.feta_seg_evolve.configuration import (
    FeTASegEvolveConfiguration,
)
from auto_researcher.tasks.feta_seg_evolve.evaluator import (
    EVALUATOR_ID,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_seg_evolve.openevolve import COSINE_SOURCE
from auto_researcher.tasks.feta_seg_evolve.runner import policy_trace
from auto_researcher.tasks.feta_seg_evolve.training_policy import (
    AugmentationRecipe,
    DiceWeightPolicy,
    LearningRatePolicy,
)
from auto_researcher.tasks.feta_seg.manifests import (
    DATASET_RELEASE,
    EXPECTED_MANIFEST_HASH,
)
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
)
from auto_researcher.tasks.feta_seg_search.configuration import (
    CONFIGURATION_SCHEMA_VERSION,
    FeTASegSearchConfiguration,
    baseline_search_configuration,
    normalise_search_configuration,
)
from auto_researcher.tasks.feta_seg_search.continuation import (
    CONTINUATION_VERSION,
    candidate_trajectory_identity,
)
from auto_researcher.tasks.feta_seg_search.evaluator import (
    EVALUATOR_VERSION as SEARCH_EVALUATOR_VERSION,
)
from auto_researcher.tasks.feta_seg_search.runner import (
    RUNNER_VERSION as SEARCH_RUNNER_VERSION,
)
from auto_researcher.tasks.feta_seg_search.task import FeTASegSearchTask
from auto_researcher.tasks.models import ExperimentMetadata, TaskRuntimeContext
from auto_researcher.tasks.registry import default_task_registry


DATASET_VERSION = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"


def _request(configuration: dict | None = None, *, budget: int = 3) -> SearchRequest:
    return SearchRequest(
        request_id="feta-evolve-request",
        hypothesis_id="hypothesis",
        search_type=SearchType.OPENEVOLVE,
        target="mean_subject_macro_dice",
        search_space=configuration or _search_configuration(),
        experiment_budget=budget,
        rationale="Exercise bounded TrainingPolicy evolution.",
    )


def _search_configuration() -> dict:
    from auto_researcher.tasks.feta_seg_evolve import (
        default_feta_evolve_openevolve_configuration,
    )

    return default_feta_evolve_openevolve_configuration()


def _metadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version=evaluator_code_version(DATASET_VERSION),
        dataset_version=DATASET_VERSION,
        provenance=ProvenanceKind.REAL,
    )


def _component(
    base: EvolveBaseConfiguration | None = None, mode="pure"
) -> FeTASegEvolvableComponent:
    return FeTASegEvolvableComponent(base or EvolveBaseConfiguration(), mode)


def _backend(component=None, operator=None, workspace=None) -> OpenEvolveBackend:
    return OpenEvolveBackend(
        component or _component(),
        _metadata(),
        "deterministic-verifier-v1@feta-seg-evolve-evidence-policy-v2",
        operator or DeterministicMutationOperator(),
        LocalSandboxRunner(workspace),
    )


def test_sibling_task_is_registered_without_expanding_active_search_task():
    registry = default_task_registry()
    evolve = registry.get("feta_seg_evolve", "1.0")
    search = registry.get("feta_seg_search", "1.0")
    assert isinstance(evolve, FeTASegEvolveTask)
    assert evolve.descriptor().supported_search_types == {
        SearchType.DIRECT,
        SearchType.OPENEVOLVE,
    }
    assert isinstance(search, FeTASegSearchTask)
    assert search.descriptor().supported_search_types == {
        SearchType.DIRECT,
        SearchType.OPTUNA,
    }


def test_evolve_contract_locks_dataset_split_fold_and_holdout():
    contract = default_feta_evolve_contract()
    assert contract.constraints["dataset_manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert contract.constraints["split_hash"] == EXPECTED_SPLIT_HASH
    assert contract.constraints["fold_hash"] == EXPECTED_FOLD_HASH
    assert contract.constraints["holdout_policy"] == "sealed-no-evaluation"
    assert contract.constraints["search_scope"] == "development-fold-0-only"
    assert (
        contract.constraints["mutable_surface"]
        == "TrainingPolicy@feta-training-policy-v1-only"
    )
    assert (
        contract.constraints["scientific_feasibility_policy"]
        == "feta-evolve-scientific-feasibility-v1"
    )
    assert contract.constraints["maximum_empty_predictions"] == 0
    assert contract.constraints["minimum_per_tissue_dice"] == 0.5


@pytest.mark.parametrize(
    "payload",
    [
        {"learning_rate": {"family": "exponential"}},
        {"learning_rate": {"family": "constant", "end_multiplier": 0.5}},
        {"dice_weight": {"family": "constant", "start": 0.8, "end": 1.2}},
        {"positive_negative_ratio": "4:1"},
        {"architecture": "UNETR"},
        {"augmentation": {"flip_probability": 0.31}},
        {"learning_rate": {"warmup_fraction": 0.21}},
    ],
)
def test_training_policy_rejects_invalid_or_unknown_surface(payload):
    with pytest.raises(ValidationError):
        TrainingPolicy.model_validate(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_training_policy_rejects_non_finite_values(value):
    with pytest.raises(ValidationError):
        LearningRatePolicy(warmup_fraction=value)
    with pytest.raises(ValidationError):
        DiceWeightPolicy(start=value)
    with pytest.raises(ValidationError):
        AugmentationRecipe(scale_factor=value)


def test_policy_interpretation_is_bounded_at_early_middle_and_final_epochs():
    policy = TrainingPolicy(
        learning_rate=LearningRatePolicy(
            family="cosine", warmup_fraction=0.1, end_multiplier=0.05
        ),
        dice_weight=DiceWeightPolicy(family="linear", start=0.5, end=1.5),
        augmentation=AugmentationRecipe(
            flip_probability=0.1,
            intensity_probability=0.15,
            scale_factor=0.05,
            shift_offset=0.05,
        ),
        positive_negative_ratio="3:1",
    )
    rates = [policy.learning_rate_at(epoch, 25, 5e-4) for epoch in range(1, 26)]
    dice = [policy.dice_weight_at(epoch, 25) for epoch in range(1, 26)]
    assert all(3e-5 <= value <= 5e-4 and math.isfinite(value) for value in rates)
    assert all(0.5 <= value <= 1.5 and math.isfinite(value) for value in dice)
    assert dice[0] == 0.5
    assert dice[-1] == 1.5


def test_policy_trace_records_representative_epochs():
    component = _component()
    backend = _backend(component)
    contract = backend.create_search_contract(
        _request(), default_feta_evolve_contract()
    )
    seed = backend.seed_candidate(contract)
    validated = seed.model_copy(update={"validation_result": backend.validate(seed)})
    prepared = backend.prepare(validated, contract)
    experiment = component.candidate_to_experiment(
        seed,
        prepared,
        _request(),
        default_feta_evolve_contract(),
        _metadata(),
        run_id="policy-trace",
    )
    configuration = FeTASegEvolveConfiguration.model_validate(experiment.configuration)
    assert [row["epoch"] for row in policy_trace(configuration)] == [1, 13, 25]


def test_component_has_one_import_free_file_and_valid_seed_interface():
    component = _component()
    spec = component.component_spec()
    assert spec.allowed_files == ("candidate.py",)
    assert spec.allowed_imports == ()
    assert spec.allowed_dependencies == ()
    backend = _backend(component)
    contract = backend.create_search_contract(
        _request(), default_feta_evolve_contract()
    )
    seed = backend.seed_candidate(contract)
    result = validate_candidate(seed, spec)
    assert result.status.value == "VALID"


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef evolve(configuration):\n return {}\n",
        "def evolve(configuration, extra):\n return {}\n",
        "def evolve(configuration):\n configuration['architecture']='UNETR'\n return {}\n",
    ],
)
def test_candidate_import_file_access_and_interface_escape_fail_static_validation(
    source,
):
    backend = _backend()
    contract = backend.create_search_contract(
        _request(), default_feta_evolve_contract()
    )
    seed = backend.seed_candidate(contract)
    candidate = seed.model_copy(
        update={"source_payload": source, "source_hash": source_hash(source)}
    )
    assert backend.validate(candidate).status.value == "INVALID"


def test_candidate_cannot_add_architecture_or_preprocessing_fields(tmp_path):
    source = """def evolve(configuration):
    return {"policy_version": "feta-training-policy-v1", "architecture": "UNETR"}
"""
    backend = _backend(workspace=tmp_path)
    contract = backend.create_search_contract(
        _request(), default_feta_evolve_contract()
    )
    seed = backend.seed_candidate(contract)
    candidate = seed.model_copy(
        update={"source_payload": source, "source_hash": source_hash(source)}
    )
    validation = backend.validate(candidate)
    assert validation.status.value == "VALID"
    prepared = backend.sandbox_runner.prepare(
        candidate.model_copy(update={"validation_result": validation}),
        backend.component_spec,
        contract.sandbox_policy,
        backend.component.seed_configuration(),
    )
    assert prepared.execution_status.value == "COMPLETED"
    with pytest.raises(ValidationError):
        backend.component.candidate_to_experiment(
            candidate,
            prepared,
            _request(),
            default_feta_evolve_contract(),
            _metadata(),
            run_id="prohibited-field",
        )


def test_seed_candidate_prepares_and_converts_deterministically(tmp_path):
    component = _component()
    backend = _backend(component, workspace=tmp_path)
    contract = backend.create_search_contract(
        _request(), default_feta_evolve_contract()
    )
    seed = backend.seed_candidate(contract)
    validation = backend.validate(seed)
    assert validation.status.value == "VALID"
    prepared = backend.sandbox_runner.prepare(
        seed.model_copy(update={"validation_result": validation}),
        backend.component_spec,
        contract.sandbox_policy,
        component.seed_configuration(),
    )
    first = component.candidate_to_experiment(
        seed,
        prepared,
        _request(),
        default_feta_evolve_contract(),
        _metadata(),
        run_id="seed",
    )
    second = component.candidate_to_experiment(
        seed,
        prepared,
        _request(),
        default_feta_evolve_contract(),
        _metadata(),
        run_id="seed",
    )
    assert first == second
    configuration = FeTASegEvolveConfiguration.model_validate(first.configuration)
    assert configuration.seeding_mode == "pure"
    assert configuration.candidate_provenance.creation_provenance == "SEED"
    assert configuration.base_configuration.seed == 20260807


def test_fake_mutation_candidate_preserves_lineage_and_source_identity(tmp_path):
    class Client:
        def propose_mutation(self, request):
            assert "data_dir" not in json.dumps(request)
            return {"source": COSINE_SOURCE, "description": "bounded fake policy"}

    component = _component()
    backend = _backend(component, FakeModelMutationOperator(Client()), tmp_path)
    configuration = _search_configuration()
    configuration["openevolve"]["maximum_model_calls"] = 1
    search = backend.create_search_contract(
        _request(configuration), default_feta_evolve_contract()
    )
    seed = backend.seed_candidate(search)
    population = backend.initialise_population(search).model_copy(
        update={"active_population_candidate_ids": (seed.candidate_id,)}
    )
    reservation = backend.reserve_mutation(search, population, seed)
    candidate = backend.mutate_candidate(reservation, seed, search)
    assert candidate.parent_candidate_ids == (seed.candidate_id,)
    assert candidate.creation_provenance == "FAKE_MODEL"
    assert candidate.model_call_id is not None
    assert backend.validate(candidate).status.value == "VALID"


def test_optuna_base_configuration_is_runtime_injected_and_identity_bound():
    options = {
        "base_configuration": {
            "maximum_epochs": 50,
            "learning_rate": 3e-5,
            "weight_decay": 3e-4,
            "dropout": 0.4,
            "dice_weight": 1.5,
            "positive_negative_ratio": "2:1",
            "augmentation_strength": "light",
        }
    }
    component = FeTASegEvolveTask().create_evolvable_component(
        default_feta_evolve_contract(), TaskRuntimeContext(task_options=options)
    )
    assert component.seeding_mode == "optuna"
    assert component.base_configuration.learning_rate == 3e-5
    assert component.seed_policy.dice_weight.start == 1.5
    assert component.seed_policy.positive_negative_ratio == "2:1"
    assert payload_hash(component.base_configuration) != payload_hash(
        EvolveBaseConfiguration()
    )


def test_mutation_context_is_metadata_only_and_reaches_generic_constraints():
    spec = _component().component_spec()
    context = json.loads(json.dumps(spec.task_mutation_context))
    encoded = json.dumps(context, sort_keys=True).casefold()
    assert "subject_id" not in encoded
    assert context["data_boundary"] == (
        "Only the approved aggregate task metadata and bounded policy schema are exposed."
    )
    assert "/users/" not in encoded and "/home/" not in encoded
    constraints = mutation_constraints(spec)
    assert (
        constraints.parameter_schema["mutation_context"] == spec.task_mutation_context
    )


@pytest.mark.parametrize(
    "observation",
    (
        "subject 001 had low Dice",
        "case 7 used reconstruction X",
        "MRI scan quality varied",
        "prediction stored at /protected/result.json",
        "holdout metric was high",
    ),
)
def test_mutation_context_rejects_nonaggregate_hpo_observations(observation):
    with pytest.raises(ValueError, match="feta_evolve_hpo_observations_invalid"):
        FeTASegEvolvableComponent(
            EvolveBaseConfiguration(),
            "pure",
            task_options={"hpo_observations": [observation]},
        )


def test_generation_zero_counts_toward_candidate_budget():
    backend = _backend()
    configuration = _search_configuration()
    configuration["openevolve"]["maximum_generations"] = 1
    configuration["openevolve"]["maximum_candidate_evaluations"] = 2
    search = backend.create_search_contract(
        _request(configuration, budget=2),
        default_feta_evolve_contract(maximum_experiments=2),
    )
    assert search.maximum_candidate_evaluations == 2
    assert backend.seed_candidate(search).generation == 0
    configuration["openevolve"]["maximum_candidate_evaluations"] = 1
    with pytest.raises(ValueError, match="mutation_evaluation_budget_too_small"):
        backend.create_search_contract(
            _request(configuration, budget=1),
            default_feta_evolve_contract(maximum_experiments=1),
        )


def test_live_model_mutation_remains_fail_closed_for_mri_task():
    task = FeTASegEvolveTask()
    assert not hasattr(task, "live_mutation_dataset_class")


def test_agent_context_describes_attested_metadata_only_live_boundary():
    context = FeTASegEvolveTask().create_agent_context(
        default_feta_evolve_contract(),
        TaskRuntimeContext(),
        {},
    )
    limitations = " ".join(context.task_limitations)
    assert "No live-model mutation approval for MRI-backed tasks." not in limitations
    assert limitations == (
        "Live-model mutation requires a fresh attested metadata-only v2 approval; "
        "no MRI or evaluator data may cross the mutation-model boundary."
    )
    assert context.safety_notes == (
        "No MRI, masks, paths, subject rows, predictions, checkpoints or holdout "
        "information enter mutation context.",
    )


def test_examples_parse_and_build_real_search_contracts():
    root = Path(__file__).parents[2] / "examples" / "tasks" / "feta_seg_evolve"
    for name, expected_mode in (
        ("openevolve-deterministic-smoke.yaml", "pure"),
        ("openevolve-production-template.yaml", "pure"),
        ("openevolve-hybrid-template.yaml", "optuna"),
    ):
        search, runtime = _load_task_configuration(
            root / name, "feta_seg_evolve", "1.0"
        )
        options = runtime.get("options", {})
        component = FeTASegEvolveTask().create_evolvable_component(
            default_feta_evolve_contract(), TaskRuntimeContext(task_options=options)
        )
        backend = _backend(component)
        contract = backend.create_search_contract(
            _request(search), default_feta_evolve_contract()
        )
        assert component.seeding_mode == expected_mode
        assert contract.maximum_candidate_evaluations == 3
        assert contract.maximum_model_calls == 0
        assert contract.verifier_identity == (
            "deterministic-verifier-v1@feta-seg-evolve-evidence-policy-v2"
        )
        assert (
            contract.selection_policy.policy_id
            == "constraint-verification-objective-v2"
        )


def test_active_feta_search_scientific_and_continuation_identity_is_unchanged():
    assert CONFIGURATION_SCHEMA_VERSION == "feta-segresnet-search-configuration-v1"
    assert SEARCH_RUNNER_VERSION == "feta-fold0-search-runner-v2"
    assert SEARCH_EVALUATOR_VERSION == "feta-segresnet-search-evaluator-v3"
    assert CONTINUATION_VERSION == "feta-search-stateful-optimisation-continuation-v1"
    at_25 = FeTASegSearchConfiguration(maximum_epochs=25)
    at_50 = FeTASegSearchConfiguration(maximum_epochs=50)
    assert candidate_trajectory_identity(at_25) == candidate_trajectory_identity(at_50)
    assert normalise_search_configuration(baseline_search_configuration(25)) == {
        "fold": 0,
        "maximum_epochs": 25,
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "dropout": 0.2,
        "dice_weight": 1.0,
        "positive_negative_ratio": "1:1",
        "augmentation_strength": "baseline",
    }
