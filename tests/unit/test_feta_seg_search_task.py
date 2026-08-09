from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from auto_researcher.cli import _load_task_configuration, _load_yaml
from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import ExperimentSpec, SearchRequest
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.direct import DirectSearchBackend
from auto_researcher.tasks.feta_seg import FeTASegTask
from auto_researcher.tasks.feta_seg.manifests import EXPECTED_MANIFEST_HASH
from auto_researcher.tasks.feta_seg.metrics import LABEL_NAMES, aggregate_subject_metrics
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    FOLD_ID,
    SPLIT_ID,
)
from auto_researcher.tasks.feta_seg_search import (
    FeTASegSearchConfiguration,
    FeTASegSearchTask,
    baseline_search_configuration,
    default_feta_search_contract,
    validation_epochs,
)
from auto_researcher.tasks.feta_seg_search.evaluator import (
    EVALUATOR_ID,
    FeTASegSearchEvaluator,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_seg_search.transforms import (
    augmentation_policy,
    positive_negative_counts,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)
from auto_researcher.tasks.registry import default_task_registry


def _request(*, maximum_epochs: int = 50) -> SearchRequest:
    return SearchRequest(
        request_id="feta-search-request",
        hypothesis_id="hypothesis",
        search_type=SearchType.OPTUNA,
        target="mean_subject_macro_dice",
        search_space={
            "trial_budget": 48,
            "fixed": {"fold": 0, "maximum_epochs": maximum_epochs},
        },
        experiment_budget=48,
        rationale="Exercise the registered bounded space.",
    )


def _metrics(configuration: FeTASegSearchConfiguration) -> dict:
    rows = []
    for index in range(14):
        per_class = {
            str(label): {
                "label_name": LABEL_NAMES[label],
                "dice": 0.6,
                "hd95_mm": 2.0,
                "volume_similarity": 0.8,
                "euler_distance": 0,
                "empty_prediction": False,
            }
            for label in range(1, 8)
        }
        rows.append(
            {
                "subject_id": f"safe-{index:02d}",
                "reconstruction_method": "mial" if index < 7 else "irtk",
                "fold": 0,
                "per_class": per_class,
                "macro_dice": 0.6,
                "macro_hd95_mm": 2.0,
                "macro_volume_similarity": 0.8,
                "macro_euler_distance": 0.0,
                "empty_prediction_count": 0,
            }
        )
    result = aggregate_subject_metrics(rows)
    result.update(
        {
            "best_epoch": 25,
            "validation_score": 0.6,
            "training_duration_seconds": 1.0,
            "total_duration_seconds": 2.0,
            "peak_gpu_memory_bytes": 1024,
            "validation_epochs": list(configuration.validation_epochs()),
            "training_subject_count": 54,
            "validation_subject_count": 14,
            "holdout_subjects_evaluated": 0,
            "fold": 0,
            "configuration_identity": payload_hash(configuration),
            "checkpoint_reference": {
                "fold": 0,
                "relative_path": "checkpoints/best.pt",
                "size_bytes": 1,
                "sha256": "a" * 64,
                "best_epoch": 25,
                "validation_score": 0.6,
            },
            "environment": {"gpu": "fixture"},
            "environment_identity": payload_hash({"gpu": "fixture"}),
            "valid_prediction_labels": list(range(8)),
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_identity": SPLIT_ID,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_identity": FOLD_ID,
            "fold_hash": EXPECTED_FOLD_HASH,
        }
    )
    return result


def _evaluator(tmp_path: Path, runner):
    dataset_version = f"feta-2.1-export-80+{EXPECTED_MANIFEST_HASH}"
    manifest = DatasetManifest(
        task_id="feta_seg_search",
        dataset_version=dataset_version,
        files=(),
        hashes={},
        loader_version="fixture",
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
        metadata={"manifest_hash": EXPECTED_MANIFEST_HASH},
    )
    metadata = ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version=evaluator_code_version(dataset_version),
        dataset_version=dataset_version,
        provenance=ProvenanceKind.REAL,
    )
    context = TaskRuntimeContext(workspace_dir=tmp_path)
    configuration = FeTASegSearchConfiguration()
    experiment = ExperimentSpec(
        experiment_id="candidate",
        hypothesis_id="hypothesis",
        search_request_id="request",
        configuration=FeTASegSearchTask().normalise_configuration(
            configuration.scientific_configuration()
        ),
        evaluator_id=metadata.evaluator_id,
        code_version=metadata.code_version,
        dataset_version=metadata.dataset_version,
        provenance=metadata.provenance,
    )
    return (
        FeTASegSearchEvaluator(
            context, metadata, manifest, search_runner=runner
        ),
        experiment,
        configuration,
    )


def test_registry_keeps_baseline_and_adds_search_task():
    registry = default_task_registry()
    assert isinstance(registry.get("feta_seg", "1.0"), FeTASegTask)
    assert isinstance(registry.get("feta_seg_search", "1.0"), FeTASegSearchTask)
    assert FeTASegTask().descriptor().supported_search_types == {SearchType.DIRECT}
    assert FeTASegSearchTask().descriptor().supported_search_types == {
        SearchType.DIRECT,
        SearchType.OPTUNA,
    }


def test_checked_in_search_examples_bind_runtime_data_and_optuna_controls():
    examples = (
        Path(__file__).resolve().parents[2] / "examples" / "tasks" / "feta_seg_search"
    )
    direct_path = examples / "direct-50.yaml"
    optuna_path = examples / "optuna-50.yaml"

    for path in (direct_path, optuna_path):
        payload = _load_yaml(path)
        assert payload["task"] == {"id": "feta_seg_search", "version": "1.0"}
        assert payload["runtime"]["data_dir"] == "/absolute/path/to/feta"
        assert "--data-dir" not in path.read_text(encoding="utf-8")

    direct, direct_runtime = _load_task_configuration(
        direct_path, "feta_seg_search", "1.0"
    )
    optuna, optuna_runtime = _load_task_configuration(
        optuna_path, "feta_seg_search", "1.0"
    )
    assert direct["maximum_epochs"] == 50
    assert direct_runtime["data_dir"] == "/absolute/path/to/feta"
    assert optuna_runtime["data_dir"] == "/absolute/path/to/feta"
    assert optuna["seed"] == 20260807
    assert optuna["n_startup_trials"] == 12


@pytest.mark.parametrize(
    "values",
    [
        {
            "learning_rate": 3e-5,
            "weight_decay": 1e-6,
            "dropout": 0.0,
            "dice_weight": 0.5,
        },
        {
            "learning_rate": 5e-4,
            "weight_decay": 3e-4,
            "dropout": 0.4,
            "dice_weight": 1.5,
        },
    ],
)
def test_configuration_accepts_registered_bounds(values):
    assert FeTASegSearchConfiguration(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 2e-5),
        ("weight_decay", 4e-4),
        ("dropout", -0.1),
        ("dice_weight", float("nan")),
    ],
)
def test_configuration_rejects_out_of_bounds_and_non_finite(field, value):
    with pytest.raises(ValidationError):
        FeTASegSearchConfiguration(**{field: value})


def test_only_registered_fidelities_and_exact_validation_schedules():
    expected = {
        25: (25,),
        50: (25, 50),
        100: (50, 100),
        150: (50, 100, 150),
        300: tuple(range(25, 301, 25)),
    }
    for fidelity, schedule in expected.items():
        assert FeTASegSearchConfiguration(maximum_epochs=fidelity).validation_epochs() == schedule
        assert validation_epochs(fidelity) == schedule
    with pytest.raises(ValidationError):
        FeTASegSearchConfiguration(maximum_epochs=75)


def test_baseline_augmentation_and_ratio_mappings_are_exact():
    baseline = augmentation_policy("baseline")
    assert baseline.probability == 0.2
    assert baseline.scale_factor == 0.1
    assert baseline.shift_offset == 0.1
    assert positive_negative_counts("1:1") == (1, 1)
    assert positive_negative_counts("2:1") == (2, 1)
    assert positive_negative_counts("3:1") == (3, 1)


def test_optuna_spec_has_exact_six_axes_and_fixed_fidelity():
    specification = FeTASegSearchTask().create_optuna_study_spec(
        default_feta_search_contract(), _request(maximum_epochs=100)
    )
    parameters = {parameter.name: parameter for parameter in specification.parameters}
    assert tuple(parameters) == (
        "learning_rate",
        "weight_decay",
        "dropout",
        "dice_weight",
        "positive_negative_ratio",
        "augmentation_strength",
    )
    assert (parameters["learning_rate"].low, parameters["learning_rate"].high) == (
        3e-5,
        5e-4,
    )
    assert parameters["learning_rate"].log is True
    assert (parameters["weight_decay"].low, parameters["weight_decay"].high) == (
        1e-6,
        3e-4,
    )
    assert parameters["weight_decay"].log is True
    assert specification.fixed_configuration["fold"] == 0
    assert specification.fixed_configuration["maximum_epochs"] == 100
    assert specification.n_startup_trials == 12
    assert specification.seed == 20260807


def test_optuna_space_can_be_narrowed_and_registered_axis_pinned():
    request = _request()
    request = request.model_copy(
        update={
            "search_space": {
                "trial_budget": 12,
                "seed": 17,
                "fixed": {
                    "fold": 0,
                    "maximum_epochs": 50,
                    "augmentation_strength": "baseline",
                },
                "parameters": {
                    "dropout": {"low": 0.1, "high": 0.3},
                },
            },
            "experiment_budget": 12,
        }
    )
    specification = FeTASegSearchTask().create_optuna_study_spec(
        default_feta_search_contract(), request
    )
    assert specification.trial_budget == 12
    assert specification.seed == 17
    assert specification.fixed_configuration["augmentation_strength"] == "baseline"
    assert "augmentation_strength" not in {
        parameter.name for parameter in specification.parameters
    }
    dropout = next(
        parameter for parameter in specification.parameters if parameter.name == "dropout"
    )
    assert (dropout.low, dropout.high) == (0.1, 0.3)


def test_direct_baseline_configuration_normalises_at_calibration_fidelities():
    task = FeTASegSearchTask()
    expected_fields = {
        "fold",
        "maximum_epochs",
        "learning_rate",
        "weight_decay",
        "dropout",
        "dice_weight",
        "positive_negative_ratio",
        "augmentation_strength",
    }
    for fidelity in (25, 50, 100, 150):
        normalised = task.normalise_configuration(
            baseline_search_configuration(fidelity)
        )
        assert set(normalised) == expected_fields
        assert not any(isinstance(value, list) for value in normalised.values())
        assert normalised["maximum_epochs"] == fidelity
        assert normalised["learning_rate"] == 1e-4
        assert normalised["augmentation_strength"] == "baseline"


def test_direct_backend_reconstructs_task_owned_vector_constants():
    task = FeTASegSearchTask()
    metadata = ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version="fixture-code",
        dataset_version="fixture-data",
        provenance=ProvenanceKind.REAL,
    )
    normalised = task.normalise_configuration(baseline_search_configuration(25))
    request = SearchRequest(
        request_id="direct-vector-regression",
        hypothesis_id="hypothesis",
        search_type=SearchType.DIRECT,
        target="mean_subject_macro_dice",
        search_space=normalised,
        experiment_budget=1,
        rationale="Reproduce the real DIRECT vector-configuration path.",
    )
    experiment = DirectSearchBackend(
        metadata, task.normalise_configuration
    ).create_experiment(
        request,
        default_feta_search_contract(),
        run_id="direct-vector-regression",
    )
    reconstructed = FeTASegSearchConfiguration.model_validate(
        experiment.configuration
    )
    assert reconstructed.blocks_down == (1, 2, 2, 4)
    assert reconstructed.blocks_up == (1, 1, 1)
    assert reconstructed.spacing_mm == (0.5, 0.5, 0.5)
    assert reconstructed.patch_size == (128, 128, 128)


def test_optuna_candidate_normalises_to_scalar_fields_and_reconstructs_defaults():
    task = FeTASegSearchTask()
    specification = task.create_optuna_study_spec(
        default_feta_search_contract(), _request(maximum_epochs=50)
    )
    candidate = dict(specification.fixed_configuration)
    candidate.update(
        {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "dropout": 0.2,
            "dice_weight": 1.0,
            "positive_negative_ratio": "1:1",
            "augmentation_strength": "baseline",
        }
    )
    normalised = task.normalise_configuration(candidate)
    assert not any(isinstance(value, list) for value in normalised.values())
    reconstructed = FeTASegSearchConfiguration.model_validate(normalised)
    assert reconstructed.patch_size == (128, 128, 128)
    assert payload_hash(reconstructed) == payload_hash(
        FeTASegSearchConfiguration(maximum_epochs=50)
    )


def test_generated_runner_result_is_valid_and_verifiable(tmp_path):
    evaluator, experiment, configuration = _evaluator(
        tmp_path, lambda _context, candidate, _experiment_id: _metrics(candidate)
    )
    contract = default_feta_search_contract()
    result = evaluator.evaluate(experiment, contract)
    assert result.success is True
    assert result.primary_score == pytest.approx(0.6)
    reconstructed = FeTASegSearchConfiguration.model_validate(
        experiment.configuration
    )
    assert result.metrics["configuration_identity"] == payload_hash(reconstructed)
    assert result.metrics["configuration"]["blocks_down"] == [1, 2, 2, 4]
    decision = FeTASegSearchTask().create_verification_policy(
        contract
    ).evaluate_constraints(result, contract)
    assert decision.constraint_compliant is True


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("dataset_manifest_hash", "wrong"),
        ("split_hash", "wrong"),
        ("fold_hash", "wrong"),
    ],
)
def test_evaluator_rejects_wrong_scientific_identity(tmp_path, field, wrong):
    def runner(_context, candidate, _experiment_id):
        result = _metrics(candidate)
        result[field] = wrong
        return result

    evaluator, experiment, _ = _evaluator(tmp_path, runner)
    result = evaluator.evaluate(experiment, default_feta_search_contract())
    assert result.success is False
    assert result.error == "feta_search_scientific_constraints_failed"


def test_non_finite_trial_fails_safely(tmp_path):
    def runner(_context, candidate, _experiment_id):
        result = _metrics(candidate)
        result["mean_subject_macro_dice"] = float("nan")
        return result

    evaluator, experiment, _ = _evaluator(tmp_path, runner)
    result = evaluator.evaluate(experiment, default_feta_search_contract())
    assert result.success is False
    assert result.error == "feta_search_scientific_json_invalid"
    assert result.metrics == {}
    assert "NaN" not in result.model_dump_json()


def test_artefact_policy_prohibits_patient_and_holdout_outputs():
    prohibited = FeTASegSearchTask().artefact_policy().prohibited_artefact_types
    assert {
        "raw_mri",
        "raw_mask",
        "raw_segmentation",
        "prediction",
        "holdout_data",
        "holdout_prediction",
        "holdout_metric",
    }.issubset(prohibited)
