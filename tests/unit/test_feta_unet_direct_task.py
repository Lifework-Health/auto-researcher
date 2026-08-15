from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from auto_researcher.agents.mock import MockHypothesisAgent, MockPlannerAgent
from auto_researcher.cli import _load_task_configuration
from auto_researcher.contracts.enums import ProvenanceKind, RunStatus, SearchType
from auto_researcher.contracts.models import ExperimentSpec
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.execution import start_run
from auto_researcher.tasks.feta_seg import FeTASegTask
from auto_researcher.tasks.feta_seg.manifests import (
    DATASET_RELEASE,
    EXPECTED_MANIFEST_HASH,
    FeTASubject,
)
from auto_researcher.tasks.feta_seg.metrics import aggregate_subject_metrics
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    locked_partition,
)
from auto_researcher.tasks.feta_unet_direct import (
    FeTAUNetDirectConfiguration,
    FeTAUNetDirectTask,
    default_feta_unet_direct_contract,
    engineering_smoke_configuration,
)
from auto_researcher.tasks.feta_unet_direct.evaluator import (
    ENGINEERING_SMOKE_RUNNER_ID,
    EVALUATOR_ID,
    EVALUATOR_VERSION,
    RESULT_ID,
    SCIENTIFIC_ID,
    FeTAUNetDirectEvaluator,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_unet_direct.fold_resume import (
    load_fold_result,
    persist_fold_result,
)
from auto_researcher.tasks.feta_unet_direct.identities import DATA_LOADER_ID
from auto_researcher.tasks.feta_unet_direct.model import (
    ARCHITECTURE_ID,
    TRAINABLE_PARAMETER_COUNT,
)
from auto_researcher.tasks.feta_unet_direct.runner import (
    FoldExecutionResult,
    orchestrate_profile_folds,
    select_profile_folds,
)
from auto_researcher.tasks.feta_unet_direct.trainer import checkpoint_reference
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    ReadinessCheck,
    ReadinessResult,
    TaskRuntimeContext,
)
from auto_researcher.tasks.registry import default_task_registry


def _methods() -> dict[str, str]:
    return {
        f"sub-{index:03d}": "mial" if index <= 40 else "irtk" for index in range(1, 81)
    }


def _subjects() -> tuple[FeTASubject, ...]:
    return tuple(
        FeTASubject(
            subject_id,
            method,
            Path(f"{subject_id}_image.nii.gz"),
            Path(f"{subject_id}_label.nii.gz"),
            "a" * 64,
            "b" * 64,
            (256, 256, 256),
            (0.5, 0.5, 0.5),
            tuple(range(8)),
        )
        for subject_id, method in _methods().items()
    )


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        task_id="feta_unet_direct",
        dataset_version=f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}",
        files=(),
        hashes={},
        loader_version="feta-flat-nifti-loader-v1",
        created_at=datetime(2026, 8, 15, tzinfo=UTC),
        metadata={
            "manifest_hash": EXPECTED_MANIFEST_HASH,
            "contains_subject_identifiers": False,
        },
    )


def _metadata() -> ExperimentMetadata:
    manifest = _manifest()
    return ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version=evaluator_code_version(manifest.dataset_version),
        dataset_version=manifest.dataset_version,
        provenance=ProvenanceKind.REAL,
    )


def _complete_smoke_metrics() -> dict:
    per_class = {
        str(label): {
            "label_name": f"label-{label}",
            "dice": 0.5,
            "hd95_mm": 2.0,
            "volume_similarity": 0.75,
            "euler_distance": 0,
            "empty_prediction": False,
        }
        for label in range(1, 8)
    }
    metrics = aggregate_subject_metrics(
        [
            {
                "subject_id": "protected-only",
                "reconstruction_method": "mial",
                "fold": 0,
                "per_class": per_class,
                "macro_dice": 0.5,
                "macro_hd95_mm": 2.0,
                "macro_volume_similarity": 0.75,
                "macro_euler_distance": 0.0,
                "empty_prediction_count": 0,
            }
        ]
    )
    metrics.pop("subject_metrics")
    metrics.update(
        {
            "fold_summaries": [],
            "checkpoint_references": [],
            "environment": {},
            "runner_id": ENGINEERING_SMOKE_RUNNER_ID,
            "data_loader_id": "monai-persistent-train-spawn4-uncached-validation-v3",
            "folds_completed": 1,
            "oof_subject_count": 1,
            "holdout_subjects_evaluated": 0,
            "failed_training_folds": 0,
            "valid_prediction_labels": list(range(8)),
            "contains_subject_identifiers": False,
        }
    )
    return metrics


def _evaluator(
    tmp_path: Path, runner
) -> tuple[FeTAUNetDirectEvaluator, ExperimentSpec]:
    metadata = _metadata()
    context = TaskRuntimeContext(
        run_id="feta-unet-test",
        output_dir=None,
        workspace_dir=tmp_path / "protected",
    )
    evaluator = FeTAUNetDirectEvaluator(
        context, metadata, _manifest(), profile_runner=runner
    )
    experiment = ExperimentSpec(
        experiment_id="feta-unet-experiment",
        hypothesis_id="hypothesis",
        search_request_id="request",
        configuration=engineering_smoke_configuration(),
        evaluator_id=metadata.evaluator_id,
        code_version=metadata.code_version,
        dataset_version=metadata.dataset_version,
        provenance=metadata.provenance,
    )
    return evaluator, experiment


def test_configuration_freezes_both_profiles_and_architecture():
    baseline = FeTAUNetDirectConfiguration()
    smoke = FeTAUNetDirectConfiguration.model_validate(
        engineering_smoke_configuration()
    )
    assert baseline.features == (32, 32, 64, 128, 256, 32)
    assert baseline.profile == "frozen_baseline"
    assert (baseline.maximum_epochs, baseline.fold_count) == (300, 5)
    assert (smoke.maximum_epochs, smoke.validation_every, smoke.fold_count) == (
        1,
        1,
        1,
    )
    with pytest.raises(ValueError, match="feta_unet_architecture_is_locked"):
        FeTAUNetDirectConfiguration(features=(16, 32, 64, 128, 256, 32))
    with pytest.raises(ValueError, match="feta_unet_training_configuration_is_locked"):
        FeTAUNetDirectConfiguration(learning_rate=0.001)
    with pytest.raises(ValueError, match="feta_unet_baseline_profile_is_locked"):
        FeTAUNetDirectConfiguration(maximum_epochs=1)


def test_task_is_separately_registered_without_reinterpreting_feta_seg():
    registry = default_task_registry()
    original = registry.get("feta_seg", "1.0")
    added = registry.get("feta_unet_direct", "1.0")
    assert isinstance(original, FeTASegTask)
    assert isinstance(added, FeTAUNetDirectTask)
    assert original.descriptor().evaluator_id == "feta-segresnet-evaluator"
    assert added.descriptor().evaluator_id == EVALUATOR_ID
    assert added.descriptor().supported_search_types == {SearchType.DIRECT}


def test_contract_locks_manifest_split_fold_architecture_and_holdout():
    contract = default_feta_unet_direct_contract()
    assert contract.objective_version == SCIENTIFIC_ID
    assert contract.constraints["dataset_manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert contract.constraints["split_hash"] == EXPECTED_SPLIT_HASH
    assert contract.constraints["fold_hash"] == EXPECTED_FOLD_HASH
    assert contract.constraints["architecture_identity"] == ARCHITECTURE_ID
    assert (
        contract.constraints["architecture_trainable_parameters"]
        == TRAINABLE_PARAMETER_COUNT
    )
    assert contract.constraints["holdout_policy"] == "sealed-no-evaluation"
    changed = contract.model_copy(
        update={"constraints": {**contract.constraints, "fold_hash": "changed"}}
    )
    with pytest.raises(
        ValueError, match="feta_unet_contract_scientific_identity_mismatch"
    ):
        FeTAUNetDirectTask().validate_contract(changed)


def test_shareable_manifest_preserves_identity_but_withholds_subject_rows(
    monkeypatch,
):
    source = DatasetManifest(
        task_id="feta_seg",
        dataset_version=f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}",
        files=("sub-001_rec-mial_T2w.nii.gz",),
        hashes={"sub-001_rec-mial_T2w.nii.gz": "a" * 64},
        loader_version="feta-flat-nifti-loader-v1",
        created_at=datetime(2026, 8, 15, tzinfo=UTC),
        metadata={
            "manifest_version": "feta-dataset-manifest-v1",
            "dataset_release": DATASET_RELEASE,
            "loader_version": "feta-flat-nifti-loader-v1",
            "subject_count": 80,
            "reconstruction_counts": {"mial": 40, "irtk": 40},
            "label_schema": {str(label): f"label-{label}" for label in range(8)},
            "labels": list(range(8)),
            "manifest_hash": EXPECTED_MANIFEST_HASH,
            "absolute_paths_in_identity": False,
            "subjects": [{"subject_id": "sub-001"}],
        },
    )
    monkeypatch.setattr(
        "auto_researcher.tasks.feta_unet_direct.task.build_dataset_manifest",
        lambda _: source,
    )
    manifest = FeTAUNetDirectTask().dataset_manifest(TaskRuntimeContext())
    assert manifest.task_id == "feta_unet_direct"
    assert manifest.dataset_version == source.dataset_version
    assert manifest.metadata["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert manifest.files == () and dict(manifest.hashes) == {}
    assert "subjects" not in manifest.metadata
    assert manifest.metadata["contains_subject_identifiers"] is False


def test_profile_selection_enforces_locked_fold_zero_and_sealed_holdout():
    partition = locked_partition(_methods())
    smoke = FeTAUNetDirectConfiguration.model_validate(
        engineering_smoke_configuration()
    )
    selections = select_profile_folds(smoke, _subjects(), partition)
    assert len(selections) == 1
    fold, training, validation = selections[0]
    assert fold == 0 and len(training) == 1 and len(validation) == 1
    exposed = {subject.subject_id for subject in training + validation}
    assert exposed.isdisjoint(partition.holdout)
    assert partition.folds[validation[0].subject_id] == 0
    assert partition.folds[training[0].subject_id] != 0


def test_baseline_orchestration_requires_exact_68_subject_oof_membership():
    partition = locked_partition(_methods())

    def execute(fold, training, validation):
        assert not {subject.subject_id for subject in training + validation} & set(
            partition.holdout
        )
        return FoldExecutionResult(
            fold=fold,
            subject_metrics=tuple(
                {"subject_id": subject.subject_id} for subject in validation
            ),
            best_epoch=5,
            validation_score=0.5,
            training_duration_seconds=1.0,
            total_duration_seconds=2.0,
            peak_gpu_memory_bytes=1024,
            checkpoint={"relative_path": f"fold-{fold}/best.pt"},
            seed=20260807 + fold,
        )

    results = orchestrate_profile_folds(
        FeTAUNetDirectConfiguration(), _subjects(), partition, execute
    )
    assert len(results) == 5
    assert sum(len(result.subject_metrics) for result in results) == 68


def _completed_fold(root: Path) -> FoldExecutionResult:
    checkpoint = root / "checkpoints/fold-0/best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"verified-basic-unet-checkpoint")
    reference = checkpoint_reference(
        checkpoint,
        fold=0,
        best_epoch=1,
        score=0.5,
        output_root=root / "checkpoints",
    )
    return FoldExecutionResult(
        fold=0,
        subject_metrics=({"subject_id": "sub-001", "macro_dice": 0.5},),
        best_epoch=1,
        validation_score=0.5,
        training_duration_seconds=1.0,
        total_duration_seconds=2.0,
        peak_gpu_memory_bytes=1024,
        checkpoint=reference,
        seed=20260807,
        source_runner_id=ENGINEERING_SMOKE_RUNNER_ID,
        source_data_loader_id=DATA_LOADER_ID,
    )


def test_completed_fold_restart_verifies_identity_and_checkpoint_integrity(tmp_path):
    configuration = FeTAUNetDirectConfiguration.model_validate(
        engineering_smoke_configuration()
    )
    validation = (_subjects()[0],)
    source = tmp_path / "source"
    result = _completed_fold(source)
    persist_fold_result(source, result, configuration, validation)
    reused = load_fold_result(
        source,
        tmp_path / "target",
        FoldExecutionResult,
        configuration,
        0,
        validation,
    )
    assert reused is not None and reused.reused_fold_result is True
    (source / "checkpoints/fold-0/best.pt").write_bytes(b"tampered")
    with pytest.raises(
        ValueError, match="feta_unet_fold_restart_checkpoint_identity_mismatch"
    ):
        load_fold_result(
            source,
            tmp_path / "other-target",
            FoldExecutionResult,
            configuration,
            0,
            validation,
        )


def test_result_identity_is_complete_finite_and_identifier_free(tmp_path):
    evaluator, experiment = _evaluator(tmp_path, lambda *_: _complete_smoke_metrics())
    result = evaluator.evaluate(experiment, default_feta_unet_direct_contract())
    assert result.success is True
    assert result.evaluator_version == EVALUATOR_VERSION
    assert result.metrics["result_identity"] == RESULT_ID
    assert result.metrics["architecture_identity"] == ARCHITECTURE_ID
    assert result.metrics["scientific_baseline"] is False
    assert result.metrics["holdout_subjects_evaluated"] == 0
    assert "subject_metrics" not in result.metrics
    encoded = result.model_dump_json().casefold()
    assert "protected-only" not in encoded
    assert '"subject_id":' not in encoded


def test_identifier_bearing_runner_output_fails_before_publication(tmp_path):
    def leaked(*_):
        return {**_complete_smoke_metrics(), "subject_id": "sub-001"}

    evaluator, experiment = _evaluator(tmp_path, leaked)
    result = evaluator.evaluate(experiment, default_feta_unet_direct_contract())
    assert result.success is False
    assert result.error == "feta_unet_shareable_evidence_invalid"
    assert "sub-001" not in result.model_dump_json()


def test_incomplete_runner_metrics_fail_safely(tmp_path):
    evaluator, experiment = _evaluator(
        tmp_path, lambda *_: {"contains_subject_identifiers": False}
    )
    result = evaluator.evaluate(experiment, default_feta_unet_direct_contract())
    assert result.success is False
    assert result.error == "feta_unet_metrics_incomplete"


@pytest.mark.parametrize(
    ("runner", "error"),
    [
        (
            lambda *_: {
                **_complete_smoke_metrics(),
                "mean_subject_macro_hd95_mm": float("nan"),
            },
            "feta_unet_scientific_json_invalid",
        ),
        (
            lambda *_: (_ for _ in ()).throw(
                ValueError("feta_unet_training_loss_non_finite")
            ),
            "feta_unet_training_loss_non_finite",
        ),
        (
            lambda *_: (_ for _ in ()).throw(
                ValueError("feta_unet_training_gradient_non_finite")
            ),
            "feta_unet_training_gradient_non_finite",
        ),
    ],
)
def test_non_finite_failures_are_safe_and_terminal(tmp_path, runner, error):
    evaluator, experiment = _evaluator(tmp_path, runner)
    result = evaluator.evaluate(experiment, default_feta_unet_direct_contract())
    assert result.success is False
    assert result.primary_score is None
    assert result.error == error


def test_standard_direct_runtime_assembly_and_execution(tmp_path, monkeypatch):
    task = FeTAUNetDirectTask()
    ready = ReadinessResult(
        ready=True,
        checks=(ReadinessCheck(code="offline_fixture", passed=True, message="ready"),),
    )
    monkeypatch.setattr(task, "readiness", lambda _: ready)
    monkeypatch.setattr(task, "dataset_manifest", lambda _: _manifest())
    monkeypatch.setattr(task, "experiment_metadata", lambda _: _metadata())
    evaluator, _ = _evaluator(tmp_path, lambda *_: _complete_smoke_metrics())
    configuration = engineering_smoke_configuration()
    contract = default_feta_unet_direct_contract()
    context = TaskRuntimeContext(
        run_id="feta-unet-standard-runtime",
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "protected",
        output_dir=tmp_path / "shareable",
        task_options={"protected_storage_acknowledged": True},
        manifest_created_at=datetime(2026, 8, 15, tzinfo=UTC),
    )
    dependencies = task_memory_dependencies(
        task,
        context,
        contract,
        configuration,
        evaluator=evaluator,
        hypothesis_agent=MockHypothesisAgent(),
        planner_agent=MockPlannerAgent(
            search_type=SearchType.DIRECT,
            configuration=configuration,
            experiment_budget=1,
        ),
        search_type=SearchType.DIRECT,
    )
    final = start_run(
        build_graph(dependencies),
        {
            "run_id": "feta-unet-standard-runtime",
            "thread_id": "feta-unet-standard-runtime-thread",
            "contract": contract,
        },
        {"configurable": {"thread_id": "feta-unet-standard-runtime-thread"}},
    )
    assert final["status"] == RunStatus.COMPLETED
    assert final["evaluation_result"].metrics["result_identity"] == RESULT_ID
    assert dependencies.task_descriptor.task_id == "feta_unet_direct"


def test_example_profiles_use_standard_direct_runtime_and_explicit_paths():
    root = Path(__file__).resolve().parents[2] / "examples/tasks/feta_unet_direct"
    smoke, smoke_runtime = _load_task_configuration(
        root / "engineering-smoke.yaml", "feta_unet_direct", "1.0"
    )
    baseline, baseline_runtime = _load_task_configuration(
        root / "frozen-baseline.yaml", "feta_unet_direct", "1.0"
    )
    assert smoke == engineering_smoke_configuration()
    assert baseline["profile"] == "frozen_baseline"
    for runtime in (smoke_runtime, baseline_runtime):
        assert Path(runtime["data_dir"]).is_absolute()
        assert Path(runtime["workspace_dir"]).is_absolute()
        assert Path(runtime["output_dir"]).is_absolute()
        assert runtime["options"]["protected_storage_acknowledged"] is True
        assert ".auto-researcher" not in json.dumps(runtime)
