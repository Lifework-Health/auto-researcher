from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from auto_researcher.agents.mock import MockHypothesisAgent, MockPlannerAgent
from auto_researcher.contracts.enums import RunStatus, SearchType
from auto_researcher.contracts.enums import ProvenanceKind
from auto_researcher.contracts.models import ExperimentSpec
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.execution import start_run
from auto_researcher.tasks.feta_seg import (
    FeTASegConfiguration,
    FeTASegTask,
    default_feta_contract,
    smoke_configuration,
)
from auto_researcher.tasks.feta_seg.manifests import discover_pairs
from auto_researcher.tasks.feta_seg.manifests import (
    EXPECTED_MANIFEST_HASH,
    FeTASubject,
)
from auto_researcher.tasks.feta_seg.metrics import aggregate_subject_metrics, dice
from auto_researcher.tasks.feta_seg.evaluator import (
    EVALUATOR_ID,
    EVALUATOR_VERSION,
    FeTASegEvaluator,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_seg.runner import (
    FoldExecutionResult,
    orchestrate_development_folds,
)
from auto_researcher.tasks.feta_seg.splits import locked_partition
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)
from auto_researcher.tasks.registry import default_task_registry


def methods() -> dict[str, str]:
    return {
        f"sub-{index:03d}": "mial" if index <= 40 else "irtk" for index in range(1, 81)
    }


def smoke_context(tmp_path: Path) -> TaskRuntimeContext:
    return TaskRuntimeContext(
        run_id="feta-smoke-run",
        output_dir=tmp_path / "artefacts",
        workspace_dir=tmp_path / "workspace",
        task_options={"mode": "smoke"},
        manifest_created_at=datetime(2026, 8, 7, tzinfo=UTC),
    )


def fake_subjects() -> tuple[FeTASubject, ...]:
    return tuple(
        FeTASubject(
            subject_id,
            method,
            Path(f"{subject_id}_image.nii.gz"),
            Path(f"{subject_id}_label.nii.gz"),
            "a" * 64,
            "b" * 64,
            (8, 8, 8),
            (0.5, 0.5, 0.5),
            tuple(range(8)),
        )
        for subject_id, method in methods().items()
    )


def complete_fake_metrics() -> dict:
    rows = []
    partition = locked_partition(methods())
    for subject_id in partition.development:
        method = methods()[subject_id]
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
        rows.append(
            {
                "subject_id": subject_id,
                "reconstruction_method": method,
                "fold": partition.folds[subject_id],
                "per_class": per_class,
                "macro_dice": 0.5,
                "macro_hd95_mm": 2.0,
                "macro_volume_similarity": 0.75,
                "macro_euler_distance": 0.0,
                "empty_prediction_count": 0,
            }
        )
    result = aggregate_subject_metrics(rows)
    result.update(
        {
            "folds_completed": 5,
            "oof_subject_count": 68,
            "holdout_subjects_evaluated": 0,
            "failed_training_folds": 0,
            "valid_prediction_labels": list(range(8)),
            "fold_summaries": [],
            "checkpoint_references": [],
            "environment": {},
        }
    )
    return result


def fake_full_evaluator(
    tmp_path: Path, runner, *, manifest_hash=EXPECTED_MANIFEST_HASH
):
    dataset_version = f"feta-2.1-export-80+{manifest_hash}"
    manifest = DatasetManifest(
        task_id="feta_seg",
        dataset_version=dataset_version,
        files=(),
        hashes={},
        loader_version="fixture",
        created_at=datetime(2026, 8, 7, tzinfo=UTC),
        metadata={"manifest_hash": manifest_hash},
    )
    metadata = ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version=evaluator_code_version(dataset_version),
        dataset_version=dataset_version,
        provenance=ProvenanceKind.REAL,
    )
    context = TaskRuntimeContext(
        run_id="feta-full-fixture",
        output_dir=None,
        workspace_dir=tmp_path,
    )
    evaluator = FeTASegEvaluator(context, metadata, manifest, full_runner=runner)
    experiment = ExperimentSpec(
        experiment_id="feta-full-experiment",
        hypothesis_id="hypothesis",
        search_request_id="request",
        configuration=FeTASegConfiguration().scientific_configuration(),
        evaluator_id=metadata.evaluator_id,
        code_version=metadata.code_version,
        dataset_version=metadata.dataset_version,
        provenance=metadata.provenance,
    )
    return evaluator, experiment


def test_task_registered_direct_only():
    task = default_task_registry().get("feta_seg", "1.0")
    assert task.descriptor().supported_search_types == {SearchType.DIRECT}
    assert not hasattr(task, "live_mutation_dataset_class")
    assert not hasattr(task, "create_optuna_study_spec")
    assert not hasattr(task, "create_evolvable_component")


def test_readiness_without_data_is_safe_failure():
    result = FeTASegTask().readiness(TaskRuntimeContext())
    assert result.ready is False
    assert "feta_data_available" in result.errors


def test_generated_smoke_readiness_is_explicitly_non_scientific(tmp_path):
    result = FeTASegTask().readiness(smoke_context(tmp_path))
    assert result.ready is True
    assert result.warnings == ("not_scientific_baseline",)


def test_pairing_prefers_compressed_duplicate(tmp_path):
    for name in (
        "sub-001_rec-mial_T2w.nii",
        "sub-001_rec-mial_T2w.nii.gz",
        "sub-001_rec-mial_dseg.nii.gz",
    ):
        (tmp_path / name).write_bytes(b"same")
    pairs, warnings = discover_pairs(tmp_path)
    assert pairs["sub-001"][0].name.endswith(".nii.gz")
    assert warnings


@pytest.mark.parametrize("missing", ["T2w", "dseg"])
def test_pairing_rejects_missing_files(tmp_path, missing):
    kind = "dseg" if missing == "T2w" else "T2w"
    (tmp_path / f"sub-001_rec-mial_{kind}.nii.gz").write_bytes(b"x")
    with pytest.raises(ValueError, match="feta_(image|label)_missing"):
        discover_pairs(tmp_path)


def test_locked_split_is_deterministic_disjoint_and_balanced():
    first = locked_partition(methods())
    second = locked_partition(methods())
    assert first == second
    assert len(first.holdout) == 12 and len(first.development) == 68
    assert not set(first.holdout) & set(first.development)
    assert Counter(methods()[item] for item in first.holdout) == {"mial": 6, "irtk": 6}
    counts = Counter(first.folds.values())
    assert counts == {0: 14, 1: 14, 2: 14, 3: 14, 4: 12}
    assert (
        first.split_hash
        == "3ee6e9f02b4d35f7611bb70cdf19aea3ebc12f81ef89b57291eac9983df66561"
    )
    assert (
        first.fold_hash
        == "45e70dc010448d124b978a8becdc5866264b457c3b5ffddc802916f30ec28f6e"
    )


def test_locked_split_rejects_inventory_drift():
    values = methods()
    values.pop("sub-080")
    with pytest.raises(ValueError, match="feta_inventory_not_80_40_40"):
        locked_partition(values)


def test_configuration_keeps_full_and_smoke_identity_separate():
    assert FeTASegConfiguration().mode == "full"
    smoke = FeTASegConfiguration.model_validate(smoke_configuration())
    assert smoke.mode == "smoke" and smoke.fold_count == 1
    with pytest.raises(ValueError, match="feta_full_baseline_budget_is_locked"):
        FeTASegConfiguration(maximum_epochs=2)


def test_dice_and_subject_macro_arithmetic():
    assert dice([1, 1, 2, 2], [1, 2, 2, 2], 1) == pytest.approx(2 / 3)
    rows = [
        {
            "reconstruction_method": method,
            "dice": {str(label): value for label in range(1, 8)},
        }
        for method, value in (("mial", 0.6), ("irtk", 0.8))
    ]
    result = aggregate_subject_metrics(rows)
    assert result["mean_subject_macro_dice"] == pytest.approx(0.7)
    assert result["reconstruction_gap"] == pytest.approx(0.2)


def test_metric_rejects_absent_reference_tissue():
    with pytest.raises(ValueError, match="feta_subject_tissue_absent"):
        dice([1, 1], [1, 1], 2)


def test_smoke_direct_graph_lifecycle_and_holdout_exclusion(tmp_path):
    task = FeTASegTask()
    contract = default_feta_contract()
    context = smoke_context(tmp_path)
    configuration = smoke_configuration()
    hypothesis_agent = MockHypothesisAgent()
    planner_agent = MockPlannerAgent(
        search_type=SearchType.DIRECT, configuration=configuration, experiment_budget=1
    )
    dependencies = task_memory_dependencies(
        task,
        context,
        contract,
        configuration,
        hypothesis_agent=hypothesis_agent,
        planner_agent=planner_agent,
        search_type=SearchType.DIRECT,
    )
    final = start_run(
        build_graph(dependencies),
        {
            "run_id": "feta-smoke-run",
            "thread_id": "feta-smoke-thread",
            "contract": contract,
        },
        {"configurable": {"thread_id": "feta-smoke-thread"}},
    )
    assert final["status"] == RunStatus.COMPLETED
    assert final["evaluation_result"].success is True
    assert final["evaluation_result"].metrics["scientific_baseline"] is False
    assert final["evaluation_result"].metrics["holdout_subjects_evaluated"] == 0
    assert final["verification_result"].verified is True
    assert (
        dependencies.provenance_store.get_evaluation_reuse(
            "feta-smoke-run", final["experiment_spec"].experiment_id
        )
        is not None
    )


def test_smoke_identity_cannot_validate_as_full():
    smoke = FeTASegConfiguration.model_validate(smoke_configuration())
    assert (
        smoke.scientific_configuration()
        != FeTASegConfiguration().scientific_configuration()
    )


def test_full_fold_orchestration_is_oof_complete_and_never_exposes_holdout():
    partition = locked_partition(methods())
    calls = []

    def execute(fold, training, validation):
        calls.append((fold, training, validation))
        rows = tuple({"subject_id": subject.subject_id} for subject in validation)
        return FoldExecutionResult(
            fold=fold,
            subject_metrics=rows,
            best_epoch=5,
            validation_score=0.5,
            training_duration_seconds=1.0,
            total_duration_seconds=2.0,
            peak_gpu_memory_bytes=1024,
            checkpoint={"relative_path": f"fold-{fold}/best.pt"},
            seed=20260807 + fold,
        )

    results = orchestrate_development_folds(
        FeTASegConfiguration(), fake_subjects(), partition, execute
    )
    assert len(results) == 5
    assert {
        row["subject_id"] for result in results for row in result.subject_metrics
    } == set(partition.development)
    for _, training, validation in calls:
        exposed = {subject.subject_id for subject in training + validation}
        assert exposed.isdisjoint(partition.holdout)


def test_fold_orchestration_rejects_incomplete_or_wrong_oof_membership():
    partition = locked_partition(methods())

    def execute(fold, training, validation):
        rows = tuple({"subject_id": subject.subject_id} for subject in validation[:-1])
        return FoldExecutionResult(
            fold=fold,
            subject_metrics=rows,
            best_epoch=5,
            validation_score=0.5,
            training_duration_seconds=1.0,
            total_duration_seconds=2.0,
            peak_gpu_memory_bytes=1024,
            checkpoint={},
            seed=20260807 + fold,
        )

    with pytest.raises(ValueError, match="feta_oof_membership_invalid"):
        orchestrate_development_folds(
            FeTASegConfiguration(), fake_subjects(), partition, execute
        )


def test_full_evaluator_accepts_complete_finite_fake_runner(tmp_path):
    evaluator, experiment = fake_full_evaluator(
        tmp_path, lambda context, configuration, experiment_id: complete_fake_metrics()
    )
    result = evaluator.evaluate(experiment, default_feta_contract())
    assert result.success is True
    assert result.evaluator_version == EVALUATOR_VERSION
    assert result.primary_score == pytest.approx(0.5)
    assert all(result.constraint_results.values())
    assert result.metrics["holdout_subjects_evaluated"] == 0


def test_full_evaluator_rejects_non_finite_scientific_json(tmp_path):
    def non_finite(*args):
        result = complete_fake_metrics()
        result["mean_subject_macro_hd95_mm"] = float("nan")
        return result

    evaluator, experiment = fake_full_evaluator(tmp_path, non_finite)
    result = evaluator.evaluate(experiment, default_feta_contract())
    assert result.success is False
    assert result.error == "feta_scientific_json_invalid"
    assert result.artefact_references == ()


def test_full_evaluator_rejects_dataset_identity_mismatch(tmp_path):
    evaluator, experiment = fake_full_evaluator(
        tmp_path, lambda *args: complete_fake_metrics(), manifest_hash="0" * 64
    )
    result = evaluator.evaluate(experiment, default_feta_contract())
    assert result.success is False
    assert result.constraint_results["dataset_identity_exact"] is False


def test_feta_artefact_publication_failure_returns_no_references(tmp_path, monkeypatch):
    task = FeTASegTask()
    context = smoke_context(tmp_path)
    evaluator = task.create_evaluator(context)
    metadata = task.experiment_metadata(context)
    experiment = ExperimentSpec(
        experiment_id="feta-persistence-failure",
        hypothesis_id="hypothesis",
        search_request_id="request",
        configuration=smoke_configuration(),
        evaluator_id=metadata.evaluator_id,
        code_version=metadata.code_version,
        dataset_version=metadata.dataset_version,
        provenance=metadata.provenance,
    )

    def fail_bundle(*args, **kwargs):
        raise OSError("private filesystem detail")

    monkeypatch.setattr(
        "auto_researcher.tasks.feta_seg.evaluator.write_artefact_bundle",
        fail_bundle,
    )
    result = evaluator.evaluate(experiment, default_feta_contract())
    assert result.success is False
    assert result.artefact_references == ()
    assert result.error == "artefact_bundle_publication_failed:OSError"
    assert "private filesystem detail" not in result.model_dump_json()


def test_feta_verifier_rejects_split_fold_and_evaluator_identity_drift(tmp_path):
    evaluator, experiment = fake_full_evaluator(
        tmp_path, lambda *args: complete_fake_metrics()
    )
    result = evaluator.evaluate(experiment, default_feta_contract())
    assert result.success is True
    policy = FeTASegTask().create_verification_policy(default_feta_contract())
    for field, value, reason in (
        ("split_hash", "wrong", "feta_split_identity_mismatch"),
        ("fold_hash", "wrong", "feta_fold_identity_mismatch"),
        ("evaluator_version", "wrong", "feta_evaluator_identity_mismatch"),
    ):
        metrics = dict(result.metrics)
        metrics[field] = value
        changed = result.model_copy(update={"metrics": metrics})
        decision = policy.evaluate_constraints(changed, default_feta_contract())
        assert decision.constraint_compliant is False
        assert reason in decision.reasons


def test_feta_and_synthetic_share_graph_topology_and_direct_node_sequence(tmp_path):
    feta_contract = default_feta_contract()
    feta_dependencies = task_memory_dependencies(
        FeTASegTask(),
        smoke_context(tmp_path / "feta"),
        feta_contract,
        smoke_configuration(),
        hypothesis_agent=MockHypothesisAgent(),
        planner_agent=MockPlannerAgent(
            search_type=SearchType.DIRECT,
            configuration=smoke_configuration(),
            experiment_budget=1,
        ),
        search_type=SearchType.DIRECT,
    )
    synthetic_contract = default_synthetic_contract()
    synthetic_dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(manifest_created_at=datetime(2026, 8, 7, tzinfo=UTC)),
        synthetic_contract,
        default_synthetic_configuration(),
        hypothesis_agent=MockHypothesisAgent(),
        planner_agent=MockPlannerAgent(
            search_type=SearchType.DIRECT,
            configuration=default_synthetic_configuration(),
            experiment_budget=1,
        ),
        search_type=SearchType.DIRECT,
    )
    feta_graph = build_graph(feta_dependencies)
    synthetic_graph = build_graph(synthetic_dependencies)
    assert (
        feta_graph.get_graph().draw_mermaid()
        == synthetic_graph.get_graph().draw_mermaid()
    )
    feta_final = start_run(
        feta_graph,
        {
            "run_id": "feta-shared",
            "thread_id": "feta-shared-thread",
            "contract": feta_contract,
        },
        {"configurable": {"thread_id": "feta-shared-thread"}},
    )
    synthetic_final = start_run(
        synthetic_graph,
        {
            "run_id": "synthetic-shared",
            "thread_id": "synthetic-shared-thread",
            "contract": synthetic_contract,
        },
        {"configurable": {"thread_id": "synthetic-shared-thread"}},
    )
    assert feta_final["executed_nodes"] == synthetic_final["executed_nodes"]
