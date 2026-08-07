from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest

from auto_researcher.agents.mock import MockHypothesisAgent, MockPlannerAgent
from auto_researcher.contracts.enums import RunStatus, SearchType
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
from auto_researcher.tasks.feta_seg.metrics import aggregate_subject_metrics, dice
from auto_researcher.tasks.feta_seg.splits import locked_partition
from auto_researcher.tasks.models import TaskRuntimeContext
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
