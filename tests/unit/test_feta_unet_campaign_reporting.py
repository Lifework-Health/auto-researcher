from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from auto_researcher.contracts.enums import (
    EventType,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore
from auto_researcher.tasks.feta_unet_search.configuration import (
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.reporting import (
    campaign_report,
    snapshot_champion,
    write_campaign_report,
)


def _configuration(*, fidelity: int, learning_rate: float) -> dict:
    return FeTAUNetSearchConfiguration(
        maximum_epochs=fidelity,  # type: ignore[arg-type]
        learning_rate=learning_rate,
    ).model_dump(mode="json")


def _event(
    index: int,
    *,
    search_type: SearchType,
    configuration: dict,
    endpoint: float,
    best: float,
) -> DecisionEvent:
    fidelity = configuration["maximum_epochs"]
    return DecisionEvent(
        event_id=f"event-{index}",
        run_id="campaign-run",
        cycle=index,
        event_type=EventType.EVIDENCE_VERIFIED,
        actor="verifier",
        input_references=(f"experiment-{index}",),
        output_references=(
            "evidence:SUPPORTED",
            "verified:true",
            "constraints:true",
            f"score:{best}",
            f"search_type:{search_type.value}",
        ),
        rationale="verified aggregate development evidence",
        timestamp=datetime(2026, 8, 19, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
        safe_payload={
            "configuration": configuration,
            "aggregate_metrics": {
                "validation_history": [
                    {"epoch": 5, "validation_score": endpoint - 0.1},
                    {"epoch": fidelity, "validation_score": endpoint},
                ]
            },
        },
    )


def _runtime(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    (root / "control").mkdir(parents=True)
    store = SQLiteProvenanceStore(root / "control" / "provenance.sqlite")
    index = 0
    for learning_rate, origin in (
        (0.0001, SearchType.OPTUNA),
        (0.0002, SearchType.OPENEVOLVE),
    ):
        for fidelity, endpoint, best in (
            (25, 0.70 + index / 100, 0.70 + index / 100),
            (50, 0.75 + index / 100, 0.76 + index / 100),
            (100, 0.79 + index / 100, 0.80 + index / 100),
            (150, 0.81 + index / 100, 0.82 + index / 100),
        ):
            store.append_event(
                _event(
                    index,
                    search_type=origin if fidelity == 25 else SearchType.DIRECT,
                    configuration=_configuration(
                        fidelity=fidelity, learning_rate=learning_rate
                    ),
                    endpoint=endpoint,
                    best=best,
                )
            )
            index += 1
    # OpenEvolve imports the Optuna incumbent. This is an execution but not a
    # new scientific trajectory, and its origin must remain OPTUNA.
    store.append_event(
        _event(
            index,
            search_type=SearchType.OPENEVOLVE,
            configuration=_configuration(fidelity=25, learning_rate=0.0001),
            endpoint=0.70,
            best=0.70,
        )
    )
    return root


def test_campaign_report_separates_executions_trajectories_and_origins(tmp_path):
    root = _runtime(tmp_path)
    report, candidates = campaign_report(
        runtime_root=root, run_id="campaign-run"
    )

    assert report["verified_execution_count"] == 9
    assert report["stages"]["25"] == {
        "executions": 3,
        "unique_trajectories": 2,
        "duplicate_executions": 1,
        "origin_method_counts": {"OPENEVOLVE": 1, "OPTUNA": 1},
        "best_rung_score": 0.74,
        "best_validation_score": 0.74,
    }
    assert report["champion"]["origin_search_type"] == "OPENEVOLVE"
    assert report["champion"]["best_score"] == 0.86
    assert report["rank_correlations"]["25_to_150"] == {
        "common_trajectories": 2,
        "spearman_rho": 1.0,
    }
    assert len(candidates) == 8


def test_report_writer_and_champion_snapshot_are_checksum_verified(tmp_path):
    root = _runtime(tmp_path)
    report_dir = tmp_path / "report"
    report = write_campaign_report(
        runtime_root=root,
        run_id="campaign-run",
        report_dir=report_dir,
    )
    experiment_id = report["champion"]["experiment_id"]
    result_root = root / "output" / "runs" / "campaign-run" / experiment_id
    checkpoint_root = (
        root / "workspace" / "namespace" / experiment_id / "checkpoints" / "fold-0"
    )
    result_root.mkdir(parents=True)
    checkpoint_root.mkdir(parents=True)
    for name in ("experiment_spec.json", "evaluation_result.json"):
        (result_root / name).write_text("{}\n", encoding="utf-8")
    for name in ("validation-history.json", "continuation.json"):
        (checkpoint_root / name).write_text("{}\n", encoding="utf-8")
    (checkpoint_root / "best.pt").write_bytes(b"checkpoint")

    manifest = snapshot_champion(
        runtime_root=root,
        run_id="campaign-run",
        report=report,
        snapshot_dir=tmp_path / "champion",
    )

    assert (report_dir / "campaign-report.json").is_file()
    assert (report_dir / "candidate-trajectories.csv").is_file()
    assert {item["name"] for item in manifest["files"]} == {
        "experiment_spec.json",
        "evaluation_result.json",
        "validation-history.json",
        "continuation.json",
        "best.pt",
    }
    persisted = json.loads(
        (tmp_path / "champion" / "snapshot-manifest.json").read_text()
    )
    assert persisted == manifest


def test_rank_correlations_follow_the_fidelities_actually_observed(tmp_path):
    root = tmp_path / "runtime"
    (root / "control").mkdir(parents=True)
    store = SQLiteProvenanceStore(root / "control" / "provenance.sqlite")
    index = 0
    for learning_rate in (0.0001, 0.0002):
        for fidelity in (10, 15, 150):
            store.append_event(
                _event(
                    index,
                    search_type=SearchType.OPTUNA,
                    configuration=_configuration(
                        fidelity=fidelity, learning_rate=learning_rate
                    ),
                    endpoint=0.7 + fidelity / 1_000 + index / 10_000,
                    best=0.7 + fidelity / 1_000 + index / 10_000,
                )
            )
            index += 1

    report, _ = campaign_report(runtime_root=root, run_id="campaign-run")

    assert tuple(report["rank_correlations"]) == (
        "10_to_15",
        "15_to_150",
        "10_to_150",
    )
    assert all(
        item["common_trajectories"] == 2
        for item in report["rank_correlations"].values()
    )
