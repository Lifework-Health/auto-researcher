"""Read-only reporting and champion preservation for U-Net campaigns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.tasks.feta_unet_search.portfolio import (
    CandidateEvidence,
    _evidence,
    _tree_candidates,
)

REPORT_SCHEMA_VERSION = "feta-unet-campaign-postmortem-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_events(path: Path, run_id: str) -> tuple[DecisionEvent, ...]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError("feta_unet_campaign_provenance_missing")
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            "SELECT payload FROM decision_events WHERE run_id = ? ORDER BY sequence",
            (run_id,),
        ).fetchall()
    finally:
        connection.close()
    return tuple(DecisionEvent.model_validate_json(row[0]) for row in rows)


def _average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for identity, _ in ordered[index:end]:
            result[identity] = rank
        index = end
    return result


def _pearson(left: Iterable[float], right: Iterable[float]) -> float | None:
    xs = tuple(float(value) for value in left)
    ys = tuple(float(value) for value in right)
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return None if denominator == 0.0 else numerator / denominator


def _spearman(source: dict[str, float], target: dict[str, float]) -> dict[str, Any]:
    identities = sorted(set(source) & set(target))
    source_ranks = _average_ranks(
        {identity: source[identity] for identity in identities}
    )
    target_ranks = _average_ranks(
        {identity: target[identity] for identity in identities}
    )
    correlation = _pearson(
        (source_ranks[identity] for identity in identities),
        (target_ranks[identity] for identity in identities),
    )
    return {
        "common_trajectories": len(identities),
        "spearman_rho": correlation,
    }


def _best_rows(
    rows: tuple[CandidateEvidence, ...],
) -> dict[tuple[str, int], CandidateEvidence]:
    selected: dict[tuple[str, int], CandidateEvidence] = {}
    for row in rows:
        key = (row.trajectory_identity, row.fidelity)
        existing = selected.get(key)
        if existing is None or (row.best_score, row.experiment_id) > (
            existing.best_score,
            existing.experiment_id,
        ):
            selected[key] = row
    return selected


def campaign_report(
    *, runtime_root: Path, run_id: str
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    events = _read_events(runtime_root / "control" / "provenance.sqlite", run_id)
    rows = _evidence(events)
    if not rows:
        raise ValueError("feta_unet_campaign_evidence_missing")
    selected = _best_rows(rows)
    tree_by_experiment = {
        item.evidence.experiment_id: item for item in _tree_candidates(events, rows)
    }
    origin_by_trajectory: dict[str, CandidateEvidence] = {}
    for row in rows:
        existing = origin_by_trajectory.get(row.trajectory_identity)
        # Events are returned in durable sequence order. Keep the first
        # observation at the lowest fidelity so an OpenEvolve-imported seed is
        # attributed to the method that originally generated it.
        if existing is None or row.fidelity < existing.fidelity:
            origin_by_trajectory[row.trajectory_identity] = row

    stage_rows: dict[int, list[CandidateEvidence]] = defaultdict(list)
    for row in rows:
        stage_rows[row.fidelity].append(row)
    unique_stage_rows: dict[int, list[CandidateEvidence]] = defaultdict(list)
    for (_, fidelity), row in selected.items():
        unique_stage_rows[fidelity].append(row)

    stages: dict[str, Any] = {}
    for fidelity in sorted(stage_rows):
        executions = stage_rows[fidelity]
        unique = unique_stage_rows[fidelity]
        method_counts = Counter(
            origin_by_trajectory[item.trajectory_identity].search_type.value
            for item in unique
        )
        stages[str(fidelity)] = {
            "executions": len(executions),
            "unique_trajectories": len(unique),
            "duplicate_executions": len(executions) - len(unique),
            "origin_method_counts": dict(sorted(method_counts.items())),
            "best_rung_score": max(item.rung_score for item in unique),
            "best_validation_score": max(item.best_score for item in unique),
        }

    correlations: dict[str, Any] = {}
    for source, target in ((25, 50), (50, 100), (100, 150), (25, 150)):
        source_scores = {
            item.trajectory_identity: item.rung_score
            for item in unique_stage_rows.get(source, ())
        }
        target_scores = {
            item.trajectory_identity: item.rung_score
            for item in unique_stage_rows.get(target, ())
        }
        correlations[f"{source}_to_{target}"] = _spearman(source_scores, target_scores)

    finalists = unique_stage_rows.get(150, [])
    if not finalists:
        raise ValueError("feta_unet_campaign_finalist_missing")
    champion = max(
        finalists,
        key=lambda item: (item.best_score, item.trajectory_identity),
    )
    champion_origin = origin_by_trajectory[champion.trajectory_identity]

    candidate_rows = tuple(
        {
            "experiment_id": row.experiment_id,
            "trajectory_identity": row.trajectory_identity,
            "fidelity": row.fidelity,
            "recorded_search_type": row.search_type.value,
            "origin_search_type": origin_by_trajectory[
                row.trajectory_identity
            ].search_type.value,
            "rung_score": row.rung_score,
            "best_score": row.best_score,
            "trajectory_slope": row.trajectory_slope,
            "tree_stage": (
                tree_by_experiment[row.experiment_id].stage
                if row.experiment_id in tree_by_experiment
                else None
            ),
            "tree_action": (
                tree_by_experiment[row.experiment_id].action.value
                if row.experiment_id in tree_by_experiment
                else None
            ),
            "parent_trajectory": (
                tree_by_experiment[row.experiment_id].parent_trajectory
                if row.experiment_id in tree_by_experiment
                else None
            ),
            "root_trajectory": (
                tree_by_experiment[row.experiment_id].root_trajectory
                if row.experiment_id in tree_by_experiment
                else None
            ),
            "configuration": row.configuration,
        }
        for row in sorted(
            selected.values(),
            key=lambda item: (
                item.fidelity,
                -item.best_score,
                item.trajectory_identity,
            ),
        )
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "event_count": len(events),
        "verified_execution_count": len(rows),
        "unique_trajectory_fidelity_count": len(selected),
        "stages": stages,
        "rank_correlations": correlations,
        "tree_stage_counts": dict(
            sorted(Counter(item.stage for item in tree_by_experiment.values()).items())
        ),
        "champion": {
            "experiment_id": champion.experiment_id,
            "trajectory_identity": champion.trajectory_identity,
            "origin_search_type": champion_origin.search_type.value,
            "rung_score": champion.rung_score,
            "best_score": champion.best_score,
            "configuration": champion.configuration,
        },
    }
    return report, candidate_rows


def write_campaign_report(
    *,
    runtime_root: Path,
    run_id: str,
    report_dir: Path,
) -> dict[str, Any]:
    if report_dir.exists() and any(report_dir.iterdir()):
        raise ValueError("feta_unet_campaign_report_directory_not_empty")
    report_dir.mkdir(parents=True, exist_ok=True)
    report, candidates = campaign_report(runtime_root=runtime_root, run_id=run_id)
    (report_dir / "campaign-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (report_dir / "candidate-trajectories.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = (
            "experiment_id",
            "trajectory_identity",
            "fidelity",
            "recorded_search_type",
            "origin_search_type",
            "rung_score",
            "best_score",
            "trajectory_slope",
            "tree_stage",
            "tree_action",
            "parent_trajectory",
            "root_trajectory",
            "configuration_json",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames[:-1]},
                    "configuration_json": json.dumps(
                        row["configuration"], sort_keys=True, separators=(",", ":")
                    ),
                }
            )
    champion = report["champion"]
    summary = (
        "# FeTA U-Net campaign postmortem\n\n"
        f"- Run: `{run_id}`\n"
        f"- Champion experiment: `{champion['experiment_id']}`\n"
        f"- Champion origin: `{champion['origin_search_type']}`\n"
        f"- Champion best validation Dice: `{champion['best_score']:.9f}`\n"
        f"- Champion rung endpoint Dice: `{champion['rung_score']:.9f}`\n"
        f"- Verified executions: `{report['verified_execution_count']}`\n"
    )
    (report_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
    return report


def snapshot_champion(
    *,
    runtime_root: Path,
    run_id: str,
    report: dict[str, Any],
    snapshot_dir: Path,
) -> dict[str, Any]:
    if snapshot_dir.exists():
        raise ValueError("feta_unet_champion_snapshot_exists")
    experiment_id = str(report["champion"]["experiment_id"])
    result_root = runtime_root / "output" / "runs" / run_id / experiment_id
    workspace_matches = tuple(
        path
        for path in (runtime_root / "workspace").glob(f"*/{experiment_id}")
        if path.is_dir()
    )
    if not result_root.is_dir() or len(workspace_matches) != 1:
        raise ValueError("feta_unet_champion_artefacts_missing")
    checkpoint_root = workspace_matches[0] / "checkpoints" / "fold-0"
    sources = {
        "experiment_spec.json": result_root / "experiment_spec.json",
        "evaluation_result.json": result_root / "evaluation_result.json",
        "validation-history.json": checkpoint_root / "validation-history.json",
        "continuation.json": checkpoint_root / "continuation.json",
        "best.pt": checkpoint_root / "best.pt",
    }
    if any(not path.is_file() for path in sources.values()):
        raise ValueError("feta_unet_champion_artefacts_missing")
    snapshot_dir.mkdir(parents=True)
    files = []
    for name, source in sources.items():
        target = snapshot_dir / name
        shutil.copy2(source, target)
        files.append(
            {
                "name": name,
                "size_bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    manifest = {
        "schema_version": "feta-unet-champion-snapshot-v1",
        "run_id": run_id,
        "champion": report["champion"],
        "files": files,
    }
    (snapshot_dir / "snapshot-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    args = parser.parse_args(argv)
    report = write_campaign_report(
        runtime_root=args.runtime_root,
        run_id=args.run_id,
        report_dir=args.report_dir,
    )
    if args.snapshot_dir is not None:
        snapshot_champion(
            runtime_root=args.runtime_root,
            run_id=args.run_id,
            report=report,
            snapshot_dir=args.snapshot_dir,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
