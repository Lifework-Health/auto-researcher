from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from auto_researcher.benchmarks.vcc2026.models import (
    BaselineFixture,
    load_baseline_fixture,
)
from auto_researcher.benchmarks.vcc2026.preflight import run_baseline_preflight

ROOT = Path(__file__).resolve().parents[2]
CHECKED_IN_FIXTURE = ROOT / "examples/benchmarks/vcc2026/viet-b2-rank82-baseline.json"
SIX_METRICS = (
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
)
REQUIRED_GUARDRAILS = (
    "13.1_never_tune_on_leaderboard",
    "13.2_no_target_perturbed_data",
    "13.3_no_crispra",
    "13.4_dispersion_source",
    "13.5_no_single_aggregate",
    "13.6_d1_gate",
    "13.7_collapse_canary",
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write(path, json.dumps(value, indent=2) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(checkout: Path) -> str:
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test User"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "remote",
            "add",
            "origin",
            "https://github.com/v-iettran/vcc2026.git",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _submission_report() -> dict[str, object]:
    contexts = {
        context: {
            "n_perturbations": 300,
            "collapse_canary": 0.9924837350845337,
            "n_zero_predictions": 36,
        }
        for context in ("A", "B", "C")
    }
    return {
        "baseline": "B2",
        "params": {},
        "alpha": 1.0,
        "contexts": contexts,
        "validation": {"ok_total": True, "ok_nonzeros": True},
        "n_cells": 360000,
        "n_genes": 18533,
        "has_control_rows": False,
        "vcc_prep": {"returncode": 0, "passed": True},
    }


def _build_checkout(
    tmp_path: Path,
    *,
    campaign_ready: bool,
) -> tuple[Path, BaselineFixture]:
    checkout = tmp_path / "vcc2026"
    checkout.mkdir()
    _write(checkout / "README.md", "VCC 2026 baseline\n")
    _write(
        checkout / "requirements.txt",
        "cell-eval2==0.16.0\n" if campaign_ready else "cell-eval>=0.8.2\n",
    )
    _write(
        checkout / "spec/vcc2026_baseline_spec.md",
        "Official profile: vcc2026\n",
    )
    _write(checkout / "src/config.py", 'SCORER_PROFILE = "vcc2026"\n')
    _write(
        checkout / "src/eval/score.py",
        'CELL_EVAL2_VERSION = version("cell-eval2")\n'
        "# cell-eval` v1 is a different package\n",
    )
    _write(
        checkout / "results/README.md",
        "Official cell-eval2 reporting.\n"
        if campaign_ready
        else "The scorer is a version behind.\ncell-eval 0.8.2 is the VCC-2025 metric set.\n",
    )
    _write(
        checkout / "results/decisions.md",
        "# D1-D7\nAll frozen decisions computed.\n"
        if campaign_ready
        else "# Decision rules\n_Not yet computed._\n",
    )
    guardrails = {code: {"ok": True} for code in REQUIRED_GUARDRAILS}
    if not campaign_ready:
        guardrails["13.5_no_single_aggregate"] = {"ok": None}
        guardrails["13.6_d1_gate"] = {"ok": None}
        guardrails["13.7_collapse_canary"] = {"ok": None}
    _write_json(checkout / "results/guardrails.json", guardrails)
    _write_json(
        checkout / "results/submission/submission_B2_report.json",
        _submission_report(),
    )

    commit = _git_commit(checkout)
    bound_paths = (
        "README.md",
        "requirements.txt",
        "spec/vcc2026_baseline_spec.md",
        "src/config.py",
        "src/eval/score.py",
        "results/README.md",
        "results/decisions.md",
        "results/guardrails.json",
        "results/submission/submission_B2_report.json",
    )
    expected_blockers = (
        frozenset()
        if campaign_ready
        else frozenset(
            {
                "runtime_dataset_manifest_unbound",
                "scientific_decisions_uncomputed",
                "scientific_guardrails_unresolved",
                "scorer_dependency_not_pinned",
                "stale_scorer_reporting",
                "submission_payload_unbound",
            }
        )
    )
    frozen_hash = "a" * 64
    fixture = BaselineFixture.model_validate(
        {
            "schema_version": "vcc2026-baseline-fixture-v1",
            "challenge_id": "arc-virtual-cell-2026",
            "source_repository": "https://github.com/v-iettran/vcc2026",
            "source_commit": commit,
            "baseline_id": "viet-b2-shared-delta-rank82",
            "files": [
                {"path": path, "sha256": _sha256(checkout / path)}
                for path in bound_paths
            ],
            "scorer": {
                "distribution": "cell-eval2",
                "version": "0.16.0",
                "source_repository": "https://github.com/ArcInstitute/cell-eval2",
                "source_commit": "5e64833518a6603a0301cbe28185d49c30f4a986",
                "profile": "vcc2026",
                "scored_metrics": SIX_METRICS,
            },
            "submission": {
                "baseline": "B2",
                "alpha": 1.0,
                "contexts": ("A", "B", "C"),
                "perturbations_per_context": 300,
                "cells_per_perturbation": 400,
                "n_cells": 360000,
                "n_genes": 18533,
                "has_control_rows": False,
                "vcc_prep_required": True,
            },
            "runtime_bindings": {
                "dataset_manifest_sha256": frozen_hash if campaign_ready else None,
                "submission_h5ad_sha256": frozen_hash if campaign_ready else None,
                "submission_vcc_sha256": frozen_hash if campaign_ready else None,
            },
            "leaderboard_event": {
                "rank": 82,
                "evidence_status": "user_reported",
                "submitted_baseline": "B2",
                "tuning_signal_allowed": False,
                "receipt_sha256": None,
            },
            "required_guardrails": REQUIRED_GUARDRAILS,
            "expected_archival_blockers": expected_blockers,
        }
    )
    return checkout, fixture


def test_checked_in_fixture_freezes_viet_baseline_and_rank_event():
    fixture = load_baseline_fixture(CHECKED_IN_FIXTURE)

    assert fixture.source_commit == "dfb906d135f7b962350004b179107ef1101be353"
    assert fixture.scorer.version == "0.16.0"
    assert fixture.scorer.profile == "vcc2026"
    assert fixture.scorer.scored_metrics == SIX_METRICS
    assert fixture.leaderboard_event.rank == 82
    assert fixture.leaderboard_event.evidence_status == "user_reported"
    assert fixture.leaderboard_event.tuning_signal_allowed is False


def test_archival_fixture_can_match_while_campaign_remains_blocked(tmp_path):
    checkout, fixture = _build_checkout(tmp_path, campaign_ready=False)

    report = run_baseline_preflight(checkout, fixture)

    assert report.frozen_source_verified
    assert report.archival_fixture_matches
    assert not report.campaign_ready
    assert frozenset(report.blockers) == fixture.expected_archival_blockers
    assert report.warnings == ("leaderboard_receipt_unbound",)


def test_campaign_ready_requires_every_scientific_and_runtime_gate(tmp_path):
    checkout, fixture = _build_checkout(tmp_path, campaign_ready=True)

    report = run_baseline_preflight(checkout, fixture)

    assert report.frozen_source_verified
    assert report.archival_fixture_matches
    assert report.campaign_ready
    assert report.blockers == ()
    assert report.warnings == ("leaderboard_receipt_unbound",)


def test_bound_source_file_tampering_fails_closed(tmp_path):
    checkout, fixture = _build_checkout(tmp_path, campaign_ready=True)
    _write(checkout / "src/config.py", 'SCORER_PROFILE = "wrong"\n')

    report = run_baseline_preflight(checkout, fixture)

    assert not report.frozen_source_verified
    assert not report.archival_fixture_matches
    assert not report.campaign_ready
    assert "source_file_hash_mismatch" in report.blockers
    assert "scorer_source_contract_mismatch" in report.blockers
