"""Fail-closed source and evidence preflight for Viet's VCC 2026 baseline."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from auto_researcher.benchmarks.vcc2026.models import (
    BaselineFixture,
    BaselinePreflightReport,
    PreflightCheck,
)

_V1_DEPENDENCY = re.compile(r"^\s*cell-eval(?:\s|[<>=!~])", re.MULTILINE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(checkout: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_head(checkout: Path) -> str | None:
    value = _git_output(checkout, "rev-parse", "HEAD")
    return value if value and re.fullmatch(r"[0-9a-f]{40}", value) else None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_baseline_preflight(
    checkout: Path,
    fixture: BaselineFixture,
) -> BaselinePreflightReport:
    """Verify frozen source identity and decide whether campaign use is allowed.

    A matching archival fixture may intentionally remain campaign-blocked. This
    distinction lets Auto Researcher preserve Viet's submitted baseline exactly
    while refusing to launch JEPA search until scorer, decisions, guardrails,
    data and submission payloads are all bound.
    """

    root = checkout.resolve()
    checks: list[PreflightCheck] = []

    def record(code: str, passed: bool, message: str, *, blocker: bool = True) -> None:
        checks.append(
            PreflightCheck(
                code=code,
                passed=passed,
                message=message,
                blocker=blocker,
            )
        )

    head = _git_head(root)
    record(
        "source_commit_mismatch",
        head == fixture.source_commit,
        "source checkout is pinned to the frozen Viet baseline commit"
        if head == fixture.source_commit
        else "source checkout does not match the frozen Viet baseline commit",
    )
    remote = _git_output(root, "remote", "get-url", "origin")
    expected_remotes = {
        fixture.source_repository,
        f"{fixture.source_repository}.git",
        "git@github.com:v-iettran/vcc2026.git",
    }
    record(
        "source_repository_mismatch",
        remote in expected_remotes,
        "source checkout origin matches Viet's frozen repository"
        if remote in expected_remotes
        else "source checkout origin does not match Viet's frozen repository",
    )
    worktree_status = _git_output(
        root, "status", "--porcelain", "--untracked-files=all"
    )
    worktree_clean = worktree_status == ""
    record(
        "source_worktree_dirty",
        worktree_clean,
        "source checkout has no tracked or untracked changes"
        if worktree_clean
        else "source checkout has tracked or untracked changes",
    )

    file_hashes_match = True
    for binding in fixture.files:
        path = root / binding.path
        passed = path.is_file() and _sha256(path) == binding.sha256
        file_hashes_match = file_hashes_match and passed
        record(
            "source_file_hash_mismatch",
            passed,
            f"{binding.path} matches its frozen SHA-256"
            if passed
            else f"{binding.path} is missing or differs from its frozen SHA-256",
        )

    requirements_path = root / "requirements.txt"
    requirements = (
        requirements_path.read_text(encoding="utf-8")
        if requirements_path.is_file()
        else ""
    )
    pinned_requirement = f"cell-eval2=={fixture.scorer.version}"
    scorer_dependency_ok = (
        pinned_requirement in requirements
        and _V1_DEPENDENCY.search(requirements) is None
    )
    record(
        "scorer_dependency_not_pinned",
        scorer_dependency_ok,
        f"requirements pin {pinned_requirement} and exclude cell-eval v1"
        if scorer_dependency_ok
        else f"requirements must pin {pinned_requirement} and remove cell-eval v1",
    )

    config_text = (root / "src/config.py").read_text(encoding="utf-8")
    score_text = (root / "src/eval/score.py").read_text(encoding="utf-8")
    scorer_source_ok = (
        'SCORER_PROFILE = "vcc2026"' in config_text
        and 'version("cell-eval2")' in score_text
        and "cell-eval` v1 is a different package" in score_text
    )
    record(
        "scorer_source_contract_mismatch",
        scorer_source_ok,
        "source code uses cell-eval2 with the vcc2026 profile"
        if scorer_source_ok
        else "source code does not expose the expected cell-eval2 vcc2026 contract",
    )

    stale_phrases = (
        "The scorer is a version behind",
        "cell-eval 0.8.2 is the VCC-2025 metric set",
    )
    reporting_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "README.md", root / "results/README.md")
        if path.is_file()
    )
    reporting_current = not any(phrase in reporting_text for phrase in stale_phrases)
    record(
        "stale_scorer_reporting",
        reporting_current,
        "reporting agrees with the frozen 2026 scorer"
        if reporting_current
        else "reporting still claims the obsolete 2025 scorer is authoritative",
    )

    decisions_path = root / "results/decisions.md"
    decisions_text = (
        decisions_path.read_text(encoding="utf-8") if decisions_path.is_file() else ""
    )
    decisions_complete = (
        bool(decisions_text.strip()) and "Not yet computed" not in decisions_text
    )
    record(
        "scientific_decisions_uncomputed",
        decisions_complete,
        "decision rules D1-D7 are materialised"
        if decisions_complete
        else "decision rules D1-D7 have not been computed",
    )

    guardrails = _read_json(root / "results/guardrails.json") or {}
    unresolved_guardrails = tuple(
        code
        for code in fixture.required_guardrails
        if not isinstance(guardrails.get(code), dict)
        or guardrails[code].get("ok") is not True
    )
    record(
        "scientific_guardrails_unresolved",
        not unresolved_guardrails,
        "all required scientific guardrails are true"
        if not unresolved_guardrails
        else "unresolved guardrails: " + ", ".join(unresolved_guardrails),
    )

    submission_report = _read_json(
        root / "results/submission/submission_B2_report.json"
    )
    expected_contexts = set(fixture.submission.contexts)
    observed_contexts = (
        set(submission_report.get("contexts", {})) if submission_report else set()
    )
    context_counts_ok = bool(submission_report) and all(
        submission_report["contexts"][context].get("n_perturbations")
        == fixture.submission.perturbations_per_context
        for context in expected_contexts
        if context in submission_report["contexts"]
    )
    submission_report_ok = bool(submission_report) and all(
        (
            submission_report.get("baseline") == fixture.submission.baseline,
            submission_report.get("alpha") == fixture.submission.alpha,
            submission_report.get("n_cells") == fixture.submission.n_cells,
            submission_report.get("n_genes") == fixture.submission.n_genes,
            submission_report.get("has_control_rows")
            == fixture.submission.has_control_rows,
            observed_contexts == expected_contexts,
            context_counts_ok,
            submission_report.get("vcc_prep", {}).get("passed") is True,
            submission_report.get("vcc_prep", {}).get("returncode") == 0,
        )
    )
    record(
        "submission_report_invalid",
        submission_report_ok,
        "B2 submission report has the frozen dimensions and passed vcc prep"
        if submission_report_ok
        else "B2 submission report is missing, malformed or inconsistent",
    )

    dataset_bound = fixture.runtime_bindings.dataset_manifest_sha256 is not None
    record(
        "runtime_dataset_manifest_unbound",
        dataset_bound,
        "runtime dataset manifest is SHA-256 bound"
        if dataset_bound
        else "runtime dataset manifest has not been supplied and SHA-256 bound",
    )
    payloads_bound = all(
        (
            fixture.runtime_bindings.submission_h5ad_sha256,
            fixture.runtime_bindings.submission_vcc_sha256,
        )
    )
    record(
        "submission_payload_unbound",
        payloads_bound,
        "H5AD and VCC submission payloads are SHA-256 bound"
        if payloads_bound
        else "H5AD and VCC submission payload hashes are not yet bound",
    )

    policy_ok = fixture.leaderboard_event.tuning_signal_allowed is False
    record(
        "leaderboard_tuning_policy_invalid",
        policy_ok,
        "leaderboard event is immutable external evidence and not a tuning signal",
    )
    receipt_bound = fixture.leaderboard_event.receipt_sha256 is not None
    record(
        "leaderboard_receipt_unbound",
        receipt_bound,
        "leaderboard receipt is SHA-256 bound"
        if receipt_bound
        else "rank 82 remains user-reported until a receipt is supplied",
        blocker=False,
    )

    blockers = tuple(
        sorted({check.code for check in checks if check.blocker and not check.passed})
    )
    warnings = tuple(
        sorted(
            {check.code for check in checks if not check.blocker and not check.passed}
        )
    )
    archival_fixture_matches = frozenset(blockers) == fixture.expected_archival_blockers
    frozen_source_verified = all(
        (
            head == fixture.source_commit,
            remote in expected_remotes,
            worktree_clean,
            file_hashes_match,
            scorer_source_ok,
            submission_report_ok,
            policy_ok,
            archival_fixture_matches,
        )
    )
    return BaselinePreflightReport(
        schema_version="vcc2026-baseline-preflight-v1",
        baseline_id=fixture.baseline_id,
        source_commit=head,
        frozen_source_verified=frozen_source_verified,
        archival_fixture_matches=archival_fixture_matches,
        campaign_ready=not blockers,
        checks=tuple(checks),
        blockers=blockers,
        warnings=warnings,
    )
