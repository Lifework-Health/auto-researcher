"""Public-safe reporting and protected panel persistence for FeTA diagnostics."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from auto_researcher.diagnostics.models import DiagnosticExperiment, DiagnosticResult
from auto_researcher.tasks.feta_unet_diagnostics.panel import FeTADiagnosticPanel

REPORT_SCHEMA_VERSION = "feta-unet-diagnostic-report-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_protected_panel(panel: FeTADiagnosticPanel, path: Path) -> None:
    """Persist the case-bearing manifest with owner-only permissions."""

    if path.exists():
        raise ValueError("feta_diagnostic_panel_path_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(panel.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _summary_lines(
    experiment: DiagnosticExperiment, result: DiagnosticResult
) -> list[str]:
    lines = [
        "# FeTA U-Net diagnostic sidecar",
        "",
        f"Schema: `{REPORT_SCHEMA_VERSION}`",
        f"Diagnostic: `{experiment.diagnostic_id}`",
        f"Panel: `{experiment.panel.panel_identity}` ({experiment.panel.case_count} cases)",
        f"Baseline: `{experiment.baseline.experiment_id}`",
        "Candidates: "
        + ", ".join(f"`{item.experiment_id}`" for item in experiment.candidates),
        "",
        "## Evidence summary",
        "",
    ]
    for observation in result.observations:
        models = " vs ".join(observation.model_experiment_ids)
        lines.append(f"- `{observation.method}` — {models}")
        metrics: dict[str, Any] = dict(observation.metrics)
        for field in (
            "mean_macro_dice_delta",
            "error_displacement_case_count",
            "complementary_advantage_observed",
            "best_score",
            "score_gain",
        ):
            if field in metrics:
                lines.append(f"  - {field}: `{metrics[field]}`")
    lines.extend(
        (
            "",
            "## Interpretation boundary",
            "",
            "This report contains diagnostic observations only. Scientific interpretation and branch decisions must be recorded separately with references to observation IDs.",
            "",
            "The panel manifest containing development subject identifiers is stored separately in protected runtime storage and is not included in this report.",
            "",
        )
    )
    return lines


def write_diagnostic_report(
    *,
    experiment: DiagnosticExperiment,
    result: DiagnosticResult,
    report_dir: Path,
) -> dict[str, Any]:
    if result.diagnostic_id != experiment.diagnostic_id:
        raise ValueError("feta_diagnostic_result_identity_mismatch")
    if report_dir.exists() and any(report_dir.iterdir()):
        raise ValueError("feta_diagnostic_report_directory_not_empty")
    report_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "diagnostic-experiment.json": experiment.model_dump_json(indent=2) + "\n",
        "diagnostic-result.json": result.model_dump_json(indent=2) + "\n",
        "diagnostic-report.md": "\n".join(_summary_lines(experiment, result)),
    }
    for name, content in files.items():
        (report_dir / name).write_text(content, encoding="utf-8")
    checksums = {name: _sha256(report_dir / name) for name in sorted(files)}
    (report_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "diagnostic_id": experiment.diagnostic_id,
        "panel_identity": experiment.panel.panel_identity,
        "files": [
            {
                "name": name,
                "sha256": _sha256(report_dir / name),
                "size_bytes": (report_dir / name).stat().st_size,
            }
            for name in (*sorted(files), "SHA256SUMS")
        ],
        "contains_case_identifiers": False,
    }
