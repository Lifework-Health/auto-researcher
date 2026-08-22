"""Metric-sidecar entry point for comparing completed FeTA U-Net experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from auto_researcher.diagnostics.models import (
    DiagnosticExperiment,
    DiagnosticMethodSpec,
)
from auto_researcher.tasks.feta_seg.manifests import (
    EXPECTED_MANIFEST_HASH,
    inspect_subjects,
    manifest_hash,
)
from auto_researcher.tasks.feta_seg.metrics import LABELS
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    locked_partition,
)
from auto_researcher.tasks.feta_unet_diagnostics.comparison import (
    compare_panel_metrics,
)
from auto_researcher.tasks.feta_unet_diagnostics.io import (
    ProtectedExperimentEvidence,
    load_protected_experiment_evidence,
)
from auto_researcher.tasks.feta_unet_diagnostics.panel import (
    select_diagnostic_panel,
)
from auto_researcher.tasks.feta_unet_diagnostics.reporting import (
    write_diagnostic_report,
    write_protected_panel,
)


def _ensure_common_scientific_scope(
    baseline: ProtectedExperimentEvidence,
    candidates: Sequence[ProtectedExperimentEvidence],
) -> None:
    expected = (
        baseline.dataset_manifest_hash,
        baseline.split_hash,
        baseline.fold_hash,
    )
    if expected != (
        EXPECTED_MANIFEST_HASH,
        EXPECTED_SPLIT_HASH,
        EXPECTED_FOLD_HASH,
    ):
        raise ValueError("feta_diagnostic_baseline_scope_invalid")
    if any(
        (
            candidate.dataset_manifest_hash,
            candidate.split_hash,
            candidate.fold_hash,
        )
        != expected
        for candidate in candidates
    ):
        raise ValueError("feta_diagnostic_candidate_scope_mismatch")


def run_metric_sidecar(
    *,
    diagnostic_id: str,
    baseline_root: Path,
    candidate_roots: Sequence[Path],
    development_subject_ids: Iterable[str],
    report_dir: Path,
    protected_panel_path: Path,
    panel_size: int = 12,
) -> dict[str, Any]:
    """Create the Wave-1 metric comparison without loading MRI or model weights."""

    if not candidate_roots:
        raise ValueError("feta_diagnostic_candidates_missing")
    resolved_report = report_dir.resolve()
    resolved_panel = protected_panel_path.resolve()
    try:
        resolved_panel.relative_to(resolved_report)
    except ValueError:
        pass
    else:
        raise ValueError("feta_diagnostic_protected_panel_in_public_report")

    baseline = load_protected_experiment_evidence(baseline_root)
    candidates = tuple(
        load_protected_experiment_evidence(root) for root in candidate_roots
    )
    candidate_ids = [candidate.checkpoint.experiment_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("feta_diagnostic_candidate_identity_duplicate")
    _ensure_common_scientific_scope(baseline, candidates)

    panel = select_diagnostic_panel(
        baseline.subject_metrics,
        development_subject_ids=development_subject_ids,
        source_experiment_id=baseline.checkpoint.experiment_id,
        dataset_manifest_hash=baseline.dataset_manifest_hash,
        split_hash=baseline.split_hash,
        fold_hash=baseline.fold_hash,
        panel_size=panel_size,
    )
    experiment = DiagnosticExperiment(
        diagnostic_id=diagnostic_id,
        task_id="feta_unet_search",
        task_version="1.0",
        baseline=baseline.checkpoint,
        candidates=tuple(candidate.checkpoint for candidate in candidates),
        panel=panel.public_reference(),
        methods=(
            DiagnosticMethodSpec(
                method="per_class_error_and_displacement",
                version="feta-panel-error-comparison-v1",
                parameters={"material_dice_delta": 0.01},
            ),
            DiagnosticMethodSpec(
                method="learning_curve",
                version="feta-learning-curve-summary-v1",
            ),
        ),
        target_labels=LABELS,
    )
    result = compare_panel_metrics(
        experiment,
        panel,
        baseline_rows=baseline.subject_metrics,
        candidate_rows={
            candidate.checkpoint.experiment_id: candidate.subject_metrics
            for candidate in candidates
        },
        learning_curves={
            baseline.checkpoint.experiment_id: baseline.validation_history,
            **{
                candidate.checkpoint.experiment_id: candidate.validation_history
                for candidate in candidates
            },
        },
    )
    write_protected_panel(panel, resolved_panel)
    return write_diagnostic_report(
        experiment=experiment,
        result=result,
        report_dir=resolved_report,
    )


def _development_subjects(data_dir: Path) -> tuple[str, ...]:
    subjects = inspect_subjects(data_dir, inspect_labels=False)
    if manifest_hash(subjects) != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_dataset_identity_mismatch")
    partition = locked_partition(
        {subject.subject_id: subject.reconstruction_method for subject in subjects}
    )
    return partition.development


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a public-safe FeTA U-Net metric diagnostic sidecar."
    )
    parser.add_argument("--diagnostic-id", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, action="append", type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--protected-panel", required=True, type=Path)
    parser.add_argument("--panel-size", type=int, default=12)
    arguments = parser.parse_args()
    manifest = run_metric_sidecar(
        diagnostic_id=arguments.diagnostic_id,
        baseline_root=arguments.baseline_root,
        candidate_roots=arguments.candidate_root,
        development_subject_ids=_development_subjects(arguments.data_dir),
        report_dir=arguments.report_dir,
        protected_panel_path=arguments.protected_panel,
        panel_size=arguments.panel_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
