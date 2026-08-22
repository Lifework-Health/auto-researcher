from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import stat
from pathlib import Path

import pytest

from auto_researcher.diagnostics.models import (
    DiagnosticCheckpoint,
    DiagnosticExperiment,
    DiagnosticMethodSpec,
)
from auto_researcher.tasks.feta_seg.metrics import LABELS
from auto_researcher.tasks.feta_seg.manifests import EXPECTED_MANIFEST_HASH
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
)
from auto_researcher.tasks.feta_unet_diagnostics.attribution import (
    captum_capability,
)
from auto_researcher.tasks.feta_unet_diagnostics.comparison import (
    compare_panel_metrics,
    summarise_learning_curve,
)
from auto_researcher.tasks.feta_unet_diagnostics.io import (
    load_protected_experiment_evidence,
)
from auto_researcher.tasks.feta_unet_diagnostics.panel import (
    select_diagnostic_panel,
)
from auto_researcher.tasks.feta_unet_diagnostics.reporting import (
    write_diagnostic_report,
    write_protected_panel,
)
from auto_researcher.tasks.feta_unet_diagnostics.runner import run_metric_sidecar


def _row(
    subject_id: str,
    method: str,
    *,
    macro: float,
    class_offset: float = 0.0,
) -> dict:
    return {
        "subject_id": subject_id,
        "reconstruction_method": method,
        "fold": 0,
        "macro_dice": macro,
        "per_class": {
            str(label): {
                "dice": max(
                    0.0,
                    min(1.0, macro + class_offset + (label - 4) * 0.001),
                )
            }
            for label in LABELS
        },
    }


def _baseline_rows() -> tuple[dict, ...]:
    return tuple(
        _row(f"{method}-{index}", method, macro=0.60 + index * 0.03)
        for method in ("mial", "irtk")
        for index in range(4)
    )


def _checkpoint(experiment_id: str) -> DiagnosticCheckpoint:
    return DiagnosticCheckpoint(
        experiment_id=experiment_id,
        checkpoint_sha256s=("a" * 64,),
        architecture_identity="architecture",
        configuration_identity=("b" if experiment_id == "baseline" else "c") * 64,
        best_epochs=(25,),
    )


def _panel(rows: Sequence[dict]):
    return select_diagnostic_panel(
        rows,
        development_subject_ids={row["subject_id"] for row in rows},
        source_experiment_id="baseline",
        dataset_manifest_hash="manifest",
        split_hash="split",
        fold_hash="fold",
        panel_size=4,
    )


def test_panel_selection_is_balanced_deterministic_and_public_safe():
    rows = _baseline_rows()
    first = _panel(rows)
    second = _panel(tuple(reversed(rows)))

    assert first == second
    assert first.panel_identity == second.panel_identity
    assert [case.reconstruction_method for case in first.cases] == [
        "mial",
        "mial",
        "irtk",
        "irtk",
    ]
    reference = first.public_reference()
    assert reference.subgroup_counts == {"IRTK": 2, "MIAL": 2}
    assert reference.contains_case_identifiers is False
    assert all(
        case.subject_id not in reference.model_dump_json() for case in first.cases
    )


def test_panel_selection_fails_closed_on_holdout_membership():
    rows = _baseline_rows()
    with pytest.raises(ValueError, match="feta_diagnostic_holdout_subject_prohibited"):
        select_diagnostic_panel(
            rows,
            development_subject_ids={row["subject_id"] for row in rows[:-1]},
            source_experiment_id="baseline",
            dataset_manifest_hash="manifest",
            split_hash="split",
            fold_hash="fold",
            panel_size=4,
        )


def test_comparison_reports_improvements_displacement_and_complementarity():
    baseline_rows = _baseline_rows()
    panel = _panel(baseline_rows)
    candidate_a = [dict(row) for row in baseline_rows]
    candidate_b = [dict(row) for row in baseline_rows]
    for row in candidate_a:
        row["macro_dice"] += 0.02
        row["per_class"] = {
            label: {"dice": value["dice"] + (0.02 if label != "7" else -0.02)}
            for label, value in row["per_class"].items()
        }
    for row in candidate_b:
        row["macro_dice"] += 0.01
        row["per_class"] = {
            label: {"dice": value["dice"] + (0.03 if label == "7" else 0.0)}
            for label, value in row["per_class"].items()
        }
    experiment = DiagnosticExperiment(
        diagnostic_id="diagnostic",
        task_id="feta_unet_search",
        task_version="1.0",
        baseline=_checkpoint("baseline"),
        candidates=(_checkpoint("candidate-a"), _checkpoint("candidate-b")),
        panel=panel.public_reference(),
        methods=(
            DiagnosticMethodSpec(method="per_class_error", version="v1"),
            DiagnosticMethodSpec(method="learning_curve", version="v1"),
        ),
        target_labels=LABELS,
    )

    result = compare_panel_metrics(
        experiment,
        panel,
        baseline_rows=baseline_rows,
        candidate_rows={
            "candidate-a": candidate_a,
            "candidate-b": candidate_b,
        },
        learning_curves={
            "baseline": [
                {"epoch": 5, "validation_score": 0.4},
                {"epoch": 25, "validation_score": 0.7},
            ],
            "candidate-a": [
                {"epoch": 5, "validation_score": 0.5},
                {"epoch": 25, "validation_score": 0.75},
            ],
            "candidate-b": [
                {"epoch": 5, "validation_score": 0.6},
                {"epoch": 25, "validation_score": 0.74},
            ],
        },
    )

    assert result.success is True
    assert result.aggregate["observation_count"] == 6
    assert result.aggregate["contains_case_identifiers"] is False
    payload = result.model_dump_json()
    assert all(row["subject_id"] not in payload for row in baseline_rows)
    candidate_observation = next(
        observation
        for observation in result.observations
        if observation.model_experiment_ids == ("baseline", "candidate-a")
    )
    assert candidate_observation.metrics["error_displacement_case_count"] == 4
    complementarity = next(
        observation
        for observation in result.observations
        if observation.method == "feta-panel-complementarity-v1"
    )
    assert complementarity.metrics["complementary_advantage_observed"] is True


def test_learning_curve_validation_and_optional_captum_capability():
    summary = summarise_learning_curve(
        [
            {"epoch": 5, "validation_score": 0.4},
            {"epoch": 10, "validation_score": 0.55},
        ]
    )
    assert summary["best_epoch"] == 10
    assert summary["score_gain"] == pytest.approx(0.15)
    with pytest.raises(ValueError, match="feta_diagnostic_learning_curve_invalid"):
        summarise_learning_curve(
            [
                {"epoch": 10, "validation_score": 0.55},
                {"epoch": 5, "validation_score": 0.4},
            ]
        )
    capability = captum_capability()
    assert capability["backend_id"] == "captum"
    assert isinstance(capability["available"], bool)


def _protected_experiment_root(
    tmp_path: Path, rows: Sequence[dict], experiment_id: str = "experiment-candidate"
) -> Path:
    root = tmp_path / experiment_id
    checkpoint = root / "checkpoints" / "fold-0" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fold_root = root / "fold-results"
    fold_root.mkdir()
    (fold_root / "fold-0.json").write_text(
        json.dumps(
            {
                "schema_version": "feta-unet-direct-fold-result-v1",
                "identity": {
                    "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
                    "split_hash": EXPECTED_SPLIT_HASH,
                    "fold_hash": EXPECTED_FOLD_HASH,
                    "configuration_hash": "d" * 64,
                    "fold": 0,
                },
                "result": {
                    "architecture_identity": "architecture",
                    "subject_metrics": list(rows),
                    "validation_history": [
                        {"epoch": 5, "validation_score": 0.4},
                        {"epoch": 25, "validation_score": 0.7},
                    ],
                    "checkpoint": {
                        "relative_path": "fold-0/best.pt",
                        "sha256": checkpoint_hash,
                        "size_bytes": checkpoint.stat().st_size,
                    },
                    "best_epoch": 25,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_protected_loader_verifies_checkpoint_and_report_excludes_subjects(tmp_path):
    baseline_rows = _baseline_rows()
    root = _protected_experiment_root(tmp_path, baseline_rows)
    evidence = load_protected_experiment_evidence(root)
    assert evidence.checkpoint.experiment_id == "experiment-candidate"
    assert evidence.checkpoint.best_epochs == (25,)
    assert len(evidence.subject_metrics) == 8

    panel = _panel(baseline_rows)
    protected_path = tmp_path / "protected" / "panel.json"
    write_protected_panel(panel, protected_path)
    assert stat.S_IMODE(protected_path.stat().st_mode) == 0o600
    assert "mial-0" in protected_path.read_text(encoding="utf-8")

    experiment = DiagnosticExperiment(
        diagnostic_id="diagnostic",
        task_id="feta_unet_search",
        task_version="1.0",
        baseline=_checkpoint("baseline"),
        candidates=(_checkpoint("candidate-a"),),
        panel=panel.public_reference(),
        methods=(DiagnosticMethodSpec(method="per_class_error", version="v1"),),
        target_labels=LABELS,
    )
    result = compare_panel_metrics(
        experiment,
        panel,
        baseline_rows=baseline_rows,
        candidate_rows={"candidate-a": baseline_rows},
    )
    report_dir = tmp_path / "public-report"
    manifest = write_diagnostic_report(
        experiment=experiment,
        result=result,
        report_dir=report_dir,
    )
    assert manifest["contains_case_identifiers"] is False
    public_payload = "".join(
        path.read_text(encoding="utf-8")
        for path in report_dir.iterdir()
        if path.is_file()
    )
    assert all(row["subject_id"] not in public_payload for row in baseline_rows)
    assert (report_dir / "SHA256SUMS").is_file()


def test_protected_loader_detects_checkpoint_tampering(tmp_path):
    root = _protected_experiment_root(tmp_path, _baseline_rows())
    (root / "checkpoints" / "fold-0" / "best.pt").write_bytes(b"tampered")
    with pytest.raises(
        ValueError, match="feta_diagnostic_checkpoint_identity_mismatch"
    ):
        load_protected_experiment_evidence(root)


def test_metric_sidecar_writes_protected_and_public_outputs(tmp_path):
    rows = _baseline_rows()
    baseline_root = _protected_experiment_root(tmp_path, rows, "baseline")
    candidate_rows = [dict(row) for row in rows]
    for row in candidate_rows:
        row["macro_dice"] += 0.01
        row["per_class"] = {
            label: {"dice": value["dice"] + 0.01}
            for label, value in row["per_class"].items()
        }
    candidate_root = _protected_experiment_root(tmp_path, candidate_rows, "candidate")

    manifest = run_metric_sidecar(
        diagnostic_id="diagnostic",
        baseline_root=baseline_root,
        candidate_roots=(candidate_root,),
        development_subject_ids={row["subject_id"] for row in rows},
        report_dir=tmp_path / "public",
        protected_panel_path=tmp_path / "protected" / "panel.json",
        panel_size=4,
    )

    assert manifest["diagnostic_id"] == "diagnostic"
    assert (tmp_path / "protected" / "panel.json").is_file()
    assert (tmp_path / "public" / "diagnostic-report.md").is_file()
