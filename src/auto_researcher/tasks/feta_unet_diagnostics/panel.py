"""Deterministic, protected development-case selection for FeTA diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from auto_researcher.diagnostics.models import DiagnosticPanelReference
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_seg.metrics import LABELS

PANEL_SCHEMA_VERSION = "feta-unet-diagnostic-panel-v1"


class ProtectedPanelModel(BaseModel):
    """A protected-runtime model that may contain development subject IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FeTADiagnosticPanelCase(ProtectedPanelModel):
    subject_id: str = Field(min_length=1)
    reconstruction_method: Literal["mial", "irtk"]
    fold: int = Field(ge=0, le=4)
    selection_reasons: tuple[str, ...] = Field(min_length=1)


class FeTADiagnosticPanel(ProtectedPanelModel):
    schema_version: Literal["feta-unet-diagnostic-panel-v1"] = PANEL_SCHEMA_VERSION
    source_experiment_id: str = Field(min_length=1)
    dataset_manifest_hash: str = Field(min_length=1)
    split_hash: str = Field(min_length=1)
    fold_hash: str = Field(min_length=1)
    cases: tuple[FeTADiagnosticPanelCase, ...] = Field(min_length=2)
    holdout_subject_count: Literal[0] = 0

    @model_validator(mode="after")
    def cases_are_unique_and_balanced(self) -> "FeTADiagnosticPanel":
        subject_ids = [case.subject_id for case in self.cases]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("feta_diagnostic_panel_subject_duplicate")
        counts = Counter(case.reconstruction_method for case in self.cases)
        if set(counts) != {"mial", "irtk"} or abs(counts["mial"] - counts["irtk"]) > 1:
            raise ValueError("feta_diagnostic_panel_reconstruction_imbalance")
        return self

    @property
    def panel_identity(self) -> str:
        return payload_hash(self)

    def public_reference(self) -> DiagnosticPanelReference:
        counts = Counter(case.reconstruction_method for case in self.cases)
        return DiagnosticPanelReference(
            panel_identity=self.panel_identity,
            dataset_manifest_hash=self.dataset_manifest_hash,
            split_hash=self.split_hash,
            fold_hash=self.fold_hash,
            case_count=len(self.cases),
            subgroup_counts={key.upper(): counts[key] for key in sorted(counts)},
            contains_case_identifiers=False,
        )


def _validated_row(row: dict[str, Any]) -> dict[str, Any]:
    subject_id = row.get("subject_id")
    method = row.get("reconstruction_method")
    fold = row.get("fold")
    per_class = row.get("per_class")
    if not isinstance(subject_id, str) or not subject_id:
        raise ValueError("feta_diagnostic_subject_identity_invalid")
    if not isinstance(method, str) or method.casefold() not in {"mial", "irtk"}:
        raise ValueError("feta_diagnostic_reconstruction_method_invalid")
    row["reconstruction_method"] = method.casefold()
    if not isinstance(fold, int) or not 0 <= fold <= 4:
        raise ValueError("feta_diagnostic_fold_invalid")
    if not isinstance(per_class, dict) or set(map(int, per_class)) != set(LABELS):
        raise ValueError("feta_diagnostic_tissue_metrics_incomplete")
    for label in LABELS:
        value = per_class.get(str(label), per_class.get(label))
        if not isinstance(value, dict):
            raise ValueError("feta_diagnostic_tissue_metrics_invalid")
        dice = value.get("dice")
        if not isinstance(dice, (int, float)) or not math.isfinite(float(dice)):
            raise ValueError("feta_diagnostic_tissue_metrics_invalid")
        if not 0.0 <= float(dice) <= 1.0:
            raise ValueError("feta_diagnostic_tissue_metrics_invalid")
    macro = row.get("macro_dice")
    if not isinstance(macro, (int, float)) or not 0.0 <= float(macro) <= 1.0:
        raise ValueError("feta_diagnostic_macro_dice_invalid")
    return row


def _dice(row: dict[str, Any], label: int) -> float:
    per_class = row["per_class"]
    value = per_class.get(str(label), per_class.get(label))
    return float(value["dice"])


def _select_method_cases(
    rows: Sequence[dict[str, Any]], quota: int
) -> tuple[FeTADiagnosticPanelCase, ...]:
    reasons: dict[str, list[str]] = {}
    selected: list[dict[str, Any]] = []

    def add(row: dict[str, Any], reason: str) -> None:
        subject_id = str(row["subject_id"])
        if subject_id in reasons:
            reasons[subject_id].append(reason)
            return
        if len(selected) < quota:
            selected.append(row)
            reasons[subject_id] = [reason]

    ordered_macro = sorted(
        rows, key=lambda row: (float(row["macro_dice"]), row["subject_id"])
    )
    add(ordered_macro[0], "lowest_macro_dice")
    for label in LABELS:
        ordered_label = sorted(
            rows, key=lambda row: (_dice(row, label), row["subject_id"])
        )
        add(ordered_label[0], f"lowest_{label}_dice")
    for row in ordered_macro:
        add(row, "low_macro_dice_fill")
    if len(selected) != quota:
        raise ValueError("feta_diagnostic_panel_quota_unavailable")
    return tuple(
        FeTADiagnosticPanelCase(
            subject_id=str(row["subject_id"]),
            reconstruction_method=row["reconstruction_method"],
            fold=int(row["fold"]),
            selection_reasons=tuple(reasons[str(row["subject_id"])]),
        )
        for row in selected
    )


def select_diagnostic_panel(
    subject_metrics: Sequence[dict[str, Any]],
    *,
    development_subject_ids: Iterable[str],
    source_experiment_id: str,
    dataset_manifest_hash: str,
    split_hash: str,
    fold_hash: str,
    panel_size: int = 12,
) -> FeTADiagnosticPanel:
    """Select a balanced hard-case panel without admitting holdout subjects."""

    if panel_size < 4 or panel_size % 2:
        raise ValueError("feta_diagnostic_panel_size_invalid")
    development = set(development_subject_ids)
    rows = tuple(_validated_row(dict(row)) for row in subject_metrics)
    subject_ids = [str(row["subject_id"]) for row in rows]
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("feta_diagnostic_subject_metric_duplicate")
    if not set(subject_ids).issubset(development):
        raise ValueError("feta_diagnostic_holdout_subject_prohibited")
    quota = panel_size // 2
    selected: list[FeTADiagnosticPanelCase] = []
    for method in ("mial", "irtk"):
        method_rows = tuple(
            row for row in rows if row["reconstruction_method"] == method
        )
        if len(method_rows) < quota:
            raise ValueError("feta_diagnostic_panel_quota_unavailable")
        selected.extend(_select_method_cases(method_rows, quota))
    return FeTADiagnosticPanel(
        source_experiment_id=source_experiment_id,
        dataset_manifest_hash=dataset_manifest_hash,
        split_hash=split_hash,
        fold_hash=fold_hash,
        cases=tuple(selected),
        holdout_subject_count=0,
    )
