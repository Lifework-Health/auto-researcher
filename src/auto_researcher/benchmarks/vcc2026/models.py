"""Typed identities for a frozen VCC 2026 public-benchmark baseline."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
GIT_SHA_PATTERN = r"^[0-9a-f]{40}$"

VCC2026_SCORED_METRICS = (
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
)


class FixtureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileBinding(FixtureModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def path_is_checkout_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value.endswith("/"):
            raise ValueError("file binding paths must be checkout-relative files")
        return value


class ScorerBinding(FixtureModel):
    distribution: Literal["cell-eval2"]
    version: str = Field(min_length=1)
    source_repository: Literal["https://github.com/ArcInstitute/cell-eval2"]
    source_commit: str = Field(pattern=GIT_SHA_PATTERN)
    profile: Literal["vcc2026"]
    scored_metrics: tuple[str, ...]

    @model_validator(mode="after")
    def metrics_are_exact(self) -> "ScorerBinding":
        if self.scored_metrics != VCC2026_SCORED_METRICS:
            raise ValueError("scored_metrics must match the frozen vcc2026 profile")
        return self


class SubmissionBinding(FixtureModel):
    baseline: Literal["B2"]
    alpha: float = Field(gt=0)
    contexts: tuple[Literal["A", "B", "C"], ...]
    perturbations_per_context: int = Field(gt=0)
    cells_per_perturbation: int = Field(gt=0)
    n_cells: int = Field(gt=0)
    n_genes: int = Field(gt=0)
    has_control_rows: bool
    vcc_prep_required: bool = True

    @model_validator(mode="after")
    def dimensions_are_consistent(self) -> "SubmissionBinding":
        expected = (
            len(self.contexts)
            * self.perturbations_per_context
            * self.cells_per_perturbation
        )
        if self.n_cells != expected:
            raise ValueError("n_cells does not match contexts × perturbations × cells")
        if tuple(self.contexts) != ("A", "B", "C"):
            raise ValueError("the frozen submission must contain contexts A, B and C")
        if self.has_control_rows:
            raise ValueError("challenge submissions must not contain control rows")
        return self


class RuntimeBindings(FixtureModel):
    dataset_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    submission_h5ad_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    submission_vcc_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class LeaderboardEvent(FixtureModel):
    rank: int = Field(gt=0)
    evidence_status: Literal["user_reported", "receipt_verified"]
    submitted_baseline: Literal["B2"]
    tuning_signal_allowed: Literal[False]
    receipt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class BaselineFixture(FixtureModel):
    schema_version: Literal["vcc2026-baseline-fixture-v1"]
    challenge_id: Literal["arc-virtual-cell-2026"]
    source_repository: Literal["https://github.com/v-iettran/vcc2026"]
    source_commit: str = Field(pattern=GIT_SHA_PATTERN)
    baseline_id: Literal["viet-b2-shared-delta-rank82"]
    files: tuple[FileBinding, ...] = Field(min_length=1)
    scorer: ScorerBinding
    submission: SubmissionBinding
    runtime_bindings: RuntimeBindings
    leaderboard_event: LeaderboardEvent
    required_guardrails: tuple[str, ...] = Field(min_length=1)
    expected_archival_blockers: frozenset[str]

    @model_validator(mode="after")
    def file_paths_are_unique(self) -> "BaselineFixture":
        paths = [binding.path for binding in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("file binding paths must be unique")
        return self


class PreflightCheck(FixtureModel):
    code: str = Field(min_length=1)
    passed: bool
    message: str = Field(min_length=1)
    blocker: bool = True


class BaselinePreflightReport(FixtureModel):
    schema_version: Literal["vcc2026-baseline-preflight-v1"]
    baseline_id: str
    source_commit: str | None
    frozen_source_verified: bool
    archival_fixture_matches: bool
    campaign_ready: bool
    checks: tuple[PreflightCheck, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


def load_baseline_fixture(path: Path) -> BaselineFixture:
    """Load one immutable checked-in baseline fixture."""

    return BaselineFixture.model_validate_json(path.read_text(encoding="utf-8"))


def dump_report(report: BaselinePreflightReport) -> str:
    """Return stable strict JSON for operator evidence and tests."""

    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
