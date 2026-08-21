"""Immutable contracts that keep diagnostic observations and interpretations apart."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from auto_researcher.contracts.models import FrozenJsonDict


class DiagnosticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DiagnosticCheckpoint(DiagnosticModel):
    experiment_id: str = Field(min_length=1)
    checkpoint_sha256s: tuple[str, ...] = Field(min_length=1)
    architecture_identity: str = Field(min_length=1)
    configuration_identity: str = Field(min_length=64, max_length=64)
    best_epochs: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def checkpoint_set_is_valid(self) -> "DiagnosticCheckpoint":
        if len(self.checkpoint_sha256s) != len(self.best_epochs):
            raise ValueError("diagnostic_checkpoint_fold_count_mismatch")
        if any(len(value) != 64 for value in self.checkpoint_sha256s):
            raise ValueError("diagnostic_checkpoint_sha256_invalid")
        if any(epoch < 1 for epoch in self.best_epochs):
            raise ValueError("diagnostic_checkpoint_epoch_invalid")
        return self


class DiagnosticPanelReference(DiagnosticModel):
    panel_identity: str = Field(min_length=64, max_length=64)
    dataset_manifest_hash: str = Field(min_length=1)
    split_hash: str = Field(min_length=1)
    fold_hash: str = Field(min_length=1)
    case_count: int = Field(ge=1)
    subgroup_counts: FrozenJsonDict
    contains_case_identifiers: bool = False

    @model_validator(mode="after")
    def identifiers_must_remain_protected(self) -> "DiagnosticPanelReference":
        if self.contains_case_identifiers:
            raise ValueError("diagnostic_public_panel_contains_case_identifiers")
        return self


class DiagnosticMethodSpec(DiagnosticModel):
    method: str = Field(min_length=1)
    version: str = Field(min_length=1)
    parameters: FrozenJsonDict = Field(default_factory=dict)


class DiagnosticExperiment(DiagnosticModel):
    diagnostic_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    baseline: DiagnosticCheckpoint
    candidates: tuple[DiagnosticCheckpoint, ...] = Field(min_length=1)
    panel: DiagnosticPanelReference
    methods: tuple[DiagnosticMethodSpec, ...] = Field(min_length=1)
    target_labels: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def checkpoint_identities_are_unique(self) -> "DiagnosticExperiment":
        identities = [
            self.baseline.experiment_id,
            *(candidate.experiment_id for candidate in self.candidates),
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("diagnostic_checkpoint_identity_duplicate")
        if len(self.target_labels) != len(set(self.target_labels)):
            raise ValueError("diagnostic_target_label_duplicate")
        return self


class DiagnosticObservation(DiagnosticModel):
    observation_id: str = Field(min_length=1)
    diagnostic_id: str = Field(min_length=1)
    method: str = Field(min_length=1)
    model_experiment_ids: tuple[str, ...] = Field(min_length=1)
    target_label: int | None = None
    subgroup: str | None = None
    metrics: FrozenJsonDict
    artefact_references: tuple[str, ...] = ()


class DiagnosticInterpretation(DiagnosticModel):
    interpretation_id: str = Field(min_length=1)
    diagnostic_id: str = Field(min_length=1)
    evidence_observation_ids: tuple[str, ...] = Field(min_length=1)
    statement: str = Field(min_length=1)
    proposed_action: str | None = None


class DiagnosticResult(DiagnosticModel):
    schema_version: str = "diagnostic-result-v1"
    diagnostic_id: str = Field(min_length=1)
    success: bool
    observations: tuple[DiagnosticObservation, ...] = ()
    aggregate: FrozenJsonDict = Field(default_factory=dict)
    artefact_references: tuple[str, ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def success_and_error_are_consistent(self) -> "DiagnosticResult":
        if self.success and self.error is not None:
            raise ValueError("successful_diagnostic_result_has_error")
        if not self.success and self.error is None:
            raise ValueError("failed_diagnostic_result_missing_error")
        return self
