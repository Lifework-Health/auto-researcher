"""Immutable contracts for generic research task plugins."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from auto_researcher.contracts.enums import EvidenceStatus, ProvenanceKind, SearchType
from auto_researcher.contracts.models import FrozenDict, FrozenJsonDict


def _freeze_string_dict(value: dict[str, str]) -> FrozenDict:
    return FrozenDict(value)


FrozenStringDict = Annotated[
    dict[str, str],
    AfterValidator(_freeze_string_dict),
]


class TaskModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class TaskDescriptor(TaskModel):
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    description: str = Field(min_length=1)
    supported_search_types: frozenset[SearchType] = Field(min_length=1)
    evaluator_id: str = Field(min_length=1)
    verification_policy_id: str = Field(min_length=1)
    configuration_schema_version: str = Field(min_length=1)


class ReadinessCheck(TaskModel):
    code: str = Field(min_length=1)
    passed: bool
    message: str = Field(min_length=1)


class ReadinessResult(TaskModel):
    ready: bool
    checks: tuple[ReadinessCheck, ...]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def ready_requires_passing_checks(self) -> "ReadinessResult":
        if self.ready and (self.errors or any(not check.passed for check in self.checks)):
            raise ValueError("a ready task cannot contain failed checks or errors")
        return self


class ExperimentMetadata(TaskModel):
    evaluator_id: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    provenance: ProvenanceKind


class DatasetManifest(TaskModel):
    task_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    files: tuple[str, ...]
    hashes: FrozenStringDict
    loader_version: str = Field(min_length=1)
    created_at: datetime
    metadata: FrozenJsonDict = Field(default_factory=dict)


class ArtefactPolicy(TaskModel):
    allowed_artefact_types: frozenset[str]
    prohibited_artefact_types: frozenset[str]
    contains_sensitive_data: bool
    retention_notes: str = Field(min_length=1)


class TaskRuntimeContext(TaskModel):
    """Runtime-only paths and options; this model never enters ResearchState."""

    run_id: str | None = None
    data_dir: Path | None = None
    workspace_dir: Path | None = None
    output_dir: Path | None = None
    environment: FrozenStringDict = Field(default_factory=dict)
    task_options: FrozenJsonDict = Field(default_factory=dict)
    manifest_created_at: datetime | None = None

    @field_validator("run_id")
    @classmethod
    def run_id_must_be_a_safe_path_segment(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if (
            not value
            or value in {".", ".."}
            or any(character in value for character in ("/", "\\", "\0"))
        ):
            raise ValueError("run_id must be a non-empty path-safe segment")
        return value


class PolicyDecision(TaskModel):
    constraint_compliant: bool
    evidence_status: EvidenceStatus
    reasons: tuple[str, ...] = ()


class TaskPluginError(RuntimeError):
    """Base class for task selection and readiness failures."""


class UnknownTaskError(TaskPluginError):
    pass


class DuplicateTaskError(TaskPluginError):
    pass


class TaskNotReadyError(TaskPluginError):
    pass
