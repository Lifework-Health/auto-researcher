"""Generic research task plugin API."""

from auto_researcher.tasks.models import (
    ArtefactPolicy,
    DatasetManifest,
    ExperimentMetadata,
    PolicyDecision,
    ReadinessCheck,
    ReadinessResult,
    TaskDescriptor,
    TaskRuntimeContext,
)
from auto_researcher.tasks.protocols import ResearchTask, VerificationPolicy
from auto_researcher.tasks.registry import TaskRegistry, default_task_registry

__all__ = [
    "ArtefactPolicy",
    "DatasetManifest",
    "ExperimentMetadata",
    "PolicyDecision",
    "ReadinessCheck",
    "ReadinessResult",
    "ResearchTask",
    "TaskDescriptor",
    "TaskRegistry",
    "TaskRuntimeContext",
    "VerificationPolicy",
    "default_task_registry",
]
