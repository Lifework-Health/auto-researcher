"""FeTA locked SegResNet task."""

from auto_researcher.tasks.feta_seg.configuration import (
    FeTASegConfiguration,
    baseline_configuration,
    smoke_configuration,
)
from auto_researcher.tasks.feta_seg.task import (
    FeTASegTask,
    default_feta_configuration,
    default_feta_contract,
)

__all__ = [
    "FeTASegConfiguration",
    "FeTASegTask",
    "baseline_configuration",
    "smoke_configuration",
    "default_feta_configuration",
    "default_feta_contract",
]
