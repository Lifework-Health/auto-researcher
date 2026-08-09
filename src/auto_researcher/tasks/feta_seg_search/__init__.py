"""FeTA bounded fold-0 SegResNet search task."""

from auto_researcher.tasks.feta_seg_search.configuration import (
    FeTASegSearchConfiguration,
    baseline_search_configuration,
    normalise_search_configuration,
    validation_epochs,
)
from auto_researcher.tasks.feta_seg_search.task import (
    FeTASegSearchTask,
    default_feta_search_configuration,
    default_feta_search_contract,
)

__all__ = [
    "FeTASegSearchConfiguration",
    "FeTASegSearchTask",
    "baseline_search_configuration",
    "default_feta_search_configuration",
    "default_feta_search_contract",
    "normalise_search_configuration",
    "validation_epochs",
]
