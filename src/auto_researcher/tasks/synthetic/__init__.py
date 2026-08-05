"""Deterministic synthetic reference task."""

from auto_researcher.tasks.synthetic.task import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)
from auto_researcher.tasks.synthetic.openevolve import (
    SyntheticEvolvableComponent,
    default_synthetic_openevolve_configuration,
)

__all__ = [
    "SyntheticTask",
    "default_synthetic_configuration",
    "default_synthetic_contract",
    "SyntheticEvolvableComponent",
    "default_synthetic_openevolve_configuration",
]
