"""Iris weighted k-NN real-data benchmark task."""

from auto_researcher.tasks.iris_knn.configuration import (
    IrisKNNConfiguration,
    baseline_configuration,
)
from auto_researcher.tasks.iris_knn.openevolve import (
    IrisKNNEvolvableComponent,
    default_iris_openevolve_configuration,
)
from auto_researcher.tasks.iris_knn.task import (
    IrisKNNTask,
    default_iris_configuration,
    default_iris_contract,
)

__all__ = [
    "IrisKNNConfiguration",
    "IrisKNNEvolvableComponent",
    "IrisKNNTask",
    "baseline_configuration",
    "default_iris_configuration",
    "default_iris_contract",
    "default_iris_openevolve_configuration",
]
