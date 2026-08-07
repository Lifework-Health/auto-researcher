"""Safe aggregate-only agent context for future Iris search."""

from auto_researcher.agents.models import TaskAgentContext
from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.search.protocols import SearchCapability
from auto_researcher.tasks.iris_knn.configuration import (
    baseline_configuration,
    configuration_schema,
)
from auto_researcher.tasks.iris_knn.manifests import (
    CLASS_NAMES,
    DATASET_VERSION,
    FEATURE_NAMES,
    FOLD_VERSION,
)


def create_iris_agent_context(
    contract: ResearchContract,
    search_capabilities: dict[SearchType, SearchCapability],
) -> TaskAgentContext:
    available = tuple(
        sorted(
            (
                item
                for item, capability in search_capabilities.items()
                if capability.available and item in contract.allowed_search_types
            ),
            key=lambda item: item.value,
        )
    )
    return TaskAgentContext(
        task_id="iris_knn",
        task_version="1.0",
        display_name="Iris Weighted k-NN Benchmark",
        domain="biology",
        task_description="Classify three Iris species from four flower measurements using fixed stratified folds.",
        safe_scientific_vocabulary=(
            *FEATURE_NAMES,
            *CLASS_NAMES,
            "balanced accuracy",
            "weighted k-nearest neighbours",
        ),
        primary_metric_description="mean_balanced_accuracy is the mean species-balanced validation accuracy across five fixed folds",
        scientific_constraint_summary=(
            "feature weights in [0.1, 4.0]",
            "k in {1,3,5,7,9}",
            "distance power in {1,2}",
        ),
        dataset_summary={
            "dataset_version": DATASET_VERSION,
            "fold_version": FOLD_VERSION,
            "row_count": 150,
            "feature_names": list(FEATURE_NAMES),
            "class_names": list(CLASS_NAMES),
            "contains_patient_data": False,
        },
        available_search_types=available,
        direct_configuration_schema=configuration_schema(),
        optuna_space_summary=configuration_schema(),
        fixed_scientific_context={
            "objective": "mean five-fold balanced accuracy",
            "baseline_configuration": baseline_configuration(),
            "preprocessing": "training-fold-only z-score standardisation",
        },
        task_limitations=(
            "This small benchmark is not evidence for clinical or ecological deployment.",
        ),
        safety_notes=(
            "Raw observations, row labels, folds, predictions, and confusion matrices are excluded from model context.",
        ),
    )
