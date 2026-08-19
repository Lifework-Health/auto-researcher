"""Trusted evaluator adapter for bounded BasicUNet search candidates."""

from auto_researcher.tasks.feta_unet_direct.evaluator import (
    FeTAUNetDirectEvaluator,
    evaluator_code_version as direct_evaluator_code_version,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    CONFIGURATION_SCHEMA_VERSION,
    SEARCH_ARCHITECTURE_FAMILY_ID,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)

EVALUATOR_ID = "feta-basic-unet-search-evaluator"
EVALUATOR_VERSION = "feta-basic-unet-search-evaluator-v2"
RESULT_ID = "feta-basic-unet-search-result-v2"
SCIENTIFIC_ID = "feta-basic-unet-fold0-bounded-tree-search-macro-dice-v2"
AUGMENTATION_ID = "feta-bounded-flip-scale-shift-and-patch-ratio-v1"
LOSS_ID = "bounded-dice-ce-or-dice-focal-no-background-v2"
OPTIMISER_ID = "adam-or-adamw-bounded-lr-wd-with-150epoch-schedules-v2"


def evaluator_code_version(dataset_version: str) -> str:
    return "+".join(
        (
            direct_evaluator_code_version(dataset_version),
            CONFIGURATION_SCHEMA_VERSION,
            EVALUATOR_VERSION,
            RESULT_ID,
            AUGMENTATION_ID,
            LOSS_ID,
            OPTIMISER_ID,
        )
    )


class FeTAUNetSearchEvaluator(FeTAUNetDirectEvaluator):
    evaluator_id = EVALUATOR_ID
    version = EVALUATOR_VERSION
    architecture_family_identity = SEARCH_ARCHITECTURE_FAMILY_ID

    def __init__(
        self,
        context: TaskRuntimeContext,
        metadata: ExperimentMetadata,
        manifest: DatasetManifest,
        **kwargs,
    ) -> None:
        super().__init__(
            context,
            metadata,
            manifest,
            configuration_model=FeTAUNetSearchConfiguration,
            task_id="feta_unet_search",
            scientific_identity=SCIENTIFIC_ID,
            result_identity=RESULT_ID,
            augmentation_identity=AUGMENTATION_ID,
            loss_identity=LOSS_ID,
            optimiser_identity=OPTIMISER_ID,
            **kwargs,
        )
