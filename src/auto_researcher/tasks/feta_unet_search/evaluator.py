"""Trusted evaluator adapter for bounded U-Net search candidates."""

from auto_researcher.tasks.feta_unet_direct.evaluator import (
    FeTAUNetDirectEvaluator,
    evaluator_code_version as direct_evaluator_code_version,
)
from auto_researcher.tasks.feta_unet_direct.runner import (
    SEARCH_DATA_LOADER_ID,
    SEARCH_RUNNER_ID,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    CONFIGURATION_SCHEMA_VERSION,
    SEARCH_ARCHITECTURE_FAMILY_ID,
    V7_ARCHITECTURE_BUDGET,
    V7_MAXIMUM_PEAK_GPU_MEMORY_BYTES,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)

EVALUATOR_ID = "feta-basic-unet-search-evaluator"
EVALUATOR_VERSION = "feta-unet-search-evaluator-v5"
RESULT_ID = "feta-unet-search-result-v5"
SCIENTIFIC_ID = "feta-unet-fold0-bounded-family-tree-search-macro-dice-v5"
AUGMENTATION_ID = "feta-bounded-explicit-geometric-intensity-policies-v2"
LOSS_ID = "bounded-dice-ce-focal-or-tversky-no-background-v3"
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
    development_runner_identity = SEARCH_RUNNER_ID
    data_loader_identity = SEARCH_DATA_LOADER_ID

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

    def evaluate(self, experiment, contract):
        """Reject V7 evidence that exceeded the frozen GPU-memory ceiling."""

        result = super().evaluate(experiment, contract)
        if not result.success:
            return result
        try:
            configuration = FeTAUNetSearchConfiguration.model_validate(
                experiment.configuration
            )
        except (TypeError, ValueError):
            return self._failure(experiment, "feta_unet_configuration_invalid")
        if configuration.architecture_budget != V7_ARCHITECTURE_BUDGET:
            return result
        configured_limit = self.context.task_options.get(
            "maximum_peak_gpu_memory_bytes"
        )
        if configured_limit != V7_MAXIMUM_PEAK_GPU_MEMORY_BYTES:
            return self._failure(
                experiment, "feta_unet_peak_gpu_memory_limit_invalid"
            )
        raw_summaries = result.metrics.get("fold_summaries")
        if not isinstance(raw_summaries, list) or not raw_summaries:
            return self._failure(
                experiment, "feta_unet_peak_gpu_memory_evidence_missing"
            )
        peaks = [
            summary.get("peak_gpu_memory_bytes")
            for summary in raw_summaries
            if isinstance(summary, dict)
        ]
        if (
            len(peaks) != len(raw_summaries)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in peaks
            )
        ):
            return self._failure(
                experiment, "feta_unet_peak_gpu_memory_evidence_invalid"
            )
        if max(peaks) > V7_MAXIMUM_PEAK_GPU_MEMORY_BYTES:
            return self._failure(
                experiment, "feta_unet_peak_gpu_memory_limit_exceeded"
            )
        return result
