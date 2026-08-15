"""Training primitives for the frozen FeTA BasicUNet task."""

from auto_researcher.tasks.feta_seg.trainer import (
    checkpoint_reference,
    create_loss,
    require_full_baseline_environment,
    seed_everything,
)
from auto_researcher.tasks.feta_unet_direct.configuration import (
    FeTAUNetDirectConfiguration,
)


def create_optimizer(model, configuration: FeTAUNetDirectConfiguration):
    try:
        from torch.optim import AdamW
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    return AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )


def sliding_window_predict(inputs, model, configuration: FeTAUNetDirectConfiguration):
    try:
        from monai.inferers import sliding_window_inference
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    return sliding_window_inference(
        inputs,
        roi_size=configuration.patch_size,
        sw_batch_size=configuration.sliding_window_batch_size,
        predictor=model,
        overlap=configuration.inference_overlap,
        mode=configuration.inference_blending,
    )


__all__ = [
    "checkpoint_reference",
    "create_loss",
    "create_optimizer",
    "require_full_baseline_environment",
    "seed_everything",
    "sliding_window_predict",
]
