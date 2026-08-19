"""Training primitives for the frozen FeTA BasicUNet task."""

from auto_researcher.tasks.feta_seg.trainer import (
    checkpoint_reference,
    require_full_baseline_environment,
    seed_everything,
)
from auto_researcher.tasks.feta_unet_direct.configuration import (
    FeTAUNetDirectConfiguration,
)


def create_loss(configuration: FeTAUNetDirectConfiguration):
    variant = str(getattr(configuration, "loss_variant", "dice_ce"))
    dice_weight = float(getattr(configuration, "dice_weight", 1.0))
    try:
        from monai.losses import DiceCELoss, DiceFocalLoss
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    common = {
        "to_onehot_y": True,
        "softmax": True,
        "include_background": False,
    }
    if variant == "dice_ce":
        return DiceCELoss(**common, lambda_dice=dice_weight, lambda_ce=1.0)
    if variant == "dice_focal":
        return DiceFocalLoss(
            **common,
            lambda_dice=dice_weight,
            lambda_focal=1.0,
        )
    raise ValueError("feta_unet_loss_variant_invalid")


def create_optimizer(model, configuration: FeTAUNetDirectConfiguration):
    try:
        from torch.optim import Adam, AdamW
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    name = str(getattr(configuration, "optimizer", "AdamW"))
    optimiser_class = {"Adam": Adam, "AdamW": AdamW}.get(name)
    if optimiser_class is None:
        raise ValueError("feta_unet_optimizer_invalid")
    return optimiser_class(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )


def create_scheduler(optimizer, configuration: FeTAUNetDirectConfiguration):
    name = str(getattr(configuration, "lr_schedule", "constant"))
    if name == "constant":
        return None
    try:
        from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    horizon = int(getattr(configuration, "scheduler_horizon_epochs", 150))
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=horizon, eta_min=0.0)
    if name == "polynomial":
        return PolynomialLR(
            optimizer,
            total_iters=horizon,
            power=float(getattr(configuration, "polynomial_power", 0.9)),
        )
    raise ValueError("feta_unet_lr_schedule_invalid")


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
    "create_scheduler",
    "require_full_baseline_environment",
    "seed_everything",
    "sliding_window_predict",
]
