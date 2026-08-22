"""Bounded post-training inference calibration for verified U-Net finalists."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class InferenceCalibrationVariant(BaseModel):
    """A non-training inference policy kept outside scientific model identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overlap: Literal[0.25, 0.5, 0.75]
    blending: Literal["gaussian", "constant"]
    flip_tta: bool
    class_specific_postprocessing: Literal["none", "largest_component"] = "none"


def calibration_variants() -> tuple[InferenceCalibrationVariant, ...]:
    """Return the frozen eight-variant V7 calibration lane."""

    return tuple(
        InferenceCalibrationVariant(**item)
        for item in (
            {"overlap": 0.5, "blending": "gaussian", "flip_tta": False},
            {"overlap": 0.25, "blending": "gaussian", "flip_tta": False},
            {"overlap": 0.75, "blending": "gaussian", "flip_tta": False},
            {"overlap": 0.5, "blending": "constant", "flip_tta": False},
            {"overlap": 0.25, "blending": "constant", "flip_tta": False},
            {"overlap": 0.75, "blending": "constant", "flip_tta": False},
            {"overlap": 0.5, "blending": "gaussian", "flip_tta": True},
            {"overlap": 0.5, "blending": "constant", "flip_tta": True},
        )
    )


def predict_calibrated_logits(inputs, model, configuration, variant):
    """Run sliding-window inference with optional three-axis flip TTA."""

    try:
        import torch
        from monai.inferers import sliding_window_inference
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    policy = InferenceCalibrationVariant.model_validate(variant)

    def predict(values):
        return sliding_window_inference(
            values,
            roi_size=configuration.patch_size,
            sw_batch_size=configuration.sliding_window_batch_size,
            predictor=model,
            overlap=policy.overlap,
            mode=policy.blending,
        )

    logits = [predict(inputs)]
    if policy.flip_tta:
        for dimension in (2, 3, 4):
            flipped = torch.flip(inputs, dims=(dimension,))
            logits.append(torch.flip(predict(flipped), dims=(dimension,)))
    result = torch.stack(logits).mean(dim=0)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("feta_unet_calibration_prediction_non_finite")
    return result


def postprocess_calibrated_prediction(prediction, variant):
    """Optionally retain the largest connected component independently per class."""

    policy = InferenceCalibrationVariant.model_validate(variant)
    if policy.class_specific_postprocessing == "none":
        return prediction
    try:
        import numpy as np
        from scipy import ndimage  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional metric dependency
        raise RuntimeError("feta_topology_dependencies_unavailable") from exc
    source = np.asarray(prediction)
    result = np.zeros_like(source)
    for label in range(1, 8):
        components, count = ndimage.label(source == label)
        if count == 0:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        result[components == int(sizes.argmax())] = label
    return result


__all__ = [
    "InferenceCalibrationVariant",
    "calibration_variants",
    "postprocess_calibrated_prediction",
    "predict_calibrated_logits",
]
