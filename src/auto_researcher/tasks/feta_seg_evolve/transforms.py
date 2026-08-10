"""Locked FeTA preprocessing with a bounded numeric augmentation recipe."""

from __future__ import annotations

from typing import Any

from auto_researcher.tasks.feta_seg_evolve.configuration import (
    FeTASegEvolveConfiguration,
)
from auto_researcher.tasks.feta_seg_search.transforms import (
    POSITIVE_NEGATIVE_COUNTS,
    PREPROCESSING_VERSION,
)

EVOLVE_AUGMENTATION_VERSION = "feta-evolve-static-numeric-flip-scale-shift-v1"


def create_evolve_transforms(
    configuration: FeTASegEvolveConfiguration, *, training: bool
) -> Any:
    try:
        from monai.transforms import (
            Compose,
            CropForegroundd,
            EnsureChannelFirstd,
            LoadImaged,
            NormalizeIntensityd,
            Orientationd,
            RandCropByPosNegLabeld,
            RandFlipd,
            RandScaleIntensityd,
            RandShiftIntensityd,
            Spacingd,
            SpatialPadd,
        )
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc

    deterministic = [
        LoadImaged(keys=("image", "label")),
        EnsureChannelFirstd(keys=("image", "label")),
        Orientationd(
            keys=("image", "label"),
            axcodes="RAS",
            labels=(("L", "R"), ("P", "A"), ("I", "S")),
        ),
        Spacingd(
            keys=("image", "label"),
            pixdim=configuration.spacing_mm,
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=("image", "label"), source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if not training:
        return Compose(deterministic)

    pos, neg = POSITIVE_NEGATIVE_COUNTS[
        configuration.training_policy.positive_negative_ratio
    ]
    recipe = configuration.training_policy.augmentation
    return Compose(
        deterministic
        + [
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=configuration.patch_size,
                pos=pos,
                neg=neg,
                num_samples=configuration.samples_per_volume,
                allow_smaller=True,
            ),
            SpatialPadd(
                keys=("image", "label"),
                spatial_size=configuration.patch_size,
                method="symmetric",
                mode="constant",
            ),
            RandFlipd(
                keys=("image", "label"),
                prob=recipe.flip_probability,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=("image", "label"),
                prob=recipe.flip_probability,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=("image", "label"),
                prob=recipe.flip_probability,
                spatial_axis=2,
            ),
            RandScaleIntensityd(
                keys="image",
                factors=recipe.scale_factor,
                prob=recipe.intensity_probability,
            ),
            RandShiftIntensityd(
                keys="image",
                offsets=recipe.shift_offset,
                prob=recipe.intensity_probability,
            ),
        ]
    )


__all__ = [
    "EVOLVE_AUGMENTATION_VERSION",
    "PREPROCESSING_VERSION",
    "create_evolve_transforms",
]
