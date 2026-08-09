"""Candidate-controlled training transforms over locked FeTA preprocessing."""

from __future__ import annotations

from typing import NamedTuple

from auto_researcher.tasks.feta_seg_search.configuration import (
    FeTASegSearchConfiguration,
)

PREPROCESSING_VERSION = "feta-ras-0.5mm-foreground-zscore-patchpad128-v2"
AUGMENTATION_POLICY_VERSION = "feta-search-flip-scale-shift-policies-v1"
CACHE_IDENTITY_VERSION = "feta-search-persistent-cache-identity-v1"


class AugmentationPolicy(NamedTuple):
    probability: float
    scale_factor: float
    shift_offset: float


AUGMENTATION_POLICIES = {
    "light": AugmentationPolicy(0.10, 0.05, 0.05),
    "baseline": AugmentationPolicy(0.20, 0.10, 0.10),
    "strong": AugmentationPolicy(0.30, 0.15, 0.15),
}
POSITIVE_NEGATIVE_COUNTS = {
    "1:1": (1, 1),
    "2:1": (2, 1),
    "3:1": (3, 1),
}


def augmentation_policy(strength: str) -> AugmentationPolicy:
    try:
        return AUGMENTATION_POLICIES[strength]
    except KeyError as exc:
        raise ValueError("feta_search_augmentation_strength_invalid") from exc


def positive_negative_counts(ratio: str) -> tuple[int, int]:
    try:
        return POSITIVE_NEGATIVE_COUNTS[ratio]
    except KeyError as exc:
        raise ValueError("feta_search_positive_negative_ratio_invalid") from exc


def create_transforms(
    configuration: FeTASegSearchConfiguration, *, training: bool
):
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

    pos, neg = positive_negative_counts(configuration.positive_negative_ratio)
    policy = augmentation_policy(configuration.augmentation_strength)
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
            RandFlipd(keys=("image", "label"), prob=policy.probability, spatial_axis=0),
            RandFlipd(keys=("image", "label"), prob=policy.probability, spatial_axis=1),
            RandFlipd(keys=("image", "label"), prob=policy.probability, spatial_axis=2),
            RandScaleIntensityd(
                keys="image", factors=policy.scale_factor, prob=policy.probability
            ),
            RandShiftIntensityd(
                keys="image", offsets=policy.shift_offset, prob=policy.probability
            ),
        ]
    )
