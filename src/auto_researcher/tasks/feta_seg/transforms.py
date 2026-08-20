"""Locked deterministic and training-only MONAI transform factories."""

PREPROCESSING_VERSION = "feta-ras-0.5mm-foreground-zscore-patchpad128-v2"
AUGMENTATION_VERSION = "feta-flip-scale-shift-v1"
CACHE_IDENTITY_VERSION = "feta-persistent-cache-identity-v1"


def create_transforms(
    *,
    training: bool,
    positive_negative_ratio: str = "1:1",
    augmentation_strength: str = "baseline",
    augmentation_policy: str | None = None,
):
    try:
        from monai.transforms import (
            Compose,
            CropForegroundd,
            EnsureChannelFirstd,
            LoadImaged,
            NormalizeIntensityd,
            Orientationd,
            RandAffined,
            RandFlipd,
            RandGaussianNoised,
            RandScaleIntensityd,
            RandShiftIntensityd,
            RandCropByPosNegLabeld,
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
            pixdim=(0.5, 0.5, 0.5),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=("image", "label"), source_key="image"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if not training:
        return Compose(deterministic)
    try:
        positive, negative = (
            int(item) for item in positive_negative_ratio.split(":", maxsplit=1)
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("feta_positive_negative_ratio_invalid") from exc
    if augmentation_policy is None:
        augmentation = {
            "light": (0.1, 0.05, 0.05),
            "baseline": (0.2, 0.1, 0.1),
            "strong": (0.35, 0.2, 0.2),
        }
        try:
            probability, intensity_scale, intensity_shift = augmentation[
                augmentation_strength
            ]
        except KeyError as exc:
            raise ValueError("feta_augmentation_strength_invalid") from exc
        augmentation_ops = [
            RandFlipd(keys=("image", "label"), prob=probability, spatial_axis=0),
            RandFlipd(keys=("image", "label"), prob=probability, spatial_axis=1),
            RandFlipd(keys=("image", "label"), prob=probability, spatial_axis=2),
            RandScaleIntensityd(
                keys="image", factors=intensity_scale, prob=probability
            ),
            RandShiftIntensityd(
                keys="image", offsets=intensity_shift, prob=probability
            ),
        ]
    else:
        if augmentation_policy not in {
            "reference_light",
            "geometric",
            "intensity",
            "combined",
        }:
            raise ValueError("feta_augmentation_policy_invalid")
        augmentation_ops = []
        if augmentation_policy in {"reference_light", "geometric", "combined"}:
            flip_probability = 0.1 if augmentation_policy == "reference_light" else 0.2
            augmentation_ops.extend(
                RandFlipd(
                    keys=("image", "label"),
                    prob=flip_probability,
                    spatial_axis=axis,
                )
                for axis in range(3)
            )
        if augmentation_policy in {"geometric", "combined"}:
            augmentation_ops.append(
                RandAffined(
                    keys=("image", "label"),
                    prob=0.25 if augmentation_policy == "geometric" else 0.2,
                    rotate_range=(0.15, 0.15, 0.15),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                )
            )
        if augmentation_policy in {"reference_light", "intensity", "combined"}:
            if augmentation_policy == "reference_light":
                probability, scale, shift = 0.1, 0.05, 0.05
            elif augmentation_policy == "intensity":
                probability, scale, shift = 0.3, 0.15, 0.1
            else:
                probability, scale, shift = 0.2, 0.1, 0.075
            augmentation_ops.extend(
                (
                    RandScaleIntensityd(keys="image", factors=scale, prob=probability),
                    RandShiftIntensityd(keys="image", offsets=shift, prob=probability),
                )
            )
            if augmentation_policy in {"intensity", "combined"}:
                augmentation_ops.append(
                    RandGaussianNoised(
                        keys="image",
                        prob=0.15,
                        mean=0.0,
                        std=0.03,
                    )
                )
    return Compose(
        deterministic
        + [
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=(128, 128, 128),
                pos=positive,
                neg=negative,
                num_samples=2,
                allow_smaller=True,
            ),
            SpatialPadd(
                keys=("image", "label"),
                spatial_size=(128, 128, 128),
                method="symmetric",
                mode="constant",
            ),
            *augmentation_ops,
        ]
    )
