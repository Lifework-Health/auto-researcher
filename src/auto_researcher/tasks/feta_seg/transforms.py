"""Locked deterministic and training-only MONAI transform factories."""

PREPROCESSING_VERSION = "feta-ras-0.5mm-foreground-zscore-patchpad128-v2"
AUGMENTATION_VERSION = "feta-flip-scale-shift-v1"
CACHE_IDENTITY_VERSION = "feta-persistent-cache-identity-v1"


def create_transforms(*, training: bool):
    try:
        from monai.transforms import (
            Compose,
            CropForegroundd,
            EnsureChannelFirstd,
            LoadImaged,
            NormalizeIntensityd,
            Orientationd,
            RandFlipd,
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
    return Compose(
        deterministic
        + [
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=2,
                allow_smaller=True,
            ),
            SpatialPadd(
                keys=("image", "label"),
                spatial_size=(128, 128, 128),
                method="symmetric",
                mode="constant",
            ),
            RandFlipd(keys=("image", "label"), prob=0.2, spatial_axis=0),
            RandFlipd(keys=("image", "label"), prob=0.2, spatial_axis=1),
            RandFlipd(keys=("image", "label"), prob=0.2, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
        ]
    )
