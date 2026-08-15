"""Lazy construction of the frozen MONAI BasicUNet."""

from auto_researcher.tasks.feta_unet_direct.configuration import (
    FeTAUNetDirectConfiguration,
)

ARCHITECTURE_ID = "monai-basic-unet-3d-v1"
TRAINABLE_PARAMETER_COUNT = 5_749_608
MEASURED_INPUT_SHAPE = (1, 1, 128, 128, 128)
MEASURED_OUTPUT_SHAPE = (1, 8, 128, 128, 128)
MEASURED_PEAK_CUDA_ALLOCATED_MIB = 2_338
MEASURED_PEAK_CUDA_RESERVED_MIB = 3_076
MEASURED_ALLOCATOR_CEILING_MIB = 20_638


def create_basic_unet(configuration: FeTAUNetDirectConfiguration):
    try:
        from monai.networks.nets import BasicUNet
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    return BasicUNet(
        spatial_dims=configuration.spatial_dims,
        in_channels=configuration.in_channels,
        out_channels=configuration.out_channels,
        features=configuration.features,
        act=(
            configuration.activation,
            {
                "negative_slope": configuration.negative_slope,
                "inplace": configuration.activation_inplace,
            },
        ),
        norm=(configuration.norm, {"affine": configuration.norm_affine}),
        dropout=configuration.dropout,
        upsample=configuration.upsample,
    )


def trainable_parameter_count(model) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
