"""Lazy construction of the frozen and bounded-search MONAI BasicUNets."""

import hashlib
import json

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


def architecture_identity(configuration: FeTAUNetDirectConfiguration) -> str:
    payload = {
        "spatial_dims": configuration.spatial_dims,
        "in_channels": configuration.in_channels,
        "out_channels": configuration.out_channels,
        "features": list(configuration.features),
        "activation": configuration.activation,
        "norm": configuration.norm,
        "upsample": configuration.upsample,
    }
    if payload == {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 8,
        "features": [32, 32, 64, 128, 256, 32],
        "activation": "LeakyReLU",
        "norm": "instance",
        "upsample": "deconv",
    }:
        return ARCHITECTURE_ID
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"monai-basic-unet-3d-bounded-v2-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _activation(configuration: FeTAUNetDirectConfiguration):
    if configuration.activation == "LeakyReLU":
        return (
            "LeakyReLU",
            {
                "negative_slope": configuration.negative_slope,
                "inplace": configuration.activation_inplace,
            },
        )
    if configuration.activation == "ReLU":
        return ("ReLU", {"inplace": configuration.activation_inplace})
    if configuration.activation == "PReLU":
        return ("PReLU", {"num_parameters": 1, "init": 0.25})
    raise ValueError("feta_unet_activation_invalid")


def _normalisation(configuration: FeTAUNetDirectConfiguration):
    if configuration.norm == "instance":
        return ("instance", {"affine": configuration.norm_affine})
    if configuration.norm == "group":
        groups = int(getattr(configuration, "norm_num_groups", 8))
        return ("group", {"num_groups": groups, "affine": configuration.norm_affine})
    raise ValueError("feta_unet_normalisation_invalid")


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
        act=_activation(configuration),
        norm=_normalisation(configuration),
        dropout=configuration.dropout,
        upsample=configuration.upsample,
    )


def trainable_parameter_count(model) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
