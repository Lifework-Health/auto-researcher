"""Lazy MONAI SegResNet construction; the core package has no ML dependency."""

from auto_researcher.tasks.feta_seg.configuration import FeTASegConfiguration


def create_segresnet(configuration: FeTASegConfiguration):
    try:
        from monai.networks.nets import SegResNet
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    return SegResNet(
        spatial_dims=configuration.spatial_dims,
        init_filters=configuration.init_filters,
        in_channels=configuration.in_channels,
        out_channels=configuration.out_channels,
        dropout_prob=configuration.dropout,
        norm=(configuration.norm, {"num_groups": 8}),
        act=configuration.activation,
        blocks_down=configuration.blocks_down,
        blocks_up=configuration.blocks_up,
        upsample_mode=configuration.upsample_mode,
    )


def trainable_parameter_count(model) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
