"""Constrained structural BasicUNet lineage used by the V7 campaign."""

from __future__ import annotations


def create_structural_basic_unet(configuration):
    """Build a 3-D BasicUNet encoder-decoder with bounded structural choices."""

    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as exc:  # pragma: no cover - optional ML dependency
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc

    features = tuple(int(value) for value in configuration.features)
    encoder_features = features[:-1]
    terminal_decoder_width = features[-1]
    depth = len(encoder_features)
    uniform_convolutions = int(configuration.convolutions_per_stage)

    def stage_convolutions(stage: int) -> int:
        profile = str(getattr(configuration, "stage_block_profile", "uniform"))
        if profile == "uniform":
            return uniform_convolutions
        if profile == "shallow_to_deep":
            return min(3, 1 + stage)
        if profile == "deep_to_shallow":
            return max(1, 3 - stage)
        if profile == "bottleneck_heavy":
            return 3 if stage == depth - 1 else 1
        raise ValueError("feta_unet_stage_block_profile_invalid")

    def stage_has_residual(stage: int, branch: str) -> bool:
        if not bool(configuration.residual_blocks):
            return False
        profile = str(getattr(configuration, "residual_profile", "uniform"))
        if profile == "uniform":
            return True
        if profile == "encoder_only":
            return branch == "encoder"
        if profile == "decoder_only":
            return branch == "decoder"
        if profile == "deep_only":
            return stage >= depth // 2
        raise ValueError("feta_unet_residual_profile_invalid")

    def activation():
        if configuration.activation == "ReLU":
            return nn.ReLU(inplace=True)
        if configuration.activation == "PReLU":
            return nn.PReLU(num_parameters=1, init=0.25)
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def normalisation(channels: int):
        if configuration.norm == "group":
            return nn.GroupNorm(8, channels, affine=True)
        return nn.InstanceNorm3d(channels, affine=True)

    def stage_kernel(stage: int) -> int:
        if configuration.kernel_profile == "large_front" and stage == 0:
            return 5
        if configuration.kernel_profile == "context_deep" and stage >= depth // 2:
            return 5
        return 3

    def stage_dilation(stage: int, convolution: int) -> int:
        profile = configuration.dilation_profile
        if profile == "deep" and stage >= depth // 2:
            return 2
        if profile == "multiscale" and convolution == stage_convolutions(stage) - 1:
            return 2 if stage < depth - 1 else 3
        return 1

    class ConvStage(nn.Module):
        def __init__(
            self, in_channels: int, out_channels: int, stage: int, branch: str
        ):
            super().__init__()
            blocks = []
            current = in_channels
            for convolution in range(stage_convolutions(stage)):
                kernel = stage_kernel(stage)
                dilation = stage_dilation(stage, convolution)
                padding = dilation * (kernel // 2)
                blocks.extend(
                    (
                        nn.Conv3d(
                            current,
                            out_channels,
                            kernel_size=kernel,
                            padding=padding,
                            dilation=dilation,
                            bias=False,
                        ),
                        normalisation(out_channels),
                        activation(),
                        nn.Dropout3d(float(configuration.dropout)),
                    )
                )
                current = out_channels
            self.body = nn.Sequential(*blocks)
            self.residual = stage_has_residual(stage, branch)
            self.projection = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        def forward(self, inputs):
            output = self.body(inputs)
            return output + self.projection(inputs) if self.residual else output

    class PixelShuffle3d(nn.Module):
        def forward(self, inputs):
            batch, channels, depth_size, height, width = inputs.shape
            if channels % 8:
                raise ValueError("feta_unet_pixelshuffle_channels_invalid")
            output_channels = channels // 8
            return (
                inputs.reshape(
                    batch,
                    output_channels,
                    2,
                    2,
                    2,
                    depth_size,
                    height,
                    width,
                )
                .permute(0, 1, 5, 2, 6, 3, 7, 4)
                .reshape(
                    batch,
                    output_channels,
                    depth_size * 2,
                    height * 2,
                    width * 2,
                )
            )

    class Downsample(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.operation = (
                nn.MaxPool3d(kernel_size=2, stride=2)
                if configuration.downsample == "max_pool"
                else nn.Conv3d(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )

        def forward(self, inputs):
            return self.operation(inputs)

    class Upsample(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.mode = configuration.upsample
            if self.mode == "deconv":
                self.operation = nn.ConvTranspose3d(
                    in_channels, out_channels, kernel_size=2, stride=2
                )
            elif self.mode == "pixelshuffle":
                self.operation = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels * 8, kernel_size=1),
                    PixelShuffle3d(),
                )
            else:
                self.operation = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        def forward(self, inputs):
            if self.mode == "nontrainable":
                inputs = functional.interpolate(
                    inputs,
                    scale_factor=2,
                    mode="trilinear",
                    align_corners=False,
                )
            return self.operation(inputs)

    class StructuralBasicUNet(nn.Module):
        def __init__(self):
            super().__init__()
            encoders = []
            downsamplers = []
            current = int(configuration.in_channels)
            for stage, channels in enumerate(encoder_features):
                encoders.append(ConvStage(current, channels, stage, "encoder"))
                current = channels
                if stage < depth - 1:
                    downsamplers.append(Downsample(channels))
            self.encoders = nn.ModuleList(encoders)
            self.downsamplers = nn.ModuleList(downsamplers)

            upsamples = []
            gates = []
            decoders = []
            auxiliary = []
            current = encoder_features[-1]
            decoder_widths = list(reversed(encoder_features[:-1]))
            decoder_widths[-1] = terminal_decoder_width
            for decoder_index, (skip_channels, decoder_channels) in enumerate(
                zip(reversed(encoder_features[:-1]), decoder_widths, strict=True)
            ):
                upsamples.append(Upsample(current, decoder_channels))
                gates.append(
                    nn.Conv3d(decoder_channels, skip_channels, kernel_size=1)
                    if configuration.skip_fusion == "gated_concat"
                    else nn.Identity()
                )
                decoder_input = (
                    decoder_channels
                    if configuration.skip_fusion == "add"
                    else decoder_channels + skip_channels
                )
                stage = depth - decoder_index - 2
                decoders.append(
                    ConvStage(decoder_input, decoder_channels, stage, "decoder")
                )
                auxiliary.append(
                    nn.Conv3d(
                        decoder_channels, configuration.out_channels, kernel_size=1
                    )
                )
                current = decoder_channels
            self.upsamples = nn.ModuleList(upsamples)
            self.gates = nn.ModuleList(gates)
            self.decoders = nn.ModuleList(decoders)
            self.auxiliary = nn.ModuleList(auxiliary)
            self.output = nn.Conv3d(current, configuration.out_channels, kernel_size=1)

        def forward(self, inputs):
            skips = []
            output = inputs
            for index, encoder in enumerate(self.encoders):
                output = encoder(output)
                skips.append(output)
                if index < len(self.downsamplers):
                    output = self.downsamplers[index](output)
            auxiliary_outputs = []
            for index, (upsample, gate, decoder) in enumerate(
                zip(self.upsamples, self.gates, self.decoders, strict=True)
            ):
                output = upsample(output)
                skip = skips[-index - 2]
                if configuration.skip_fusion == "add":
                    if output.shape[1] != skip.shape[1]:
                        raise ValueError("feta_unet_additive_skip_width_invalid")
                    output = output + skip
                else:
                    if configuration.skip_fusion == "gated_concat":
                        skip = skip * torch.sigmoid(gate(output))
                    output = torch.cat((output, skip), dim=1)
                output = decoder(output)
                if index >= len(self.decoders) - configuration.deep_supervision_heads:
                    auxiliary_outputs.append(self.auxiliary[index](output))
            primary = self.output(output)
            if not self.training or configuration.deep_supervision_heads == 0:
                return primary
            resized = [primary]
            for auxiliary_output in reversed(auxiliary_outputs):
                resized.append(
                    functional.interpolate(
                        auxiliary_output,
                        size=primary.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                )
            return torch.stack(resized, dim=1)

    return StructuralBasicUNet()


__all__ = ["create_structural_basic_unet"]
