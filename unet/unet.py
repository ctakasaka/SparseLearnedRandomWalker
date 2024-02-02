import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv Block.

    Performs two convolutions, each followed by 2D batch normalization,
    followed by a ReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class EncoderBlock(nn.Module):
    """Encoder Block.

    This block performs a 2D max-pooling operation followed by a ConvBlock.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_block(x)


class DecoderBlock(nn.Module):
    """Decoder Block.

    This block performs a 2D transposed convolution (upsampling),
    concatenates the encoder feature map, and applies a ConvBlock.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3)
        )

    def forward(self, h: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = self.up_conv(h)
        x = torch.cat([skip, h], dim=1)
        return self.conv_block(x)


class FinalConv(nn.Module):
    """Final Conv.

    This block applies a final 1x1 convolution to produce class logits.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net model for image segmentation tasks."""
    def __init__(self, n_channels=3, starting_h_channels=64, number_of_blocks=4):
        super().__init__()

        encoder_blocks = []
        decoder_blocks = []

        cur_channels = starting_h_channels
        for _ in range(number_of_blocks):
            encoder_block = EncoderBlock(cur_channels, 2 * cur_channels)
            encoder_blocks.append(encoder_block)
            cur_channels = cur_channels * 2

        for _ in range(number_of_blocks):
            decoder_block = DecoderBlock(cur_channels, cur_channels // 2)
            decoder_blocks.append(decoder_block)
            cur_channels = cur_channels // 2

        self.initial_conv = ConvBlock(n_channels, starting_h_channels)
        self.encoder_blocks = torch.nn.ModuleList(encoder_blocks)
        self.decoder_blocks = torch.nn.ModuleList(decoder_blocks)
        self.final_conv = FinalConv(starting_h_channels, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_feature_maps = []

        h = self.initial_conv(x)
        encoder_feature_maps.append(h)

        for encoder_block in self.encoder_blocks:
            h = encoder_block(h)
            encoder_feature_maps.append(h)

        # the last encoder feature map is not passed as skip connection
        # also, the feature maps are passed to the decoder in reversed order
        skip_feature_maps = encoder_feature_maps[:-1][::-1]
        for idx, decoder_block in enumerate(self.decoder_blocks):
            h = decoder_block(h, skip_feature_maps[idx])

        logits = self.final_conv(h)
        return logits
