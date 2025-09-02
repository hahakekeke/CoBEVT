'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2):
        super().__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y, x)

        return y
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim=None, use_skip=True, factor=2):
        super().__init__()

        hidden_dim = out_channels // factor

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_skip = use_skip
        if use_skip and skip_dim is not None:
            self.skip_proj = nn.Conv2d(skip_dim, out_channels, 1, bias=False)
        else:
            self.skip_proj = None

        # Channel attention
        self.se = SEBlock(out_channels)

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.use_skip and skip is not None:
            skip = self.skip_proj(skip)
            skip = F.interpolate(skip, x.shape[-2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)  # concat instead of sum
            # Fuse after concat
            x = nn.Conv2d(x.shape[1], x.shape[1] // 2, 1)(x)

        x = self.se(x)  # apply channel attention

        return self.relu_out(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, blocks, skip_dims=None, factor=2):
        super().__init__()

        layers = []
        channels = in_channels

        if skip_dims is None:
            skip_dims = [None] * len(blocks)

        for out_channels, skip_dim in zip(blocks, skip_dims):
            layers.append(DecoderBlock(channels, out_channels, skip_dim, use_skip=(skip_dim is not None), factor=factor))
            channels = out_channels

        self.layers = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, x, skips=None):
        y = x
        if skips is None:
            skips = [None] * len(self.layers)

        for layer, skip in zip(self.layers, skips):
            y = layer(y, skip)

        return y
