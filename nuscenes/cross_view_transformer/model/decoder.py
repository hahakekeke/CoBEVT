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


class ResidualAttentionBlock(nn.Module):
    """간단한 채널+공간 Attention Block (정확도 향상 목적)"""
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim=None, residual=True, factor=2):
        super().__init__()
        dim = out_channels // factor

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, dim, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.attn = ResidualAttentionBlock(out_channels)

        if residual and skip_dim is not None:
            self.skip_proj = nn.Conv2d(skip_dim, out_channels, 1, bias=False)
        else:
            self.skip_proj = None

    def forward(self, x, skip=None):
        x = self.upsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if skip is not None and self.skip_proj is not None:
            skip = self.skip_proj(skip)
            skip = F.interpolate(skip, x.shape[-2:], mode="bilinear", align_corners=True)
            x = x + skip

        x = self.attn(self.relu(x))
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, blocks, skip_dims=None, residual=True, factor=2):
        super().__init__()

        if skip_dims is None:
            skip_dims = [None] * len(blocks)

        layers = []
        channels = in_channels

        for out_channels, skip_dim in zip(blocks, skip_dims):
            layers.append(DecoderBlock(channels, out_channels, skip_dim, residual, factor))
            channels = out_channels

        self.layers = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, x):
        """
        x가 dict로 들어올 수도 있고 (예: {'features': tensor, 'skips': [...]})
        그냥 tensor로 들어올 수도 있음.
        """
        if isinstance(x, dict):
            features = x.get("features", None)
            skips = x.get("skips", None)
            if features is None:
                raise ValueError("Decoder forward: dict 입력에는 'features' key가 필요합니다.")
        else:
            features = x
            skips = None

        y = features
        for i, layer in enumerate(self.layers):
            skip = skips[i] if (skips is not None and i < len(skips)) else None
            y = layer(y, skip)

        return y
