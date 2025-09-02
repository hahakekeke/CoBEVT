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
from typing import List

class ImprovedDecoderBlock(nn.Module):
    """
    U-Net 스타일의 Decoder 블록.
    Upsample -> Concatenate with skip connection -> Double Convolution
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        conv_in_channels = (in_channels // 2) + skip_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        combined_features = torch.cat([x, skip], dim=1)
        
        return self.conv_block(combined_features)

class ImprovedDecoder(nn.Module):
    def __init__(
        self,
        encoder_dims: List[int],
        decoder_out_channels: List[int],
        num_classes: int,
    ):
        super().__init__()
        
        encoder_dims = encoder_dims[::-1]
        
        self.decoder_blocks = nn.ModuleList()
        
        in_channels = encoder_dims[0]
        
        for i in range(len(decoder_out_channels)):
            skip_channels = encoder_dims[i+1]
            out_channels = decoder_out_channels[i]
            
            self.decoder_blocks.append(
                ImprovedDecoderBlock(in_channels, skip_channels, out_channels)
            )
            in_channels = out_channels

        self.segmentation_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.out_channels = num_classes

    def forward(self, bev_features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Encoder에서 반환된 다중 스케일 BEV 특징 맵 리스트를 입력으로 받습니다.
        bev_features_list: [고해상도 특징, ..., 저해상도 특징] 순서
        """
        
        skips = bev_features_list[::-1]
        
        x = skips[0]
        remaining_skips = skips[1:]
        
        for i, block in enumerate(self.decoder_blocks):
            skip_connection = remaining_skips[i]
            x = block(x, skip_connection)
            
        output = self.segmentation_head(x)
        
        return output
