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

# [수정] U-Net 스타일의 향상된 Decoder 블록
class ImprovedDecoderBlock(nn.Module):
    """
    U-Net 스타일의 Decoder 블록.
    Upsample -> Concatenate with skip connection -> Double Convolution
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        # 1. Upsample
        # ConvTranspose2d가 학습 가능한 파라미터를 제공하여 더 유연할 수 있습니다.
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # 2. Concatenate 이후의 Double Convolution Block
        # 입력 채널: 업샘플링된 채널(in_channels // 2) + skip connection 채널(skip_channels)
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
        # x: 이전 디코더 블록에서 온 저해상도 특징 맵
        # skip: 인코더의 해당 레벨에서 온 고해상도 특징 맵 (skip connection)
        
        x = self.upsample(x)
        
        # 크기가 미세하게 다를 경우 skip connection의 크기를 맞춰줍니다. (e.g., 홀수 크기 때문에)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        # 채널 축(dimension 1)을 따라 concatenate
        combined_features = torch.cat([x, skip], dim=1)
        
        return self.conv_block(combined_features)

# [수정] Encoder의 출력을 받아 U-Net 구조로 동작하는 전체 Decoder
class ImprovedDecoder(nn.Module):
    def __init__(
        self,
        encoder_dims: List[int],     # Encoder 각 레벨의 출력 채널 리스트. 예: [64, 128, 256]
        decoder_out_channels: List[int], # Decoder 각 블록의 출력 채널 리스트. 예: [128, 64]
        num_classes: int,            # 최종 출력 클래스 수
    ):
        super().__init__()
        
        # Encoder 특징은 고해상도 -> 저해상도 순서. Decoder는 역순으로 사용.
        encoder_dims = encoder_dims[::-1] # [256, 128, 64]
        
        self.decoder_blocks = nn.ModuleList()
        
        # 가장 깊은(저해상도) 특징부터 시작
        in_channels = encoder_dims[0] # 256
        
        for i in range(len(decoder_out_channels)):
            skip_channels = encoder_dims[i+1] # 128, 64 ...
            out_channels = decoder_out_channels[i] # 128, 64 ...
            
            self.decoder_blocks.append(
                ImprovedDecoderBlock(in_channels, skip_channels, out_channels)
            )
            in_channels = out_channels

        # 최종 예측을 위한 Segmentation Head
        self.segmentation_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.out_channels = num_classes

    def forward(self, bev_features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Encoder에서 반환된 다중 스케일 BEV 특징 맵 리스트를 입력으로 받습니다.
        bev_features_list: [고해상도 특징, ..., 저해상도 특징] 순서
        """
        
        # U-Net 구조를 위해 특징 리스트를 뒤집습니다. (저해상도 -> 고해상도 순으로 처리)
        skips = bev_features_list[::-1] # [저해상도 특징, ..., 고해상도 특징]
        
        # 가장 깊은 특징 맵을 초기 입력으로 사용
        x = skips[0]
        
        # 나머지 특징 맵을 skip connection으로 사용
        remaining_skips = skips[1:]
        
        # Decoder 블록들을 순서대로 통과
        for i, block in enumerate(self.decoder_blocks):
            skip_connection = remaining_skips[i]
            x = block(x, skip_connection)
            
        # 최종 Segmentation Head를 통과하여 예측 결과 생성
        output = self.segmentation_head(x)
        
        return output
