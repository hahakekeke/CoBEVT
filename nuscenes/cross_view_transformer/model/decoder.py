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

class DecoderBlock(nn.Module):
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

class Decoder(nn.Module):
    """
    YAML 수정 없이 Encoder 출력에 자동으로 맞춰 레이어를 생성하는 Decoder (지연 초기화 적용).
    """
    def __init__(
        self,
        dim: int,
        blocks: List[int],
        residual: bool = True, # YAML과의 호환성을 위해 남겨둠 (사용 X)
        factor: int = 2,       # YAML과의 호환성을 위해 남겨둠 (사용 X)
    ):
        super().__init__()
        # YAML에서 받은 파라미터를 저장
        self.decoder_out_channels = blocks
        self.out_channels = blocks[-1] # 최종 출력 채널 (원래 코드와 호환)

        # 레이어를 None으로 초기화 (지연 초기화를 위한 플래그)
        self.decoder_blocks = None

    def _initialize_layers(self, bev_features_list: List[torch.Tensor]):
        """
        첫 forward 호출 시 동적으로 레이어를 생성하는 함수.
        """
        # 1. 입력 데이터로부터 encoder_dims 정보를 자동으로 추출
        encoder_dims = [feat.shape[1] for feat in bev_features_list]
        reversed_encoder_dims = encoder_dims[::-1]
        
        # 2. 추출된 정보를 바탕으로 DecoderBlock들을 생성
        decoder_blocks = nn.ModuleList()
        in_channels = reversed_encoder_dims[0]
        
        for i in range(len(self.decoder_out_channels)):
            skip_channels = reversed_encoder_dims[i+1]
            out_channels = self.decoder_out_channels[i]
            
            decoder_blocks.append(
                DecoderBlock(in_channels, skip_channels, out_channels)
            )
            in_channels = out_channels
        
        # 3. 생성된 레이어를 self.decoder_blocks에 할당
        self.decoder_blocks = decoder_blocks
        
        # 4. 생성된 레이어를 입력 데이터와 동일한 장치(cpu/gpu)로 이동
        device = bev_features_list[0].device
        self.to(device)


    def forward(self, bev_features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Encoder에서 반환된 다중 스케일 BEV 특징 맵 리스트를 입력으로 받습니다.
        """
        # [지연 초기화] 첫 forward pass일 때만 레이어를 생성
        if self.decoder_blocks is None:
            self._initialize_layers(bev_features_list)
        
        # U-Net 구조를 위해 특징 리스트를 뒤집습니다.
        skips = bev_features_list[::-1]
        
        x = skips[0]
        remaining_skips = skips[1:]
        
        # Decoder 블록들을 순서대로 통과
        for i, block in enumerate(self.decoder_blocks):
            skip_connection = remaining_skips[i]
            x = block(x, skip_connection)
            
        return x
