'''
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


# Precomputed aliases
MODELS = {
    'efficientnet-b0': [
        ('reduction_1', (0, 2)),
        ('reduction_2', (2, 4)),
        ('reduction_3', (4, 6)),
        ('reduction_4', (6, 12))
    ],
    'efficientnet-b4': [
        ('reduction_1', (0, 3)),
        ('reduction_2', (3, 7)),
        ('reduction_3', (7, 11)),
        ('reduction_4', (11, 23)),
    ]
}


class EfficientNetExtractor(torch.nn.Module):
    """
    Helper wrapper that uses torch.utils.checkpoint.checkpoint to save memory while training.

    This runs a fake input with shape (1, 3, input_height, input_width)
    to give the shapes of the features requested.

    Sample usage:
        backbone = EfficientNetExtractor(224, 480, ['reduction_2', 'reduction_4'])

        # [[1, 56, 28, 60], [1, 272, 7, 15]]
        backbone.output_shapes

        # [f1, f2], where f1 is 'reduction_1', which is shape [b, d, 128, 128]
        backbone(x)
    """
    def __init__(self, layer_names, image_height, image_width, model_name='efficientnet-b4'):
        super().__init__()

        assert model_name in MODELS
        assert all(k in [k for k, v in MODELS[model_name]] for k in layer_names)

        idx_max = -1
        layer_to_idx = {}

        # Find which blocks to return
        for i, (layer_name, _) in enumerate(MODELS[model_name]):
            if layer_name in layer_names:
                idx_max = max(idx_max, i)
                layer_to_idx[layer_name] = i

        # We can set memory efficient swish to false since we're using checkpointing
        net = EfficientNet.from_pretrained(model_name)
        net.set_swish(False)

        drop = net._global_params.drop_connect_rate / len(net._blocks)
        blocks = [nn.Sequential(net._conv_stem, net._bn0, net._swish)]

        # Only run needed blocks
        for idx in range(idx_max):
            l, r = MODELS[model_name][idx][1]

            block = SequentialWithArgs(*[(net._blocks[i], [i * drop]) for i in range(l, r)])
            blocks.append(block)

        self.layers = nn.Sequential(*blocks)
        self.layer_names = layer_names
        # the larger resolution result should be at the last position
        self.idx_pick = [layer_to_idx[l] for l in layer_names]

        # Pass a dummy tensor to precompute intermediate shapes
        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self(dummy)]

        self.output_shapes = output_shapes

    def forward(self, x):
        if self.training:
            x = x.requires_grad_(True)

        result = []

        for layer in self.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)

            result.append(x)

        return [result[i] for i in self.idx_pick]


class SequentialWithArgs(nn.Sequential):
    def __init__(self, *layers_args):
        layers = [layer for layer, args in layers_args]
        args = [args for layer, args in layers_args]

        super().__init__(*layers)

        self.args = args

    def forward(self, x):
        for l, a in zip(self, self.args):
            x = l(x, *a)

        return x


if __name__ == '__main__':
    """
    Helper to generate aliases for efficientnet backbones
    """
    device = torch.device('cuda')
    dummy = torch.rand(6, 3, 224, 480).to(device)

    for model_name in ['efficientnet-b0', 'efficientnet-b4']:
        net = EfficientNet.from_pretrained(model_name)
        net = net.to(device)
        net.set_swish(False)

        drop = net._global_params.drop_connect_rate / len(net._blocks)
        conv = nn.Sequential(net._conv_stem, net._bn0, net._swish)

        record = list()

        x = conv(dummy)
        px = x
        pi = 0

        # Terminal early to save computation
        for i, block in enumerate(net._blocks):
            x = block(x, i * drop)

            if px.shape[-2:] != x.shape[-2:]:
                record.append((f'reduction_{len(record)+1}', (pi, i+1)))

                pi = i + 1
                px = x

        print(model_name, ':', {k: v for k, v in record})

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

# Precomputed aliases
MODELS = {
    'efficientnet-b0': [
        ('reduction_1', (0, 2)),
        ('reduction_2', (2, 4)),
        ('reduction_3', (4, 6)),
        ('reduction_4', (6, 12))
    ],
    'efficientnet-b4': [
        ('reduction_1', (0, 3)),
        ('reduction_2', (3, 7)),
        ('reduction_3', (7, 11)),
        ('reduction_4', (11, 23)),
    ],
    'efficientnet-b7': [
        ('reduction_1', (0, 5)),
        ('reduction_2', (5, 10)),
        ('reduction_3', (10, 20)),
        ('reduction_4', (20, 35))
    ]
}


class SequentialWithArgs(nn.Sequential):
    def __init__(self, *layers_args):
        layers = [layer for layer, args in layers_args]
        args = [args for layer, args in layers_args]
        super().__init__(*layers)
        self.args = args

    def forward(self, x):
        for l, a in zip(self, self.args):
            x = l(x, *a)
        return x


class EfficientNetExtractor(torch.nn.Module):
    def __init__(self, layer_names, image_height, image_width, model_name='efficientnet-b4'):
        super().__init__()
        assert model_name in MODELS
        assert all(k in [k for k, v in MODELS[model_name]] for k in layer_names)
        idx_max = -1
        layer_to_idx = {}
        for i, (layer_name, _) in enumerate(MODELS[model_name]):
            if layer_name in layer_names:
                idx_max = max(idx_max, i)
                layer_to_idx[layer_name] = i
        net = EfficientNet.from_pretrained(model_name)
        net.set_swish(False)
        drop = net._global_params.drop_connect_rate / len(net._blocks)
        blocks = [nn.Sequential(net._conv_stem, net._bn0, net._swish)]
        for idx in range(idx_max + 1):
            l, r = MODELS[model_name][idx][1]
            block = SequentialWithArgs(*[(net._blocks[i], [i * drop]) for i in range(l, r)])
            blocks.append(block)
        self.layers = nn.Sequential(*blocks)
        self.layer_names = layer_names
        self.idx_pick = [layer_to_idx[l] + 1 for l in layer_names]
        with torch.no_grad():
            dummy = torch.rand(1, 3, image_height, image_width)
            output_shapes = [x.shape for x in self(dummy)]
        self.output_shapes = output_shapes

    def forward(self, x):
        if self.training:
            x = x.requires_grad_(True)
        result = []
        for layer in self.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
            result.append(x)
        return [result[i] for i in self.idx_pick]


# ==========================================================
# (최종 수정) 상속 대신 포함(Composition) 구조로 변경하여 문제 해결
# ==========================================================
class UpgradedEfficientNetExtractor(nn.Module): # 더 이상 EfficientNetExtractor를 상속하지 않음
    def __init__(self, layer_names, image_height, image_width,
                 model_name='efficientnet-b7',
                 target_model_name_for_shape='efficientnet-b4'):
        
        # 1. PyTorch 규칙: 가장 먼저 super().__init__() 호출
        super().__init__()

        # 2. 내부에 실제 고성능 백본(b7)을 포함시킴
        print(f"Upgraded Wrapper: Initializing '{model_name}' backbone internally.")
        self.backbone = EfficientNetExtractor(
            layer_names, image_height, image_width, model_name
        )
        
        # 3. 목표 shape(b4 기준)를 얻기 위해 임시 객체 사용
        print(f"Calculating target shapes from '{target_model_name_for_shape}'...")
        with torch.no_grad():
            dummy_shape_extractor = EfficientNetExtractor(
                layer_names, image_height, image_width, target_model_name_for_shape
            )
            target_output_shapes = dummy_shape_extractor.output_shapes
        
        # 4. 실제 shape(b7 기준)는 내부 백본에서 직접 가져옴
        actual_output_shapes = self.backbone.output_shapes
        
        # 5. 채널 프로젝터 생성
        self.channel_projectors = nn.ModuleList()
        print("--- Comparing Backbone Channels ---")
        for i, actual_shape in enumerate(actual_output_shapes):
            target_shape = target_output_shapes[i]
            actual_channels = actual_shape[1]
            target_channels = target_shape[1]
            
            print(f"Layer {i}: Actual({model_name})={actual_channels} vs Target({target_model_name_for_shape})={target_channels}")

            if actual_channels != target_channels:
                self.channel_projectors.append(
                    nn.Conv2d(actual_channels, target_channels, kernel_size=1, bias=False)
                )
            else:
                self.channel_projectors.append(nn.Identity())
        
        # 6. 이 클래스가 외부에 보고할 최종 shape 정보 설정
        self.output_shapes = target_output_shapes
        self.target_output_shapes = target_output_shapes # forward에서 사용하기 위해 저장

        print(f"Final reported output shapes will be: {[s for s in self.output_shapes]}")

    def forward(self, x):
        # 1. 내부 백본(b7)을 통해 피처 추출
        features = self.backbone(x)
        
        final_features = []
        for i, feature_map in enumerate(features):
            # 2. 채널 수 맞추기
            projected_map = self.channel_projectors[i](feature_map)
            
            # 3. 높이/너비 맞추기
            target_shape = self.target_output_shapes[i]
            resized_map = F.interpolate(
                projected_map, 
                size=(target_shape[2], target_shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
            final_features.append(resized_map)
            
        return final_features
