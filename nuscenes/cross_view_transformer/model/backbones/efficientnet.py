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
import torch.nn.functional as F  # 리사이징을 위해 import
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
    # ==========================================================
    # (추가된 부분) efficientnet-b7에 대한 블록 인덱스 정보
    # ==========================================================
    'efficientnet-b7': [
        ('reduction_1', (0, 5)),
        ('reduction_2', (5, 10)),
        ('reduction_3', (10, 20)),
        ('reduction_4', (20, 35))
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
        for idx in range(idx_max + 1): # +1 to include the last reduction stage's blocks
            l, r = MODELS[model_name][idx][1]

            block = SequentialWithArgs(*[(net._blocks[i], [i * drop]) for i in range(l, r)])
            blocks.append(block)

        self.layers = nn.Sequential(*blocks)
        self.layer_names = layer_names
        # the larger resolution result should be at the last position
        self.idx_pick = [layer_to_idx[l] + 1 for l in layer_names] # +1 because of the initial stem block

        # Pass a dummy tensor to precompute intermediate shapes
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
        
        # result[0] is from stem, result[1] is from reduction_1 blocks, etc.
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy = torch.rand(6, 3, 224, 480).to(device)

    # b7을 추가하여 직접 확인할 수도 있습니다.
    for model_name in ['efficientnet-b0', 'efficientnet-b4', 'efficientnet-b7']:
        print(f"--- Analyzing {model_name} ---")
        try:
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
                    record.append((f'reduction_{len(record)+1}', (pi, i)))
                    pi = i
                    px = x
            
            # Add the last block range
            record.append((f'reduction_{len(record)+1}', (pi, len(net._blocks))))

            print(model_name, ':', {k: v for k, v in record})
        except Exception as e:
            print(f"Could not analyze {model_name}: {e}")


# ==========================================================
# (추가된 부분) 리사이징 기능이 포함된 업그레이드 래퍼
# ==========================================================
class UpgradedEfficientNetExtractor(EfficientNetExtractor):
    """
    고성능 백본을 사용하되, 출력 피처맵의 크기는
    기존 기본 백본의 크기와 동일하게 맞춰주는 업그레이드 버전 래퍼.
    """
    def __init__(self, layer_names, image_height, image_width,
                 model_name='efficientnet-b7',          # 사용할 실제 고성능 모델
                 target_model_name_for_shape='efficientnet-b4'): # 크기를 맞출 기준 모델
        
        # 1. 먼저, 실제 고성능 모델(b7) 기준으로 부모 클래스(EfficientNetExtractor)를 초기화
        super().__init__(layer_names, image_height, image_width, model_name)
        
        print(f"Upgraded Wrapper: Using '{model_name}' backbone internally.")
        print(f"                 Resizing outputs to match '{target_model_name_for_shape}' shapes.")

        # 2. 크기 기준이 될 모델(b4)의 출력 shape들을 알아내기 위해 임시로 생성
        #    이렇게 하면 b4의 shape를 하드코딩할 필요 없이 자동으로 알아낼 수 있습니다.
        with torch.no_grad():
            dummy_shape_extractor = EfficientNetExtractor(
                layer_names, image_height, image_width, target_model_name_for_shape
            )
            # 이 래퍼의 최종 목표 shape는 b4의 shape가 됩니다.
            self.target_output_shapes = dummy_shape_extractor.output_shapes
        
        # 부모 클래스에서 계산된 b7의 shape 대신, b4의 shape를 최종 output_shapes로 덮어씁니다.
        # 이렇게 해야 PyramidAxialEncoder가 b4 기준으로 초기화됩니다.
        self.output_shapes = self.target_output_shapes

        print(f"Final output shapes will be: {[s for s in self.output_shapes]}")


    def forward(self, x):
        # 1. 부모 클래스의 forward를 호출하여 고성능 백본(b7)의 피처맵들을 추출
        #    이 피처들은 성능은 좋지만, 크기가 b4와 다릅니다.
        features = super().forward(x)
        
        # 2. 추출된 피처맵들을 목표 shape(b4의 shape)에 맞게 하나씩 리사이즈
        resized_features = []
        for i, feature_map in enumerate(features):
            target_shape = self.target_output_shapes[i]
            
            # (높이, 너비) 정보만 사용하여 리사이즈
            resized_map = F.interpolate(
                feature_map, 
                size=(target_shape[2], target_shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
            resized_features.append(resized_map)
            
        # 3. 크기가 맞춰진 최종 피처맵 리스트를 반환
        return resized_features
