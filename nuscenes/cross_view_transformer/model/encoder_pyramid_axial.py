import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

# ==========================================================================================
# 사용자의 실제 Decoder 구현을 위한 Placeholder 클래스입니다.
# 에러의 원인이 된 `interpolate` 호출 부분을 포함하도록 간단히 구성했습니다.
# 실제 환경에서는 이 부분을 사용자의 `decoder.py` 파일 내용으로 대체해야 합니다.
# = a placeholder class for the user's actual Decoder implementation.
# It is simply configured to include the `interpolate` call that caused the error.
# In a real environment, this part should be replaced with the contents of the user's `decoder.py` file.
# ==========================================================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # This is a simplified placeholder. The user should use their actual implementation.
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # The user's original decoder might have skip connections `forward(self, y, x_skip)`.
        # For fixing the current error, we assume a simple decoder structure.
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        # This is a simplified placeholder.
        self.layer = DecoderBlock(in_channels, in_channels // 2)
        self.final_conv = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.layer(x)
        return self.final_conv(x)
# ==========================================================================================


ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)
    indices = indices[None]
    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters
    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()
        linear = nn.Conv2d(*args, **kwargs)
        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {'stride': stride, 'padding': padding}

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(self, dim: int, sigma: int, bev_height: int, bev_width: int, h_meters: int, w_meters: int, offset: int, upsample_scales: list):
        super().__init__()
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
        V_inv = torch.FloatTensor(V).inverse()
        for i, scale in enumerate(upsample_scales):
            h, w = bev_height // scale, bev_width // scale
            grid = generate_grid(h, w).squeeze(0)
            grid[0], grid[1] = bev_width * grid[0], bev_height * grid[1]
            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)
            self.register_buffer(f'grid{i}', grid, persistent=False)
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, bev_height // upsample_scales[0], bev_width // upsample_scales[0]))

    def get_prior(self):
        return self.learned_features


class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0., window_size=25):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension must be divisible by dimension per head'
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        _, _, height, width, h = *x.shape, self.heads
        x = rearrange(x, 'b d h w -> b (h w) d')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (H W) d -> b H W (h d)', H=height, W=width)
        out = self.to_out(out)
        return rearrange(out, 'b H W d -> b d H W')


class CrossWinAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.proj = nn.Linear(heads * dim_head, dim)

    def forward(self, q, k, v, skip=None):
        _, _, q_h, q_w, q_win_h, q_win_w, _ = q.shape
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h=self.heads), (q, k, v))
        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)
        att = dot.softmax(dim=-1)
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b h) ... d -> b ... (h d)', h=self.heads)
        a = rearrange(a, 'b (x y) (n w1 w2) d -> b n x y w1 w2 d', x=q_h, y=q_w, w1=q_win_h, w2=q_win_w)
        z = self.proj(a).mean(1)
        if skip is not None:
            z = z + skip
        return z


class CrossViewSwapAttention(nn.Module):
    def __init__(self, feat_height, feat_width, feat_dim, dim, index, image_height, image_width, qkv_bias, q_win_size, feat_win_size, heads, dim_head, bev_embedding_flag, no_image_features=False, skip=True, norm=nn.LayerNorm):
        super().__init__()
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)
        self.feature_linear = nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        self.feature_proj = None if no_image_features else nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.q_win_size = q_win_size[index]
        self.feat_win_size = feat_win_size[index]
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)
        self.depth_fusion = nn.Sequential(nn.Conv2d(feat_dim + 1, feat_dim, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True))

    def pad_divisble(self, x, win_h, win_w):
        _, _, _, h, w = x.shape
        padh = (win_h - h % win_h) % win_h
        padw = (win_w - w % win_w) % win_w
        return F.pad(x, (0, padw, 0, padh))

    def forward(self, index, x, bev, feature, I_inv, E_inv, object_count, depth_map=None):
        b, n, _, h_f, w_f = feature.shape
        pixel = self.image_plane
        c_embed = self.cam_embed(rearrange(E_inv[..., -1:], 'b n d c -> (b n) d c 1'))
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1), value=1)
        d = E_inv @ cam
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h_f, w=w_f)
        img_embed = self.img_embed(d_flat) - c_embed
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)
        
        world = getattr(bev, f'grid{index}')[:2]

        query = x[:, None]
        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)
            query = query_pos + query
        
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        if depth_map is not None:
            resized_depth = F.interpolate(depth_map, size=(h_f, w_f), mode='bilinear', align_corners=False)
            feature_flat = self.depth_fusion(torch.cat([feature_flat, resized_depth], dim=1))
        
        key_flat = img_embed
        if self.feature_proj is not None:
            key_flat = key_flat + self.feature_proj(feature_flat)
        val_flat = self.feature_linear(feature_flat)
        
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])
        
        q_h, q_w = x.shape[-2] // self.q_win_size[0], x.shape[-1] // self.q_win_size[1]
        
        query, key, val = map(lambda t: rearrange(t, 'b n d (h dh) (w dw) -> b n h w dh dw d', dh=t.shape[-2]//q_h, dw=t.shape[-1]//q_w), (query, key, val))
        
        skip = rearrange(x, 'b d (h dh) (w dw) -> b h w dh dw d', h=q_h, w=q_w) if self.skip else None
        query = self.cross_win_attend_1(query, key, val, skip=skip)
        query = rearrange(query, 'b h w dh dw d -> b (h dh) (w dw) d')
        query = query + self.mlp_1(self.prenorm_1(query))
        
        x_skip = query
        query = repeat(query, 'b H W d -> b n H W d', n=n)
        
        query = rearrange(query, 'b n (h dh) (w dw) d -> b n h w dh dw d', h=q_h, w=q_w)
        k_h, k_w = key.shape[-4], key.shape[-3] # use padded key/val shape
        key, val = map(lambda t: rearrange(t, 'b n h w dh dw d -> b n dh dw h w d'), (key, val))
        
        skip2 = rearrange(x_skip, 'b (h dh) (w dw) d -> b h w dh dw d', h=q_h, w=q_w) if self.skip else None
        query = self.cross_win_attend_2(query, key, val, skip=skip2)
        query = rearrange(query, 'b h w dh dw d -> b (h dh) (w dw) d')

        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        query = rearrange(query, 'b H W d -> b d H W')
        return query


class DepthDecoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class PyramidAxialEncoder(nn.Module):
    def __init__(self, backbone, cross_view: dict, cross_view_swap: dict, bev_embedding: dict, dim: list, middle: List[int], scale: float = 1.0):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        self.down = (lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)) if scale < 1.0 else (lambda x: x)
        assert len(self.backbone.output_shapes) == len(middle)
        
        first_feat_dim = self.backbone.output_shapes[0][1]
        self.depth_decoder = DepthDecoder(feat_dim=first_feat_dim)

        cross_views, layers, downsample_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(1, *feat_shape[1:])).shape
            cross_views.append(CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap))
            layers.append(nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)]))
            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dim[i+1]),
                    nn.ReLU(inplace=True)
                ))
        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = cross_views
        self.layers = layers
        self.downsample_layers = downsample_layers

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image, I_inv, E_inv = batch['image'].flatten(0, 1), batch['intrinsics'].inverse(), batch['extrinsics'].inverse()
        object_count = batch.get('object_count', None)

        features = [self.down(y) for y in self.backbone(self.norm(image))]
        
        depth_prediction = None
        if object_count is not None and torch.any(object_count >= 30):
            depth_prediction = self.depth_decoder(features[0])

        x = repeat(self.bev_embedding.get_prior(), '... -> b ...', b=b)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count, depth_map=depth_prediction)
            x = layer(x)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)
        return x, depth_prediction


# =====================================================================================
# Encoder와 Decoder를 통합하는 메인 모델 클래스
# Main model class that integrates Encoder and Decoder
# =====================================================================================
class CoBEVT(nn.Module):
    def __init__(self, backbone, encoder_params, decoder_params):
        super().__init__()
        self.encoder = PyramidAxialEncoder(backbone=backbone, **encoder_params)
        # 실제 디코더의 입력 채널은 인코더의 최종 출력 채널과 일치해야 합니다.
        # The input channels of the actual decoder must match the final output channels of the encoder.
        self.decoder = Decoder(in_channels=encoder_params['dim'][-1], **decoder_params)

    def forward(self, batch):
        # Encoder는 (BEV 특징, 깊이 예측) 튜플을 반환합니다.
        # The encoder returns a tuple (BEV features, depth prediction).
        bev_features, depth_prediction = self.encoder(batch)
        
        # Decoder에는 BEV 특징만 전달합니다.
        # Pass only the BEV features to the Decoder.
        bev_prediction = self.decoder(bev_features)
        
        # 두 예측 결과를 모두 반환하여 loss 계산에 사용합니다.
        # Return both prediction results to be used in the loss calculation.
        return bev_prediction, depth_prediction

if __name__ == "__main__":
    # 가짜 백본 클래스 정의 (테스트용) / Fake backbone class definition (for testing)
    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            # 실제 백본의 출력 채널과 해상도에 맞춰야 합니다.
            # Must match the output channels and resolution of the actual backbone.
            self.conv1 = nn.Conv2d(3, 64, 1)
            self.conv2 = nn.Conv2d(64, 128, 1)
            self.output_shapes = [(1, 64, 112, 200), (1, 128, 56, 100)]

        def forward(self, x):
            x = F.interpolate(x, size=(224, 400), mode='bilinear', align_corners=False)
            out1 = self.conv1(x)
            out2 = self.conv2(F.max_pool2d(out1, 2))
            return [out1, out2]

    # 테스트를 위한 설정값 / Configuration values for testing
    B, N, C, H, W = 2, 6, 3, 224, 400
    bev_h, bev_w = 200, 200

    # 모델 초기화 / Model initialization
    backbone = FakeBackbone()
    encoder_params = {
        'cross_view': {'image_height': H, 'image_width': W, 'qkv_bias': True},
        'cross_view_swap': {
            'q_win_size': [[bev_h//(2**i), bev_w//(2**i)] for i in range(2)],
            'feat_win_size': [[14, 25], [7, 13]],
            'heads': [4, 4],
            'dim_head': [32, 64],
            'bev_embedding_flag': [True, False]
        },
        'bev_embedding': {
            'sigma': 1.0, 'bev_height': bev_h, 'bev_width': bev_w,
            'h_meters': 100.0, 'w_meters': 100.0, 'offset': 0.0, 'upsample_scales': [1,2,4,8]
        },
        'dim': [128, 256],
        'middle': [2, 2],
        'scale': 0.5
    }
    decoder_params = {'num_classes': 10} # 예시 클래스 수 / Example number of classes

    # CoBEVT 모델 사용 / Using the CoBEVT model
    model = CoBEVT(backbone, encoder_params, decoder_params)

    # 더미 입력 데이터 생성 / Dummy input data generation
    image = torch.rand(B, N, C, H, W)
    intrinsics = torch.rand(B, N, 3, 3)
    extrinsics = torch.rand(B, N, 4, 4)
    
    # 시나리오 1: 객체 수가 30 미만인 경우 / Scenario 1: Object count < 30
    print("--- Scenario 1: Object count < 30 ---")
    object_count_1 = torch.tensor([16, 25])
    batch_1 = {'image': image, 'intrinsics': intrinsics, 'extrinsics': extrinsics, 'object_count': object_count_1}
    
    bev_pred_1, depth_map_1 = model(batch_1)
    
    print("BEV prediction shape:", bev_pred_1.shape)
    print("Predicted depth map:", depth_map_1)

    print("\n" + "="*40 + "\n")

    # 시나리오 2: 객체 수가 30 이상인 경우 / Scenario 2: Object count >= 30
    print("--- Scenario 2: Object count >= 30 ---")
    object_count_2 = torch.tensor([16, 47])
    batch_2 = {'image': image, 'intrinsics': intrinsics, 'extrinsics': extrinsics, 'object_count': object_count_2}

    bev_pred_2, depth_map_2 = model(batch_2)

    print("BEV prediction shape:", bev_pred_2.shape)
    if depth_map_2 is not None:
        print("Predicted depth map shape:", depth_map_2.shape)
    else:
        print("Predicted depth map is None")
