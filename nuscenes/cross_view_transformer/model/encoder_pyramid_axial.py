import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

# =================================================================================
# 기존 모델 코드
# - PyramidAxialEncoder는 개별 "전문가" 모델의 역할을 합니다.
# - object_count를 하위 모듈까지 전달하는 로직은 이미 구현되어 있어 그대로 활용합니다.
# =================================================================================

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
        [0., -sw, w/2.],
        [-sh, 0., h*offset+h/2.],
        [0., 0., 1.]
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
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'
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
        # b d h w -> b (h w) d
        x_rearranged = rearrange(x, 'b d h w -> b (h w) d')

        q, k, v = self.to_qkv(x_rearranged).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # b (h w) d -> b d h w
        return rearrange(out, 'b (h w) d -> b d h w', h=x.shape[2], w=x.shape[3])

# =================================================================================
# <<<<<<<<<<<<<<<<<<<< 오류 수정: 누락된 CrossWinAttention 클래스 추가 >>>>>>>>>>>>>>>>>>
# =================================================================================
class CrossWinAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb
        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.proj = nn.Linear(heads * dim_head, dim)

    def add_rel_pos_emb(self, x):
        return x

    def forward(self, q, k, v, skip=None):
        assert k.shape == v.shape
        _, view_size, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        q = rearrange(q, 'b l Q (h d) -> (b h) l Q d', h=self.heads)
        k = rearrange(k, 'b l K (h d) -> (b h) l K d', h=self.heads)
        v = rearrange(v, 'b l K (h d) -> (b h) l K d', h=self.heads)

        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)
        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        
        att = dot.softmax(dim=-1)
        a = torch.einsum('b l Q K, b l K d -> b l Q d', att, v)
        a = rearrange(a, '(b h) l Q d -> b l Q (h d)', h=self.heads)
        
        a = rearrange(a, 'b (x y) (n w1 w2) d -> b n x y w1 w2 d',
            x=q_height, y=q_width, n=view_size, w1=q_win_height, w2=q_win_width)
        
        z = self.proj(a)
        z = z.mean(1)
        
        if skip is not None:
            z = z + skip
        return z

class CrossViewSwapAttention(nn.Module):
    def __init__(self, feat_height: int, feat_width: int, feat_dim: int, dim: int, index: int, image_height: int, image_width: int, qkv_bias: bool, q_win_size: list, feat_win_size: list, heads: list, dim_head: list, bev_embedding_flag: list, rel_pos_emb: bool = False, no_image_features: bool = False, skip: bool = True, norm=nn.LayerNorm):
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
        self.rel_pos_emb = rel_pos_emb
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)
    
    def pad_divisible(self, x, win_h, win_w):
        """Pad the x to be divible by window size."""
        _, _, _, h, w = x.shape
        pad_h = (win_h - h % win_h) % win_h
        pad_w = (win_w - w % win_w) % win_w
        return F.pad(x, (0, pad_w, 0, pad_h), value=0)

    def forward(self, index: int, x: torch.FloatTensor, bev: BEVEmbedding, feature: torch.FloatTensor, I_inv: torch.FloatTensor, E_inv: torch.FloatTensor, object_count: Optional[torch.Tensor] = None):
        b, n, _, _, _ = feature.shape
        _, _, H, W = x.shape
        pixel = self.image_plane
        _, _, _, h, w = pixel.shape
        c = E_inv[..., -1:]
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        c_embed = self.cam_embed(c_flat)
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
        d = E_inv @ cam
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)
        d_embed = self.img_embed(d_flat)
        img_embed = d_embed - c_embed
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)
        
        world = getattr(bev, f'grid{index}')[:2]

        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        key_flat = img_embed + self.feature_proj(feature_flat) if self.feature_proj is not None else img_embed
        val_flat = self.feature_linear(feature_flat)

        query = query_pos + x[:, None] if self.bev_embed_flag else x[:, None]
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)

        key = self.pad_divisible(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisible(val, self.feat_win_size[0], self.feat_win_size[1])

        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        skip_conn = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query_out = self.cross_win_attend_1(query, key, val, skip=skip_conn)
        query = rearrange(query_out, 'b x y w1 w2 d -> b (x w1) (y w2) d')

        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)

        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        skip_conn_2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query_out = self.cross_win_attend_2(query, key, val, skip=skip_conn_2)
        query = rearrange(query_out, 'b x y w1 w2 d -> b (x w1) (y w2) d')

        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        query = rearrange(query, 'b H W d -> b d H W')
        return query


class PyramidAxialEncoder(nn.Module):
    def __init__(self, backbone, cross_view: dict, cross_view_swap: dict, bev_embedding: dict, self_attn: dict, dim: list, middle: List[int] = [2, 2], scale: float = 1.0):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        self.down = (lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)) if scale < 1.0 else (lambda x: x)
        assert len(self.backbone.output_shapes) == len(middle)
        cross_views, layers, downsample_layers = list(), list(), list()

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(1, *feat_shape[1:])).shape
            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)
            layers.append(nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)]))
            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i] * 2, kernel_size=3, stride=2, padding=1, bias=False), # Adjusted for simplicity
                    nn.BatchNorm2d(dim[i] * 2), nn.ReLU(inplace=True)
                ))
        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        object_count = batch.get('object_count', None)
        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = repeat(self.bev_embedding.get_prior(), '... -> b ...', b=b)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)
        return x

# =================================================================================
# <<<<<<<<<<<<<<<<<<<<<<<< 새로운 앙상블 아키텍처 구현 >>>>>>>>>>>>>>>>>>>>>>>>>
# =================================================================================

class BEVDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, num_classes, 1)
        )
    def forward(self, x):
        return self.classifier(x)

class DynamicEnsemble(nn.Module):
    def __init__(
        self,
        expert_encoders: List[PyramidAxialEncoder],
        decoder: BEVDecoder,
        ensemble_type: str = 'dynamic',
        fusion_dim: int = 128,
        object_count_dim: int = 8,
    ):
        super().__init__()
        assert ensemble_type in ['soft_voting', 'attention_fusion', 'dynamic']
        self.ensemble_type = ensemble_type
        self.num_experts = len(expert_encoders)
        self.expert_encoders = nn.ModuleList(expert_encoders)
        self.decoder = decoder
        
        if self.ensemble_type == 'attention_fusion':
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(self.num_experts * fusion_dim, fusion_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fusion_dim),
                nn.ReLU(),
                Attention(dim=fusion_dim)
            )
        elif self.ensemble_type == 'dynamic':
            self.gating_network = nn.Sequential(
                nn.Linear(object_count_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_experts),
                nn.Softmax(dim=-1)
            )

    def forward(self, batch):
        if self.ensemble_type == 'soft_voting':
            outputs = [self.decoder(expert(batch)) for expert in self.expert_encoders]
            return torch.mean(torch.stack(outputs, dim=0), dim=0)

        elif self.ensemble_type == 'attention_fusion':
            expert_features = [expert(batch) for expert in self.expert_encoders]
            concatenated_features = torch.cat(expert_features, dim=1)
            fused_features = self.feature_fusion(concatenated_features)
            return self.decoder(fused_features)

        elif self.ensemble_type == 'dynamic':
            object_count = batch.get('object_count')
            if object_count is None:
                raise ValueError("DynamicEnsemble requires 'object_count' in the batch.")
            
            weights = self.gating_network(object_count.float())
            expert_features = [expert(batch) for expert in self.expert_encoders]
            stacked_features = torch.stack(expert_features, dim=1)
            broadcastable_weights = weights.view(-1, self.num_experts, 1, 1, 1)
            fused_features = torch.sum(stacked_features * broadcastable_weights, dim=1)
            return self.decoder(fused_features)


if __name__ == "__main__":
    # 아래는 모델 사용 예시입니다.
    # 실제 사용 시에는 설정 파일(yaml)로부터 파라미터를 로드해야 합니다.
    import yaml
    import re

    # --- 설정 로드 (기존 코드와 동일) ---
    def load_yaml_file(file_path):
        with open(file_path, 'r') as stream:
            # A simple loader is used here for demonstration
            return yaml.safe_load(stream)

    # 이 예시에서는 설정 파일을 직접 생성하지 않고, 필요한 파라미터만 더미로 만듭니다.
    # 실제로는 'config/model/cvt_pyramid_swap.yaml' 같은 파일을 로드해야 합니다.
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            # 더미 백본: 4개의 다른 해상도를 가진 특징맵을 출력한다고 가정
            self.output_shapes = [(1, 64, 112, 112), (1, 128, 56, 56), (1, 256, 28, 28), (1, 512, 14, 14)]
        def forward(self, x):
            b, _, _, _ = x.shape
            return [torch.rand(b, *s[1:]) for s in self.output_shapes]

    # --- 모델 인스턴스화 ---
    NUM_EXPERTS = 3
    FUSION_DIM = 128 # PyramidAxialEncoder의 최종 출력 차원
    NUM_CLASSES = 10 # 예시 클래스 수
    OBJECT_COUNT_DIM = 8 # 예시 객체 종류 수
    
    # 더미 파라미터
    dummy_params = {
        'cross_view': {'image_height': 224, 'image_width': 224, 'qkv_bias': True},
        'cross_view_swap': {
            'q_win_size': [[25, 25], [25, 25]], 'feat_win_size': [[28, 28], [14, 14]],
            'heads': [4, 4], 'dim_head': [32, 32], 'bev_embedding_flag': [True, True]
        },
        'bev_embedding': {
            'sigma': 1.0, 'bev_height': 100, 'bev_width': 100, 'h_meters': 100,
            'w_meters': 100, 'offset': 0.0, 'upsample_scales': [1, 2, 4, 8]
        },
        'self_attn': {},
        'dim': [64, 128], # `middle`의 길이에 맞춰야 함
        'middle': [2, 2],
        'scale': 1.0,
    }

    # 3개의 전문가 인코더 생성
    expert_encoders = [
        PyramidAxialEncoder(backbone=DummyBackbone(), **dummy_params) for _ in range(NUM_EXPERTS)
    ]
    
    # BEV 디코더 생성
    bev_decoder = BEVDecoder(in_channels=FUSION_DIM, num_classes=NUM_CLASSES)

    # Dynamic Ensemble 모델 생성 (최적 방안)
    dynamic_model = DynamicEnsemble(
        expert_encoders=expert_encoders,
        decoder=bev_decoder,
        ensemble_type='dynamic', # 'soft_voting', 'attention_fusion', 'dynamic' 중 선택
        fusion_dim=FUSION_DIM,
        object_count_dim=OBJECT_COUNT_DIM
    )
    
    print("="*50)
    print(f"앙상블 모델 생성 완료 (전략: {dynamic_model.ensemble_type})")
    print(f"전문가 모델 수: {dynamic_model.num_experts}")
    print("="*50)

    # --- 더미 입력 데이터 생성 및 모델 실행 ---
    BATCH_SIZE = 4
    NUM_CAMS = 6
    
    dummy_batch = {
        'image': torch.rand(BATCH_SIZE, NUM_CAMS, 3, 224, 224),
        'intrinsics': torch.rand(BATCH_SIZE, NUM_CAMS, 3, 3),
        'extrinsics': torch.rand(BATCH_SIZE, NUM_CAMS, 4, 4),
        # object_count: [차, 트럭, 보행자, ...] 등의 객체 수를 담은 벡터
        'object_count': torch.randint(0, 50, (BATCH_SIZE, OBJECT_COUNT_DIM))
    }

    # 모델 실행
    output = dynamic_model(dummy_batch)
    
    # 출력 형태 확인 (Batch, NumClasses, Height, Width)
    print(f"최종 출력 Shape: {output.shape}")
    assert output.shape == (BATCH_SIZE, NUM_CLASSES, 25, 25) # 최종 BEV 해상도에 따라 달라짐
