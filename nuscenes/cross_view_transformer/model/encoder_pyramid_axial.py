import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings
import copy

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

# =================================================================================
# 헬퍼 함수 및 경량 모델 클래스 (CrossView 기반)
# =================================================================================
ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

def generate_grid(height: int, width: int):
    """주어진 높이와 너비에 대한 정규화된 좌표 그리드를 생성합니다."""
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)
    indices = indices[None]
    return indices

def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """BEV 그리드 좌표를 실제 세계 좌표로 변환하기 위한 행렬을 생성합니다."""
    sh = h / h_meters
    sw = w / w_meters
    return torch.tensor([
        [0., -sw, w / 2.],
        [-sh, 0., h * offset + h / 2.],
        [0., 0., 1.]
    ])

class Normalize(nn.Module):
    """입력 이미지를 정규화하는 모듈입니다."""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std

class BEVEmbedding(nn.Module):
    """학습 가능한 BEV 그리드 임베딩과 좌표계를 관리하는 모듈입니다."""
    def __init__(self, dim: int, sigma: int, bev_height: int, bev_width: int, h_meters: int,
                 w_meters: int, offset: int, upsample_scales: list):
        super().__init__()
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
        V_inv = V.inverse()
        for i, scale in enumerate(upsample_scales):
            h, w = bev_height // scale, bev_width // scale
            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]
            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)
            self.register_buffer(f'grid{i}', grid, persistent=False)
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height // upsample_scales[0], bev_width // upsample_scales[0]))
    def get_prior(self):
        return self.learned_features

class CrossWinAttention(nn.Module):
    """Window 기반의 Cross-Attention을 수행하는 저수준 모듈입니다."""
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
        _, _, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q = rearrange(q, 'b n q (h d) -> (b h) n q d', h=self.heads)
        k = rearrange(k, 'b n k (h d) -> (b h) n k d', h=self.heads)
        v = rearrange(v, 'b n v (h d) -> (b h) n v d', h=self.heads)
        dot = self.scale * torch.einsum('b n q d, b n k d -> b n q k', q, k)
        att = dot.softmax(dim=-1)
        a = torch.einsum('b n q k, b n v d -> b n q d', att, v)
        a = rearrange(a, '(b h) n q d -> b n q (h d)', h=self.heads)
        a = rearrange(a, 'b (x y) (n w1 w2) d -> b n x y w1 w2 d',
                      x=q_height, y=q_width, w1=q_win_height, n=v.shape[1] // (q_win_height * q_win_width))
        z = self.proj(a).mean(1)
        if skip is not None:
            z = z + skip
        return z

class CrossViewSwapAttention(nn.Module):
    """경량 모델의 핵심 트랜스포머 레이어입니다."""
    def __init__(self, feat_height, feat_width, feat_dim, dim, index, image_height, image_width,
                 qkv_bias, q_win_size, feat_win_size, heads, dim_head, bev_embedding_flag,
                 skip=True, norm=nn.LayerNorm, **kwargs):
        super().__init__()
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)
        self.feature_linear = nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        self.feature_proj = nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag: self.bev_embed = nn.Conv2d(2, dim, 1)
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

    def pad_divisible(self, x, win_h, win_w):
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h - 1) // win_h) * win_h, ((w + win_w - 1) // win_w) * win_w
        return F.pad(x, (0, w_pad - w, 0, h_pad - h))

    def forward(self, index: int, x: torch.FloatTensor, bev: BEVEmbedding, feature: torch.FloatTensor,
                I_inv: torch.FloatTensor, E_inv: torch.FloatTensor, object_count: Optional[torch.Tensor] = None):
        b, n, _, h_feat, w_feat = feature.shape
        pixel = self.image_plane
        
        c = E_inv[..., -1:]
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        c_embed = self.cam_embed(c_flat)

        cam = I_inv @ pixel.flatten(3)
        cam = F.pad(cam, (0, 0, 0, 1), value=1)
        d = E_inv @ cam
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h_feat, w=w_feat)
        d_embed = self.img_embed(d_flat)
        img_embed = d_embed - c_embed
        img_embed = F.normalize(img_embed, dim=1)
        
        world = getattr(bev, f'grid{index}')[:2]
        query_pos = None
        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])
            bev_embed = F.normalize(w_embed - c_embed, dim=1)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)
            
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        key_flat = img_embed + self.feature_proj(feature_flat)
        val_flat = self.feature_linear(feature_flat)
        
        query = query_pos + x[:, None] if self.bev_embed_flag else x[:, None]
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)
        
        key = self.pad_divisible(key, *self.feat_win_size)
        val = self.pad_divisible(val, *self.feat_win_size)
        
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        skip_conn = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query = rearrange(self.cross_win_attend_1(query, key, val, skip=skip_conn), 'b x y w1 w2 d -> b (x w1) (y w2) d')
        
        query = query + self.mlp_1(self.prenorm_1(query))
        
        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        
        key = rearrange(rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d'), 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d'), 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        skip_conn_2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query = rearrange(self.cross_win_attend_2(query, key, val, skip=skip_conn_2), 'b x y w1 w2 d -> b (x w1) (y w2) d')
        
        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        
        return rearrange(query, 'b H W d -> b d H W')

class PyramidAxialEncoder(nn.Module):
    """경량 모델의 전체 구조를 책임지는 메인 클래스입니다."""
    def __init__(self, backbone, cross_view: dict, cross_view_swap: dict, bev_embedding: dict,
                 dim: list, middle: List[int], scale: float = 1.0, **kwargs):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        self.down = (lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)) if scale < 1.0 else (lambda x: x)
        
        if not hasattr(self.backbone, 'output_shapes'):
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 480)
                # 백본마다 forward 방식이 다를 수 있어, 일반화된 호출을 가정합니다.
                # 실제 사용하는 백본에 맞게 수정이 필요할 수 있습니다.
                try:
                    dummy_feats = self.backbone(dummy_input)
                    if isinstance(dummy_feats, torch.Tensor): dummy_feats = [dummy_feats]
                except Exception:
                    # Some backbones (like SwinT) are more complex
                    dummy_feats = self.backbone.forward_features(dummy_input)
                    if isinstance(dummy_feats, torch.Tensor): dummy_feats = [dummy_feats]

                self.backbone.output_shapes = [f.shape for f in dummy_feats]

        cross_views, layers, downsample_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(1, *feat_shape[1:])).shape
            cva = CrossViewSwapAttention(feat_height=feat_height, feat_width=feat_width, feat_dim=feat_dim, dim=dim[i], index=i, **cross_view, **cross_view_swap)
            cross_views.append(cva)
            layers.append(nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)]))
            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i+1]*2, 3, padding=1, bias=False), nn.PixelUnshuffle(2),
                    nn.BatchNorm2d(dim[i+1]), nn.ReLU(inplace=True)))
        
        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views, self.layers, self.downsample_layers = cross_views, layers, downsample_layers

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image, I_inv, E_inv = batch['image'].flatten(0, 1), batch['intrinsics'].inverse(), batch['extrinsics'].inverse()
        object_count = batch.get('object_count', None)
        features = self.backbone(self.norm(image))
        if isinstance(features, torch.Tensor): features = [features]
        features = [self.down(y) for y in features]
        
        x = repeat(self.bev_embedding.get_prior(), '... -> b ...', b=b)
        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features) - 1:
                x = self.downsample_layers[i](x)
        return x




# =================================================================================
# MMCV 의존성을 제거한 BEVFormer 고정밀 모델
# =================================================================================

def pure_torch_ms_deform_attn(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    MMCV 없이 순수 PyTorch로 MultiScaleDeformable Attention을 구현한 버전.
    F.grid_sample을 사용하여 구현.
    """
    bs, _, num_heads, embed_dims_per_head = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    output = torch.zeros((bs, num_queries, num_heads, embed_dims_per_head), dtype=value.dtype, device=value.device)
    
    for lvl, (H, W) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lvl].transpose(1, 2).reshape(bs * num_heads, -1, H, W)
        sampling_grid_l_ = sampling_grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)
        sampled_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        attention_weights_l_ = attention_weights[:, :, :, lvl].transpose(1, 2).flatten(0, 1)
        output_l_ = (sampled_value_l_ * attention_weights_l_.unsqueeze(1)).sum(-1).transpose(1, 2)
        output_l_ = output_l_.reshape(bs, num_heads, num_queries, embed_dims_per_head).transpose(1, 2)
        output += output_l_
        
    return output.contiguous()

class MSDeformableAttention3D(nn.Module):
    """Deformable Attention의 핵심 로직을 담고 있는 모듈입니다."""
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=8, **kwargs):
        super().__init__()
        if embed_dims % num_heads != 0: raise ValueError('embed_dims must be divisible by num_heads')
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points): grid_init[:, :, i, :] *= i + 1
        with torch.no_grad(): self.sampling_offsets.bias.copy_(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, value, query_pos=None, reference_points=None, spatial_shapes=None, level_start_index=None, **kwargs):
        if query_pos is not None: query = query + query_pos
        value = self.value_proj(value)
        bs, num_query, _ = query.shape
        bs, len_val, _ = value.shape
        value = value.view(bs, len_val, self.num_heads, -1)
        
        sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points).softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        
        output = pure_torch_ms_deform_attn(value, spatial_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)

class SpatialCrossAttention(nn.Module):
    """공간적 정보(카메라 이미지)를 융합하는 어텐션 모듈입니다."""
    def __init__(self, embed_dims=256, num_cams=6, dropout=0.1, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformableAttention3D(embed_dims=embed_dims, **kwargs)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.num_cams = num_cams

    def forward(self, query, value, query_pos=None, reference_points_cam=None, **kwargs):
        bs, num_query, _ = query.shape
        # (bs, n_cams, sum(H*W), C) -> (bs*n_cams, sum(H*W), C)
        value = rearrange(value, 'b n hw c -> (b n) hw c')
        
        # 각 카메라에 대해 BEV 쿼리가 어텐션하도록 확장
        query_rebatch = query.repeat_interleave(self.num_cams, dim=0)
        reference_points_rebatch = rearrange(reference_points_cam, 'b n q ... -> (b n) q ...')
        
        output = self.deformable_attention(query=query_rebatch, value=value, query_pos=None, reference_points=reference_points_rebatch, **kwargs)
        output = rearrange(output, '(b n) q c -> b n q c', n=self.num_cams)
        
        # 카메라 축으로 평균내어 정보 융합
        output = output.mean(dim=1)
        return self.dropout(self.output_proj(query + output))

class TemporalSelfAttention(nn.Module):
    """시간적 정보(움직임)를 융합하는 어텐션 모듈입니다."""
    def __init__(self, embed_dims=256, num_bev_queue=2, **kwargs):
        super().__init__()
        self.deformable_attention = MSDeformableAttention3D(embed_dims=embed_dims, num_levels=1, **kwargs)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.num_bev_queue = num_bev_queue

    def forward(self, query, query_pos, prev_bev, **kwargs):
        bs, num_query, _ = query.shape
        value = torch.cat([prev_bev, query], dim=0) # (bs*queue, num_query, C)
        
        # temporal self-attention은 BEV-BEV 이므로 query와 value가 동일
        output = self.deformable_attention(query, value, query_pos=query_pos, **kwargs)
        return self.output_proj(output)

class BEVFormerLayer(nn.Module):
    """BEVFormer의 인코더 레이어입니다."""
    def __init__(self, embed_dims, temporal_attn_cfg, spatial_attn_cfg, ffn_cfg, **kwargs):
        super().__init__()
        self.temporal_attn = TemporalSelfAttention(embed_dims=embed_dims, **temporal_attn_cfg)
        self.spatial_attn = SpatialCrossAttention(embed_dims=embed_dims, **spatial_attn_cfg)
        self.ffn = nn.Sequential(nn.Linear(embed_dims, ffn_cfg['feedforward_channels']), nn.GELU(), nn.Linear(ffn_cfg['feedforward_channels'], embed_dims))
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, query, key, value, bev_pos, prev_bev, ref_2d, ref_3d, reference_points_cam, spatial_shapes, level_start_index, **kwargs):
        query = query + self.temporal_attn(self.norm1(query), query_pos=bev_pos, prev_bev=prev_bev, reference_points=ref_2d, spatial_shapes=spatial_shapes, level_start_index=level_start_index)
        query = query + self.spatial_attn(self.norm2(query), value=value, query_pos=bev_pos, reference_points_cam=reference_points_cam, spatial_shapes=spatial_shapes, level_start_index=level_start_index)
        query = query + self.ffn(self.norm3(query))
        return query

class BEVFormerEncoder(nn.Module):
    """고정밀 모델의 전체 구조를 책임지는 메인 클래스입니다."""
    def __init__(self, embed_dims, num_layers, layer_cfg, pc_range, num_points_in_pillar, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([BEVFormerLayer(embed_dims=embed_dims, **layer_cfg) for _ in range(num_layers)])
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1).permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            return ref_3d[None].repeat(bs, 1, 1, 1)
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                                        torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device), indexing='ij')
            ref_2d = torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), -1)
            return ref_2d.view(1, H*W, 1, 2).repeat(bs, 1, 1, 1)

    def point_sampling(self, reference_points, pc_range, img_metas):
        lidar2img = torch.stack([torch.tensor(m['lidar2img'], device=reference_points.device, dtype=reference_points.dtype) for m in img_metas], dim=0)
        reference_points_world = reference_points.clone()
        reference_points_world[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points_world[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points_world[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        reference_points_world = torch.cat((reference_points_world, torch.ones_like(reference_points_world[..., :1])), -1)
        
        B, num_cam = lidar2img.shape[:2]
        D, num_query = reference_points.shape[1], reference_points.shape[2]
        reference_points_world = reference_points_world.view(B, 1, D, num_query, 4).repeat(1, num_cam, 1, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(B, num_cam, 1, 1, 4, 4).repeat(1, 1, D, num_query, 1, 1)
        
        reference_points_cam = torch.matmul(lidar2img, reference_points_world).squeeze(-1)
        eps = 1e-5
        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        
        img_shape = torch.tensor(img_metas[0]['img_shape'], device=reference_points_cam.device)
        reference_points_cam[..., 0] /= img_shape[:, 1:2].view(1, num_cam, 1, 1, 1)
        reference_points_cam[..., 1] /= img_shape[:, 0:1].view(1, num_cam, 1, 1, 1)
        
        bev_mask = bev_mask & (reference_points_cam[..., 1:2] > 0.0) & (reference_points_cam[..., 1:2] < 1.0) & \
                     (reference_points_cam[..., 0:1] < 1.0) & (reference_points_cam[..., 0:1] > 0.0)
        bev_mask = torch.nan_to_num(bev_mask)
        return reference_points_cam.permute(0, 1, 3, 2, 4), bev_mask.permute(0, 1, 3, 2, 4).squeeze(-1)

    def forward(self, bev_query, key, value, bev_h, bev_w, bev_pos, prev_bev, img_metas, spatial_shapes, level_start_index, **kwargs):
        output = bev_query
        bs = bev_query.shape[0]
        
        ref_3d = self.get_reference_points(bev_h, bev_w, self.pc_range[5] - self.pc_range[2], self.num_points_in_pillar, '3d', bs, bev_query.device, bev_query.dtype)
        ref_2d = self.get_reference_points(bev_h, bev_w, dim='2d', bs=bs, device=bev_query.device, dtype=bev_query.dtype)
        reference_points_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas)
        
        for layer in self.layers:
            output = layer(output, key, value, bev_pos=bev_pos, prev_bev=prev_bev, ref_2d=ref_2d,
                           ref_3d=ref_3d, reference_points_cam=reference_points_cam,
                           spatial_shapes=spatial_shapes, level_start_index=level_start_index, **kwargs)
        return output

# =================================================================================
# 최종 하이브리드 모델 (MMCV 의존성 없음)
# =================================================================================
class HybridCameraBEVModel(nn.Module):
    def __init__(self, backbone, light_model_cfg: dict, heavy_model_cfg: dict,
                 bev_h, bev_w, embed_dims):
        super().__init__()
        self.bev_h, self.bev_w, self.embed_dims = bev_h, bev_w, embed_dims
        self.backbone = backbone
        self.light_model = PyramidAxialEncoder(backbone=self.backbone, **light_model_cfg)
        self.heavy_model = BEVFormerEncoder(embed_dims=embed_dims, **heavy_model_cfg)
        self.bev_queries = nn.Parameter(torch.randn(bev_h * bev_w, embed_dims))
        self.bev_pos = nn.Parameter(torch.randn(1, embed_dims, bev_h, bev_w))
        self.prev_bev = None
        self.object_count_threshold = 30
        
    def forward(self, batch):
        object_count = batch.get('object_count', None)
        bs = batch['image'].shape[0]
        
        use_heavy_model = (object_count is not None) and (torch.sum(object_count) >= self.object_count_threshold)
        print(f"Total objects: {torch.sum(object_count) if object_count is not None else 'N/A'}. Using {'HEAVY' if use_heavy_model else 'LIGHT'} model.")
        
        if use_heavy_model:
            image = batch['image'].flatten(0, 1)
            features = self.backbone(self.light_model.norm(image))
            if isinstance(features, torch.Tensor): features = [features]
            mlvl_feats = [self.light_model.down(y) for y in features]
            n_cams = batch['image'].shape[1]
            
            feat_flatten, spatial_shapes = [], []
            for lvl, feat in enumerate(mlvl_feats):
                feat = feat.view(bs, n_cams, *feat.shape[1:])
                feat_flatten.append(feat)
                spatial_shapes.append(feat.shape[3:])
            
            value = torch.cat([f.flatten(3) for f in feat_flatten], dim=3).permute(0, 1, 3, 2)
            
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=value.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            
            bev_queries = self.bev_queries.unsqueeze(0).repeat(bs, 1, 1)
            bev_pos = self.bev_pos.flatten(2).permute(0, 2, 1).repeat(bs, 1, 1)
            prev_bev = self.prev_bev if self.prev_bev is not None else torch.zeros_like(bev_queries)

            if 'img_metas' not in batch:
                batch['img_metas'] = [{'lidar2img': np.tile(np.eye(4, dtype=np.float32), (n_cams, 1, 1)),
                                       'img_shape': [[224, 480]] * n_cams} for _ in range(bs)]

            bev_embed = self.heavy_model(
                bev_query=bev_queries, key=None, value=value,
                bev_h=self.bev_h, bev_w=self.bev_w, bev_pos=bev_pos,
                spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                prev_bev=prev_bev, img_metas=batch['img_metas'])
            
            self.prev_bev = bev_embed.detach()
            return bev_embed.permute(0, 2, 1).view(bs, self.embed_dims, self.bev_h, self.bev_w)
            
        else:
            self.prev_bev = None
            return self.light_model(batch)
