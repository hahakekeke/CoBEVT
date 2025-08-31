import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional
from .decoder import DecoderBlock

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)  # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)  # 3 h w
    return indices[None]  # 1 3 h w

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

class BEVEmbedding(nn.Module):
    def __init__(self, dim: int, sigma: int, bev_height: int, bev_width: int,
                 h_meters: int, w_meters: int, offset: int, upsample_scales: list):
        super().__init__()
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
        V_inv = torch.FloatTensor(V).inverse()
        for i, scale in enumerate(upsample_scales):
            h = bev_height // scale
            w = bev_width // scale
            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]
            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)
            self.register_buffer('grid%d'%i, grid, persistent=False)
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height//upsample_scales[0], bev_width//upsample_scales[0])
        )
    def get_prior(self):
        return self.learned_features

class CrossViewSwapAttention(nn.Module):
    """ Feature-level fusion + dynamic ensemble based on object_count """
    def __init__(self, feat_height, feat_width, feat_dim, dim, index,
                 image_height, image_width, qkv_bias, q_win_size, feat_win_size,
                 heads, dim_head, bev_embedding_flag, rel_pos_emb=False,
                 no_image_features=False, skip=True, norm=nn.LayerNorm):
        super().__init__()
        self.skip = skip
        self.bev_embed_flag = bev_embedding_flag[index]
        self.q_win_size = q_win_size[index]
        self.feat_win_size = feat_win_size[index]
        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False)
        )
        self.feature_proj = nn.Sequential(
            nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False)
        ) if not no_image_features else None
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.bev_embed = nn.Conv2d(2, dim, 1) if self.bev_embed_flag else None
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.postnorm = norm(dim)

    def dynamic_weight(self, object_count):
        """ object_count 기반 dynamic weighting (scene complexity-aware) """
        if object_count is None:
            return None
        total = object_count.sum(dim=-1, keepdim=True) + 1e-5
        weights = object_count / total
        return weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (b n 1 1 1 1)

    def forward(self, index, x, bev, feature, I_inv, E_inv, object_count=None):
        b, n, _, _, _ = feature.shape
        # BEV + image embedding
        x_prior = bev.get_prior()
        x_prior = repeat(x_prior, '... -> b ...', b=b)
        feature_flat = rearrange(feature, 'b n d h w -> (b n) d h w')
        key_flat = self.feature_proj(feature_flat) + self.img_embed(feature_flat)
        val_flat = self.feature_linear(feature_flat)
        # reshape for attention
        query = repeat(x[:, None], 'b d h w -> b n d h w', n=n)
        key = rearrange(key_flat, '(b n) d h w -> b n d h w', b=b, n=n)
        val = rearrange(val_flat, '(b n) d h w -> b n d h w', b=b, n=n)
        # local-to-local
        query = rearrange(self.cross_win_attend_1(query, key, val), 'b x y d -> b (x y) d')
        query = query + self.mlp_1(self.prenorm_1(query))
        # feature-level concat + attention fusion
        query = repeat(query, 'b d h w -> b n d h w', n=n)
        query = rearrange(self.cross_win_attend_2(query, key, val), 'b x y d -> b (x y) d')
        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        # Dynamic Ensemble: object_count 기반 soft weighting
        weights = self.dynamic_weight(object_count)
        if weights is not None:
            query = reduce(query * weights, 'b n d -> b d', 'sum')  # soft voting
        else:
            query = query.mean(1)  # 기본 soft voting
        return rearrange(query, 'b d -> b d 1 1')  # BEV-like output

class PyramidAxialEncoder(nn.Module):
    """ Encoder + object_count aware dynamic ensemble """
    def __init__(self, backbone, cross_view, cross_view_swap, bev_embedding,
                 self_attn, dim: list, middle: List[int] = [2,2], scale: float=1.0):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False) if scale < 1.0 else lambda x: x
        assert len(self.backbone.output_shapes) == len(middle)
        self.cross_views = nn.ModuleList([
            CrossViewSwapAttention(
                feat_height=shape[2], feat_width=shape[3], feat_dim=shape[1], dim=dim[i],
                index=i, **cross_view, **cross_view_swap
            ) for i, shape in enumerate([self.down(torch.zeros(shape)).shape for shape in self.backbone.output_shapes])
        ])
        self.layers = nn.ModuleList([
            nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            for i, num_layers in enumerate(middle)
        ])
        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0,1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        object_count = batch.get('object_count', None)
        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)
        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
        return x
