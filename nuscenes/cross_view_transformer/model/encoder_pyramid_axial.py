import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional
from torch.autograd.function import Function, once_differentiable

# =================================================================================
# 헬퍼 함수 및 기존 모델 클래스 (올바른 순서로 재배열)
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
    return [[0., -sw, w / 2.], [-sh, 0., h * offset + h / 2.], [0., 0., 1.]]

class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std

# *** NameError 해결: BEVEmbedding 클래스를 먼저 정의 ***
class BEVEmbedding(nn.Module):
    def __init__(self, dim: int, sigma: int, bev_height: int, bev_width: int, h_meters: int,
                 w_meters: int, offset: int, upsample_scales: list):
        super().__init__()
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
        V_inv = torch.FloatTensor(V).inverse()
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
        _, _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)
        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        att = dot.softmax(dim=-1)
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, ' b (x y) (n w1 w2) d -> b n x y w1 w2 d',
                      x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)
        z = self.proj(a)
        z = z.mean(1)
        if skip is not None:
            z = z + skip
        return z
        
class CrossViewSwapAttention(nn.Module):
    def __init__(
        self, feat_height: int, feat_width: int, feat_dim: int, dim: int, index: int,
        image_height: int, image_width: int, qkv_bias: bool, q_win_size: list,
        feat_win_size: list, heads: list, dim_head: list, bev_embedding_flag: list,
        rel_pos_emb: bool = False, no_image_features: bool = False, skip: bool = True,
        norm=nn.LayerNorm,
    ):
        super().__init__()
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)
        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
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

    def pad_divisble(self, x, win_h, win_w):
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h -1) // win_h) * win_h, ((w + win_w-1) // win_w) * win_w
        padh = h_pad - h
        padw = w_pad - w
        return F.pad(x, (0, padw, 0, padh), value=0)

    def forward(
        self, index: int, x: torch.FloatTensor, bev: BEVEmbedding, feature: torch.FloatTensor,
        I_inv: torch.FloatTensor, E_inv: torch.FloatTensor,
        object_count: Optional[torch.Tensor] = None,
    ):
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
        if index == 0: world = bev.grid0[:2]
        elif index == 1: world = bev.grid1[:2]
        elif index == 2: world = bev.grid2[:2]
        elif index == 3: world = bev.grid3[:2]
        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)
        else:
            key_flat = img_embed
        val_flat = self.feature_linear(feature_flat)
        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        query = rearrange(self.cross_win_attend_1(query, key, val,
                                                skip=rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                                                                w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None),
                          'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_1(self.prenorm_1(query))
        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        query = rearrange(self.cross_win_attend_2(query, key, val,
                                                skip=rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                                                                w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None),
                          'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        query = rearrange(query, 'b H W d -> b d H W')
        return query

class PyramidAxialEncoder(nn.Module):
    def __init__(self, backbone, cross_view: dict, cross_view_swap: dict, bev_embedding: dict,
                 self_attn: dict, dim: list, middle: List[int] = [2, 2], scale: float = 1.0):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x
        assert len(self.backbone.output_shapes) == len(middle)
        cross_views, layers, downsample_layers = list(), list(), list()
        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)
            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)
            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i] // 2, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(dim[i+1], dim[i+1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim[i+1]), nn.ReLU(inplace=True),
                    nn.Conv2d(dim[i+1], dim[i+1], 1, padding=0, bias=False),
                    nn.BatchNorm2d(dim[i+1])))
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
        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)
        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features) - 1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)
        return x

# =================================================================================
# BEVFormer 핵심 모듈들 (mmcv 의존성 수정됨)
# =================================================================================

USE_CUDA_EXT = False
ext_module = None
try:
    from mmcv.utils import ext_loader
    ext_module = ext_loader.load_ext(
        '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
    print("Successfully loaded CUDA extension for ms_deform_attn.")
    USE_CUDA_EXT = True
except ImportError:
    print("CUDA extension for ms_deform_attn not found. High-precision model will fall back to PyTorch implementation.")
    print("Note: For fallback, 'mmcv' is required. It will be imported on-demand.")


class MultiScaleDeformableAttnFunction_fp32(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        if USE_CUDA_EXT and ext_module is not None:
            output = ext_module.ms_deform_attn_forward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations,
                attention_weights, im2col_step=ctx.im2col_step)
        else:
            try:
                from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
            except ImportError:
                raise ImportError("PyTorch fallback for Deformable Attention requires 'mmcv'. Please install it using 'pip install -U openmim; mim install mmcv-full'.")
            
            output = multi_scale_deformable_attn_pytorch(
                value, value_spatial_shapes, sampling_locations, attention_weights)

        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not USE_CUDA_EXT or ext_module is None:
            raise NotImplementedError("Backward pass for PyTorch version of ms_deform_attn is not implemented.")
            
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = torch.zeros_like(value), torch.zeros_like(sampling_locations), torch.zeros_like(attention_weights)
        ext_module.ms_deform_attn_backward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights,
            grad_output.contiguous(), grad_value,
            grad_sampling_loc, grad_attn_weight,
            im2col_step=ctx.im2col_step)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None

# --- 2번 코드: Spatial Cross Attention ---
class MSDeformableAttention3D(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=8,
                 im2col_step=64, batch_first=True, **kwargs):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}')
        
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.batch_first = batch_first

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, key=None, value=None, identity=None, query_pos=None,
                key_padding_mask=None, reference_points=None, spatial_shapes=None,
                level_start_index=None, **kwargs):
        if value is None: value = query
        if identity is None: identity = query
        if query_pos is not None: query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.')

        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, self.im2col_step)

        output = self.output_proj(output)
        
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return identity + output


class SpatialCrossAttention(nn.Module):
    def __init__(self, embed_dims=256, num_cams=6, dropout=0.1,
                 deformable_attention=dict(type='MSDeformableAttention3D', embed_dims=256, num_levels=4),
                 **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformableAttention3D(**deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, key, value, residual=None, query_pos=None,
                key_padding_mask=None, reference_points=None, spatial_shapes=None,
                reference_points_cam=None, bev_mask=None, level_start_index=None, **kwargs):
        if residual is None: residual = query
        if query_pos is not None: query = query + query_pos
        
        bs, num_query, _ = query.shape
        
        # NOTE: bev_mask processing is simplified for this example
        # In a real scenario, this part is crucial for efficiency
        
        # B, N, sum(H*W), C -> N, sum(H*W), B, C
        key = key.permute(1, 2, 0, 3) 
        value = value.permute(1, 2, 0, 3)

        num_cams, l, bs, embed_dims = key.shape
        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        
        # For simplicity, we assume all bev queries attend to all cams.
        # This is inefficient but demonstrates the concept.
        query_rebatch = query.repeat_interleave(self.num_cams, dim=0)
        reference_points_rebatch = reference_points_cam.reshape(bs * self.num_cams, num_query, -1, 2)
        
        queries = self.deformable_attention(
            query=query_rebatch, key=key, value=value,
            reference_points=reference_points_rebatch, spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        
        queries = queries.view(bs, self.num_cams, num_query, self.embed_dims)
        queries = queries.mean(dim=1) # Aggregate from all cameras
        
        slots = self.output_proj(queries)
        return self.dropout(slots) + residual

# --- 3번 코드: Temporal Self Attention ---
class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=1, num_points=4,
                 num_bev_queue=2, im2col_step=64, dropout=0.1, batch_first=True, **kwargs):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}')
        
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        
        self.sampling_offsets = nn.Linear(embed_dims * self.num_bev_queue, num_bev_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims * self.num_bev_queue, num_bev_queue * num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, key=None, value=None, identity=None, query_pos=None,
                reference_points=None, spatial_shapes=None, level_start_index=None, **kwargs):
        if identity is None: identity = query
        if query_pos is not None: query = query + query_pos
        
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            
        bs, num_query, embed_dims = query.shape
        
        # value is [prev_bev, current_bev]
        # value has shape (bs * num_bev_queue, num_query, embed_dims)
        value_reshaped = value.view(bs, self.num_bev_queue, num_query, embed_dims)
        
        query_enhanced = torch.cat([value_reshaped[:, 0, ...], query], dim=-1) # (bs, num_query, embed_dims*2)
        value_proj = self.value_proj(value)
        value_proj = value_proj.reshape(bs * self.num_bev_queue, num_query, self.num_heads, -1)
        
        sampling_offsets = self.sampling_offsets(query_enhanced).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query_enhanced).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points).softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(
            bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        if reference_points.shape[-1] == 2:
            reference_points_reshaped = reference_points.unsqueeze(2).repeat(1, 1, self.num_bev_queue, 1, 1)
            reference_points_reshaped = reference_points_reshaped.view(bs, num_query, self.num_bev_queue * self.num_levels, 2)
            
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points_reshaped[:, :, None, :, None, :] \
                                 + sampling_offsets.view(bs, self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2).permute(0,2,3,1,4,5,6).reshape(bs, num_query, self.num_heads, self.num_bev_queue*self.num_levels, self.num_points, 2) / offset_normalizer[None, None, None, :, None, :]

        else:
            raise ValueError("Reference points for temporal attention should be 2D")
        
        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value_proj, spatial_shapes, level_start_index.repeat(self.num_bev_queue), sampling_locations.reshape(bs, num_query, self.num_heads, -1, 2).repeat(self.num_bev_queue, 1, 1, 1, 1),
            attention_weights, self.im2col_step)
            
        output = output.view(bs, self.num_bev_queue, num_query, embed_dims)
        output = output.mean(1)
        
        output = self.output_proj(output)
        
        if not self.batch_first:
            output = output.permute(1, 0, 2)
            
        return self.dropout(output) + identity

# --- 4번 코드: Perception Transformer (Orchestrator) ---
class BEVFormerLayer(nn.Module):
    def __init__(self, temporal_attn_cfg, spatial_attn_cfg, ffn_cfg):
        super().__init__()
        self.temporal_attn = TemporalSelfAttention(**temporal_attn_cfg)
        self.spatial_attn = SpatialCrossAttention(**spatial_attn_cfg)
        self.ffn = nn.Sequential(
            nn.Linear(ffn_cfg['embed_dims'], ffn_cfg['feedforward_channels']),
            nn.GELU(),
            nn.Dropout(ffn_cfg['dropout']),
            nn.Linear(ffn_cfg['feedforward_channels'], ffn_cfg['embed_dims'])
        )
        self.norm1 = nn.LayerNorm(ffn_cfg['embed_dims'])
        self.norm2 = nn.LayerNorm(ffn_cfg['embed_dims'])
        self.norm3 = nn.LayerNorm(ffn_cfg['embed_dims'])
        
    def forward(self, query, key, value, bev_pos=None, prev_bev=None, **kwargs):
        bev_queue = torch.cat([prev_bev, query], dim=1)
        
        query = self.temporal_attn(query, value=bev_queue.permute(1,0,2), identity=query, query_pos=bev_pos, **kwargs)
        query = self.norm1(query)
        
        query = self.spatial_attn(query, key, value, residual=query, query_pos=bev_pos, **kwargs)
        query = self.norm2(query)
        
        query = self.ffn(self.norm3(query)) + query
        return query

class BEVFormerEncoder(nn.Module):
    def __init__(self, num_layers, layer_cfg):
        super().__init__()
        self.layers = nn.ModuleList([BEVFormerLayer(**layer_cfg) for _ in range(num_layers)])

    def forward(self, bev_queries, feat_flatten, feat_flatten_value, prev_bev, **kwargs):
        bev_embed = bev_queries
        for i, layer in enumerate(self.layers):
            current_prev_bev = prev_bev if i == 0 else bev_embed
            bev_embed = layer(
                query=bev_embed, key=feat_flatten, value=feat_flatten_value,
                prev_bev=current_prev_bev, **kwargs)
        return bev_embed

class PerceptionTransformer(nn.Module):
    def __init__(self, embed_dims=256, num_feature_levels=4, num_cams=6, encoder=None, **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        
        self.encoder = BEVFormerEncoder(**encoder)
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)

    def forward(self, mlvl_feats, bev_queries, bev_h, bev_w, bev_pos=None, prev_bev=None, **kwargs):
        bs = mlvl_feats[0].size(0)
        
        if prev_bev is None:
            prev_bev = torch.zeros_like(bev_queries)
        
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(0, 1, 3, 2)
            feat = feat + self.cams_embeds[None, :, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        bev_embed = self.encoder(
            bev_queries, feat_flatten, feat_flatten, prev_bev=prev_bev,
            bev_h=bev_h, bev_w=bev_w, bev_pos=bev_pos,
            spatial_shapes=spatial_shapes, level_start_index=level_start_index, **kwargs)
        return bev_embed

# =================================================================================
# 최종 하이브리드 모델
# =================================================================================
class HybridCameraBEVModel(nn.Module):
    def __init__(
        self, backbone, light_model_cfg: dict, heavy_model_cfg: dict,
        bev_h, bev_w, embed_dims
    ):
        super().__init__()
        self.bev_h, self.bev_w, self.embed_dims = bev_h, bev_w, embed_dims
        
        # backbone을 light_model과 heavy_model 양쪽에서 공유하도록 수정
        self.backbone = backbone
        self.light_model = PyramidAxialEncoder(backbone=self.backbone, **light_model_cfg)
        self.heavy_model = PerceptionTransformer(embed_dims=embed_dims, **heavy_model_cfg)
        
        self.bev_queries = nn.Parameter(torch.randn(bev_h * bev_w, embed_dims))
        self.bev_pos = nn.Parameter(torch.randn(1, embed_dims, bev_h, bev_w))

        self.prev_bev = None
        self.object_count_threshold = 30
        
    def forward(self, batch):
        object_count = batch.get('object_count', None)
        
        use_heavy_model = False
        if object_count is not None:
            total_objects = torch.sum(object_count)
            print(f"Total objects in batch: {total_objects}. Threshold is {self.object_count_threshold}.")
            if total_objects >= self.object_count_threshold:
                use_heavy_model = True
        
        if use_heavy_model:
            print("Object count is high. Switching to HEAVY model (BEVFormer-style).")
            # heavy_model은 backbone을 직접 호출하지 않고, 외부에서 받은 피처를 사용
            image = batch['image'].flatten(0, 1)
            # light_model의 norm과 down을 잠시 빌려옴 (구조 통일)
            mlvl_feats = [self.light_model.down(y) for y in self.backbone(self.light_model.norm(image))]
            bs, n_cams = batch['image'].shape[:2]
            mlvl_feats = [feat.view(bs, n_cams, *feat.shape[1:]) for feat in mlvl_feats]
            
            bev_embed = self.heavy_model(
                mlvl_feats=mlvl_feats,
                bev_queries=self.bev_queries.unsqueeze(0).repeat(bs, 1, 1),
                bev_h=self.bev_h, bev_w=self.bev_w,
                bev_pos=self.bev_pos.flatten(2).permute(0, 2, 1).repeat(bs, 1, 1),
                prev_bev=self.prev_bev,
                **batch  # intrinsics, extrinsics 등을 전달
            )
            
            self.prev_bev = bev_embed.detach()
            bev_embed = bev_embed.permute(0, 2, 1).view(bs, self.embed_dims, self.bev_h, self.bev_w)
            return bev_embed
            
        else:
            print("Object count is low. Using LIGHT model (CrossViewSwapAttention).")
            self.prev_bev = None
            return self.light_model(batch)

if __name__ == '__main__':
    # ... (이전 답변의 테스트 코드)
    pass
         
