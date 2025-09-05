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
# 1, 2, 3, 4번 코드에서 가져온 BEVFormer 핵심 모듈들
# =================================================================================

# --- 1번 코드: 저수준 CUDA 커널 인터페이스 (mmcv 의존성 수정) ---
# mmcv 관련 import를 이 블록 안에서만 처리하도록 수정
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
            # mmcv를 이 시점에서 import 시도
            try:
                from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
            except ImportError:
                raise ImportError("PyTorch fallback for Deformable Attention requires 'mmcv'. Please install it using 'pip install mmcv-full'.")
            
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
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

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

# ... (기존의 Normalize, BEVEmbedding, CrossViewSwapAttention, PyramidAxialEncoder 클래스는 여기에 위치)
# ... (생략된 코드는 이전 답변을 참고하여 채워주세요)
class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std

class CrossWinAttention(nn.Module):
    # ... (기존 CrossWinAttention 코드)
    pass
    
class CrossViewSwapAttention(nn.Module):
    # ... (기존 CrossViewSwapAttention 코드)
    # forward 메소드에 object_count 인자 추가
    def forward(
        self,
        index: int,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        object_count: Optional[torch.Tensor] = None, #object_count
    ):
        # ... (기존 forward 로직)
        pass

class PyramidAxialEncoder(nn.Module):
    # ... (기존 PyramidAxialEncoder 코드)
    # forward 메소드에서 cross_view 호출 시 object_count 전달
    def forward(self, batch):
        # ... (기존 forward 로직 상단)
        object_count = batch.get('object_count', None)
        # ...
        # for 루프 안:
        # x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
        # ...
        pass
# =================================================================================
# 최종 하이브리드 모델
# =================================================================================
class HybridCameraBEVModel(nn.Module):
    def __init__(
        self,
        backbone,
        light_model_cfg: dict,
        heavy_model_cfg: dict,
        bev_h, bev_w, embed_dims
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        
        self.backbone = backbone
        self.norm = Normalize()
        self.down = lambda x: x
        
        self.light_model = PyramidAxialEncoder(backbone=backbone, **light_model_cfg)
        self.heavy_model = PerceptionTransformer(embed_dims=embed_dims, **heavy_model_cfg)
        
        self.bev_queries = nn.Parameter(torch.randn(bev_h * bev_w, embed_dims))
        self.bev_pos = self.create_bev_pos(bev_h, bev_w, embed_dims)

        self.prev_bev = None
        self.object_count_threshold = 30
        
    def create_bev_pos(self, h, w, dim):
        pos = torch.randn(1, dim, h, w)
        return nn.Parameter(pos)

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
            image = batch['image'].flatten(0, 1)
            mlvl_feats = [self.down(y) for y in self.backbone(self.norm(image))]
            bs, n_cams = batch['image'].shape[:2]
            mlvl_feats = [feat.view(bs, n_cams, *feat.shape[1:]) for feat in mlvl_feats]
            
            bev_embed = self.heavy_model(
                mlvl_feats=mlvl_feats,
                bev_queries=self.bev_queries.unsqueeze(0).repeat(bs, 1, 1),
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                bev_pos=self.bev_pos.flatten(2).permute(0,2,1).repeat(bs,1,1),
                prev_bev=self.prev_bev,
                intrinsics=batch['intrinsics'],
                extrinsics=batch['extrinsics']
            )
            
            self.prev_bev = bev_embed.detach()
            bev_embed = bev_embed.permute(0, 2, 1).view(bs, self.embed_dims, self.bev_h, self.bev_w)
            
            return bev_embed
            
        else:
            print("Object count is low. Using LIGHT model (CrossViewSwapAttention).")
            self.prev_bev = None
            return self.light_model(batch)
