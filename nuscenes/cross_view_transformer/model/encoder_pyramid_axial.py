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

# =================================================================================
# 1, 2, 3, 4번 코드에서 가져온 BEVFormer 핵심 모듈들
# =================================================================================

# --- 1번 코드: 저수준 CUDA 커널 인터페이스 ---
from torch.autograd.function import Function, once_differentiable
# mmcv.utils.ext_loader는 실제 환경에 맞게 설치 및 설정이 필요합니다.
# 여기서는 개념적인 통합을 위해 PyTorch 순수 구현으로 대체 가능한 함수를 호출하도록 가정합니다.
# 만약 CUDA 확장 모듈이 없다면, 아래 multi_scale_deformable_attn_pytorch를 사용합니다.
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch

# 실제 CUDA 확장이 없을 경우를 대비한 Fallback 처리
try:
    from mmcv.utils import ext_loader
    ext_module = ext_loader.load_ext(
        '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
    USE_CUDA_EXT = True
except ImportError:
    print("CUDA extension for ms_deform_attn not found. Falling back to PyTorch implementation.")
    USE_CUDA_EXT = False


class MultiScaleDeformableAttnFunction_fp32(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        if USE_CUDA_EXT:
            output = ext_module.ms_deform_attn_forward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations,
                attention_weights, im2col_step=ctx.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, value_spatial_shapes, sampling_locations, attention_weights)

        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not USE_CUDA_EXT:
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
        # Initialization logic from the original code
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
        _, num_value, _ = value.shape
        
        # Here `value` is expected to be [prev_bev, current_bev] concatenated
        # The query is enhanced with history information
        query_enhanced = torch.cat([value[:, :num_query, :], query], -1)
        value = self.value_proj(value)
        value = value.reshape(bs * self.num_bev_queue, num_query, self.num_heads, -1)
        
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
        
        # reference_points are typically fixed for BEV self-attention
        # We need to reshape it for the function
        if reference_points.shape[-1] == 2:
            reference_points_reshaped = reference_points.repeat(self.num_bev_queue, 1, 1, 1)
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points_reshaped[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError("Reference points for temporal attention should be 2D")
        
        # Here we only have one level for BEV features
        spatial_shapes_temporal = spatial_shapes.repeat(self.num_bev_queue, 1)
        level_start_index_temporal = torch.cat([level_start_index, level_start_index + num_query])

        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value, spatial_shapes_temporal, level_start_index_temporal, sampling_locations,
            attention_weights, self.im2col_step)
            
        output = output.view(bs, self.num_bev_queue, num_query, embed_dims)
        output = output.mean(1) # Fuse history and current
        
        output = self.output_proj(output)
        
        if not self.batch_first:
            output = output.permute(1, 0, 2)
            
        return self.dropout(output) + identity

# --- 4번 코드: Perception Transformer (Orchestrator) ---
# Helper classes to build the transformer structure
class BEVFormerLayer(nn.Module):
    def __init__(self, temporal_attn_cfg, spatial_attn_cfg, ffn_cfg):
        super().__init__()
        self.temporal_attn = TemporalSelfAttention(**temporal_attn_cfg)
        self.spatial_attn = SpatialCrossAttention(**spatial_attn_cfg)
        self.ffn = nn.Sequential(
            nn.Linear(ffn_cfg['embed_dims'], ffn_cfg['feedforward_channels']),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_cfg['dropout']),
            nn.Linear(ffn_cfg['feedforward_channels'], ffn_cfg['embed_dims'])
        )
        self.norm1 = nn.LayerNorm(ffn_cfg['embed_dims'])
        self.norm2 = nn.LayerNorm(ffn_cfg['embed_dims'])
        self.norm3 = nn.LayerNorm(ffn_cfg['embed_dims'])
        self.dropout1 = nn.Dropout(ffn_cfg['dropout'])
        self.dropout2 = nn.Dropout(ffn_cfg['dropout'])
        self.dropout3 = nn.Dropout(ffn_cfg['dropout'])
        
    def forward(self, query, key, value, bev_pos=None, prev_bev=None, **kwargs):
        # Temporal Self Attention
        bev_queue = torch.cat([prev_bev, query], dim=0) # Simple concatenation for queue
        
        query = self.temporal_attn(query, value=bev_queue, identity=query, query_pos=bev_pos, **kwargs)
        query = self.norm1(query)
        
        # Spatial Cross Attention
        query = self.spatial_attn(query, key, value, residual=query, query_pos=bev_pos, **kwargs)
        query = self.norm2(query)
        
        # FFN
        query = query + self.dropout3(self.ffn(self.norm3(query)))
        
        return query

class BEVFormerEncoder(nn.Module):
    def __init__(self, num_layers, layer_cfg):
        super().__init__()
        self.layers = nn.ModuleList([BEVFormerLayer(**layer_cfg) for _ in range(num_layers)])

    def forward(self, bev_queries, feat_flatten, feat_flatten_value, prev_bev, **kwargs):
        bev_embed = bev_queries
        for i, layer in enumerate(self.layers):
            # For simplicity, pass the same prev_bev to all layers.
            # A more sophisticated implementation might update it progressively.
            current_prev_bev = prev_bev if i == 0 else bev_embed
            
            bev_embed = layer(
                query=bev_embed,
                key=feat_flatten,
                value=feat_flatten_value,
                prev_bev=current_prev_bev,
                **kwargs
            )
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
        # Decoder is omitted for simplicity, as we only need the BEV features
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)

    def forward(self, mlvl_feats, bev_queries, bev_h, bev_w, bev_pos=None, prev_bev=None, **kwargs):
        bs = mlvl_feats[0].size(0)
        
        # For simplicity, ego-motion compensation is assumed to be done on prev_bev before passing it here
        if prev_bev is None:
            prev_bev = torch.zeros_like(bev_queries)
        
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # N, B, H*W, C
            feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # N, B, sum(H*W), C
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        # Here, key and value are the same for cross-attention
        feat_flatten = feat_flatten.permute(1, 0, 2, 3) # B, N, sum(H*W), C
        
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            **kwargs
        )
        return bev_embed

# =================================================================================
# 기존 코드 (CrossView 기반 모델)
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

class CrossViewSwapAttention(nn.Module):
    # ... (기존 CrossViewSwapAttention 코드, 변경 없음)
    pass # 실제 구현에서는 이 부분을 기존 코드로 채워주세요.

class PyramidAxialEncoder(nn.Module):
    # ... (기존 PyramidAxialEncoder 코드, 변경 없음)
    pass # 실제 구현에서는 이 부분을 기존 코드로 채워주세요.

# =================================================================================
# 최종 하이브리드 모델
# =================================================================================

class HybridCameraBEVModel(nn.Module):
    def __init__(
        self,
        # 공통 파라미터
        backbone,
        
        # 경량 모델(PyramidAxialEncoder) 파라미터
        light_model_cfg: dict,
        
        # 고정밀 모델(PerceptionTransformer) 파라미터
        heavy_model_cfg: dict,
        
        # BEV 그리드 파라미터
        bev_h, bev_w, embed_dims
    ):
        super().__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        
        # 1. 공통 백본 (효율성을 위해 공유)
        self.backbone = backbone
        self.norm = nn.Identity() # Assuming input images are already normalized
        self.down = lambda x: x # No downsampling of backbone features by default

        # 2. 경량 모델 초기화
        self.light_model = PyramidAxialEncoder(backbone=backbone, **light_model_cfg)
        
        # 3. 고정밀 모델 초기화
        self.heavy_model = PerceptionTransformer(embed_dims=embed_dims, **heavy_model_cfg)
        
        # 4. 고정밀 모델에 필요한 BEV 쿼리 및 위치 인코딩 생성
        self.bev_queries = nn.Parameter(torch.randn(bev_h * bev_w, embed_dims))
        self.bev_pos = self.create_bev_pos(bev_h, bev_w, embed_dims) # Positional encoding

        # 5. 시간적 정보 저장을 위한 상태 변수
        self.prev_bev = None
        self.object_count_threshold = 30
        
    def create_bev_pos(self, h, w, dim):
        """BEV 그리드를 위한 학습 가능한 위치 인코딩 생성"""
        pos = torch.randn(1, dim, h, w)
        return nn.Parameter(pos)

    def forward(self, batch):
        object_count = batch.get('object_count', None)
        
        use_heavy_model = False
        if object_count is not None:
            # 배치 내 모든 객체 수의 합을 기준으로 결정
            total_objects = torch.sum(object_count)
            print(f"Total objects in batch: {total_objects}. Threshold is {self.object_count_threshold}.")
            if total_objects >= self.object_count_threshold:
                use_heavy_model = True
        
        if use_heavy_model:
            print("Object count is high. Switching to HEAVY model (BEVFormer-style).")
            # --- 고정밀 모델 경로 ---
            
            # 1. 이미지 특징 추출 (공유 백본 사용)
            image = batch['image'].flatten(0, 1)
            mlvl_feats = [self.down(y) for y in self.backbone(self.norm(image))]
            # Reshape features to (bs, num_cam, C, H, W)
            bs, n_cams = batch['image'].shape[:2]
            mlvl_feats = [feat.view(bs, n_cams, *feat.shape[1:]) for feat in mlvl_feats]

            # 2. Ego-motion 보정 (단순화된 예시)
            # 실제 구현에서는 CAN bus 데이터로 prev_bev를 warp 해야 합니다.
            # 여기서는 상태만 전달합니다.
            
            # 3. PerceptionTransformer 실행
            bev_embed = self.heavy_model(
                mlvl_feats=mlvl_feats,
                bev_queries=self.bev_queries.unsqueeze(0).repeat(bs, 1, 1),
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                bev_pos=self.bev_pos,
                prev_bev=self.prev_bev,
                # intrinsics/extrinsics for reference point projection
                intrinsics=batch['intrinsics'],
                extrinsics=batch['extrinsics']
            )
            
            # 4. 다음 스텝을 위해 prev_bev 상태 업데이트
            # detach()를 통해 gradient 흐름을 끊어줍니다.
            self.prev_bev = bev_embed.detach()
            
            # 최종 출력을 (bs, C, H, W) 형태로 변환
            bev_embed = bev_embed.permute(0, 2, 1).view(bs, self.embed_dims, self.bev_h, self.bev_w)
            
            return bev_embed
            
        else:
            print("Object count is low. Using LIGHT model (CrossViewSwapAttention).")
            # --- 경량 모델 경로 ---
            
            # 1. 시간적 연속성이 깨졌으므로 prev_bev 상태 초기화
            self.prev_bev = None
            
            # 2. 기존 PyramidAxialEncoder 실행
            # 이 모델은 내부적으로 백본을 다시 실행합니다.
            return self.light_model(batch)

if __name__ == '__main__':
    # 이 테스트 코드는 실제 데이터 및 전체 설정 파일 없이는 실행이 어렵습니다.
    # 모델의 구조와 조건부 로직을 확인하는 용도로 참고해주세요.

    # 가상 설정값
    bev_h, bev_w, embed_dims = 100, 100, 256
    
    # 가상 백본
    from torchvision.models import resnet18
    backbone = resnet18(pretrained=False)
    # BEVFormer는 보통 4개의 feature level을 사용합니다.
    # 이를 위해 Backbone을 수정하거나 FeaturePyramidNetwork를 사용해야 합니다.
    # 여기서는 간단히 하나의 feature map만 반환한다고 가정합니다.
    backbone.forward = lambda x: [backbone.layer4(backbone.layer3(backbone.layer2(backbone.layer1(backbone.relu(backbone.bn1(backbone.conv1(x)))))))]

    light_cfg = {
        # PyramidAxialEncoder에 필요한 설정값들...
        "cross_view": {}, "cross_view_swap": {}, "bev_embedding": {}, "self_attn": {}, "dim": [128], "middle": [2]
    }
    
    heavy_cfg = {
        # PerceptionTransformer에 필요한 설정값들...
        "num_feature_levels": 1,
        "num_cams": 6,
        "encoder": {
            "num_layers": 2,
            "layer_cfg": {
                "temporal_attn_cfg": {"embed_dims": embed_dims, "num_levels": 1},
                "spatial_attn_cfg": {"embed_dims": embed_dims},
                "ffn_cfg": {"embed_dims": embed_dims, "feedforward_channels": 512, "dropout": 0.1}
            }
        }
    }

    # 하이브리드 모델 생성
    hybrid_model = HybridCameraBEVModel(
        backbone=backbone,
        light_model_cfg=light_cfg,
        heavy_model_cfg=heavy_cfg,
        bev_h=bev_h, bev_w=bev_w, embed_dims=embed_dims
    )

    # 가상 입력 데이터 생성
    bs, n_cams, C, H, W = 2, 6, 3, 224, 480
    batch = {
        'image': torch.rand(bs, n_cams, C, H, W),
        'intrinsics': torch.rand(bs, n_cams, 3, 3),
        'extrinsics': torch.rand(bs, n_cams, 4, 4),
    }

    # 시나리오 1: 객체 수가 적을 때
    print("\n--- Scenario 1: Low object count ---")
    batch['object_count'] = torch.tensor([5, 10]) # 총 15개
    output_light = hybrid_model(batch)
    print(f"Output shape from light model: {output_light.shape}")

    # 시나리오 2: 객체 수가 많을 때
    print("\n--- Scenario 2: High object count ---")
    batch['object_count'] = torch.tensor([15, 20]) # 총 35개
    output_heavy = hybrid_model(batch)
    print(f"Output shape from heavy model: {output_heavy.shape}")
    
    # 시나리오 3: 고정밀 모델 연속 호출 (prev_bev 사용 확인)
    print("\n--- Scenario 3: High object count (consecutive call) ---")
    batch['object_count'] = torch.tensor([18, 22]) # 총 40개
    assert hybrid_model.prev_bev is not None, "prev_bev should be stored after a heavy model call"
    print("prev_bev is stored. Calling heavy model again...")
    output_heavy_2 = hybrid_model(batch)
    print(f"Output shape from second heavy call: {output_heavy_2.shape}")

    # 시나리오 4: 다시 경량 모델로 전환 (prev_bev 초기화 확인)
    print("\n--- Scenario 4: Switching back to low object count ---")
    batch['object_count'] = torch.tensor([3, 8]) # 총 11개
    output_light_2 = hybrid_model(batch)
    assert hybrid_model.prev_bev is None, "prev_bev should be cleared after a light model call"
    print("prev_bev is cleared as expected.")
    print(f"Output shape from second light call: {output_light_2.shape}")
