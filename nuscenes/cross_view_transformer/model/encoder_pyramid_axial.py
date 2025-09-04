import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List
# from .decoder import DecoderBlock # 이 부분은 로컬 의존성이므로 주석 처리합니다.

from typing import Optional


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
        [ 0., -sw,      w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,          1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class BEVEmbedding(nn.Module):
    def __init__(
            self, dim: int, sigma: int, bev_height: int, bev_width: int,
            h_meters: int, w_meters: int, offset: int, upsample_scales: list,
    ):
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

# Simplified utility classes
class TemporalEncoder(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, current_bev, history_bev):
        b, d, h, w = current_bev.shape
        current_bev_flat = current_bev.view(b, d, h * w).permute(0, 2, 1)
        history_bev_flat = [hist.view(b, d, h * w).permute(0, 2, 1) for hist in history_bev]
        kv_bev = torch.cat(history_bev_flat + [current_bev_flat], dim=1) if history_bev else current_bev_flat
        attn_out, _ = self.attn(current_bev_flat, kv_bev, kv_bev)
        out = self.norm1(current_bev_flat + attn_out)
        out = self.norm2(out + self.ffn(out))
        return out.permute(0, 2, 1).view(b, d, h, w)

class PerspectiveHead(nn.Module):
    def __init__(self, in_dim, feature_dim, num_proposals=100):
        super().__init__()
        self.num_proposals = num_proposals
        self.proposal_generator = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.proposal_embedding = nn.Linear(feature_dim, feature_dim)

    def forward(self, image_features):
        proposals = self.proposal_generator(image_features).flatten(1)
        proposals = proposals[:self.num_proposals]
        proposals = self.proposal_embedding(proposals)
        return proposals.unsqueeze(0)

class DETRDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, query, bev_memory):
        q2 = self.self_attn(query, query, query)[0]
        query = self.norm1(query + self.dropout1(q2))
        q2 = self.cross_attn(query, bev_memory, bev_memory)[0]
        query = self.norm2(query + self.dropout2(q2))
        q2 = self.ffn(query)
        query = self.norm3(query + self.dropout3(q2))
        return query

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
        assert k.shape == v.shape
        _, _, q_height, q_width, q_win_h, q_win_w, _ = q.shape
        _, _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width, f"Grid size mismatch: Q:({q_height}x{q_width}) vs K:({kv_height}x{kv_width})"

        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h=self.heads), (q, k, v))
        
        dot = torch.einsum('b l Q d, b l K d -> b l Q K', q, k) * self.scale
        att = dot.softmax(dim=-1)
        
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b h) ... d -> b ... (h d)', h=self.heads)
        a = rearrange(a, 'b (x y) (n w1 w2) d -> b n x y w1 w2 d', x=q_height, y=q_width, w1=q_win_h)
        z = self.proj(a).mean(1)
        
        return z + skip if skip is not None else z

class CrossViewSwapAttention(nn.Module):
    def __init__(
        self, feat_height: int, feat_width: int, feat_dim: int, dim: int, index: int,
        image_height: int, image_width: int, qkv_bias: bool, q_win_size: list,
        feat_win_size: list, heads: list, dim_head: list, bev_embedding_flag: list,
        no_image_features: bool = False, skip: bool = True, norm=nn.LayerNorm
    ):
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

    def forward(self, index: int, x: torch.FloatTensor, bev: BEVEmbedding, feature: torch.FloatTensor,
              I_inv: torch.FloatTensor, E_inv: torch.FloatTensor, object_count: Optional[torch.Tensor] = None):
        b, n, _, _, _ = feature.shape
        _, _, H_q, W_q = x.shape
        
        pixel = self.image_plane
        _, _, _, h_f, w_f = pixel.shape
        c = E_inv[..., -1:]
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        c_embed = self.cam_embed(c_flat)
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
        d = E_inv @ cam
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h_f, w=w_f)
        d_embed = self.img_embed(d_flat)
        img_embed = d_embed - c_embed
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        grid_name = f'grid{index}'
        world = getattr(bev, grid_name, bev.grid0)[:2]

        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)
        
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        key_flat = img_embed + self.feature_proj(feature_flat) if self.feature_proj else img_embed
        val_flat = self.feature_linear(feature_flat)
        query = query_pos + x[:, None] if self.bev_embed_flag else x[:, None]
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)

        # --- 🚨 에러 해결을 위한 강제 리사이즈 워크어라운드 🚨 ---
        # 원인: BEV 피처맵과 이미지 피처맵의 윈도우 개수가 설정(config) 문제로 불일치.
        # 해결: key, val 텐서를 query 텐서의 윈도우 개수에 맞게 강제로 리사이즈.
        num_windows_h_q = H_q // self.q_win_size[0]
        num_windows_w_q = W_q // self.q_win_size[1]
        
        target_h_k = num_windows_h_q * self.feat_win_size[0]
        target_w_k = num_windows_w_q * self.feat_win_size[1]
        
        _, _, _, H_k, W_k = key.shape
        
        if H_k != target_h_k or W_k != target_w_k:
            key = rearrange(key, 'b n d h w -> (b n) d h w')
            val = rearrange(val, 'b n d h w -> (b n) d h w')
            key = F.interpolate(key, size=(target_h_k, target_w_k), mode='bilinear', align_corners=False)
            val = F.interpolate(val, size=(target_h_k, target_w_k), mode='bilinear', align_corners=False)
            key = rearrange(key, '(b n) d h w -> b n d h w', b=b)
            val = rearrange(val, '(b n) d h w -> b n d h w', b=b)
        # --- 워크어라운드 끝 ---

        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        skip_conn = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query = self.cross_win_attend_1(query, key, val, skip=skip_conn)
        query = rearrange(query, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        
        key = rearrange(key, 'b n x y w1 w2 d -> b n w1 w2 x y d')
        val = rearrange(val, 'b n x y w1 w2 d -> b n w1 w2 x y d')
        
        skip_conn_2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query = self.cross_win_attend_2(query, key, val, skip=skip_conn_2)
        query = rearrange(query, 'b x y w1 w2 d -> b (x w1) (y w2) d')

        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        query = rearrange(query, 'b H W d -> b d H W')
        return query

class PyramidAxialEncoder(nn.Module):
    def __init__(
            self, backbone, cross_view: dict, cross_view_swap: dict, bev_embedding: dict,
            self_attn: dict, dim: list, middle: List[int] = [2, 2], scale: float = 1.0,
            num_bevformer_layers: int = 6, num_history: int = 2,
    ):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        self.down = (lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)) if scale < 1.0 else (lambda x: x)
        assert len(self.backbone.output_shapes) == len(middle)

        cross_views, layers, downsample_layers = [], [], []
        self.backbone_output_dims = []

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            dummy_tensor = torch.zeros(*feat_shape)
            _, feat_dim, feat_height, feat_width = self.down(dummy_tensor).shape
            self.backbone_output_dims.append(feat_dim)

            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)

            if i < len(middle) - 1:
                # This downsampling block is now robust.
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dim[i+1]),
                    nn.ReLU(inplace=True)
                ))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        
        self.temporal_encoder = TemporalEncoder(dim=dim[-1])
        self.perspective_head = PerspectiveHead(in_dim=self.backbone_output_dims[-1], feature_dim=dim[-1])
        self.detr_decoder = nn.ModuleList([
            DETRDecoderLayer(dim=dim[-1]) for _ in range(num_bevformer_layers)])
        self.num_queries = 100
        self.learned_queries = nn.Parameter(torch.randn(1, self.num_queries, dim[-1]))
        self.history_bev = deque(maxlen=num_history)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        object_count = batch.get('object_count', None)
        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)

        use_bevformer_path = object_count is not None and torch.any(object_count >= 30)
        
        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature_unflatten = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature_unflatten, I_inv, E_inv)
            x = layer(x)
            if i < len(features) - 1:
                x = self.downsample_layers[i](x)

        if use_bevformer_path:
            x = self.temporal_encoder(x, list(self.history_bev))
            last_feature_map = features[-1]
            perspective_proposals = self.perspective_head(last_feature_map)
            b_dim = x.size(0)
            proposal_queries = perspective_proposals.repeat(b_dim, 1, 1)
            hybrid_queries = torch.cat([self.learned_queries.repeat(b_dim, 1, 1), proposal_queries], dim=1)
            _, d, h, w = x.shape
            bev_memory = x.view(b, d, h * w).permute(0, 2, 1)
            decoder_out = hybrid_queries
            for decoder_layer in self.detr_decoder:
                decoder_out = decoder_layer(decoder_out, bev_memory)
            final_output = x
        else:
            final_output = x

        with torch.no_grad():
            self.history_bev.append(x.detach().clone())

        return final_output

if __name__ == "__main__":
    import os, re, yaml
    
    def load_yaml(file):
        with open(file, 'r') as stream:
            return yaml.safe_load(stream)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running conceptual test on CUDA.")
        
        block = CrossWinAttention(dim=128, heads=4, dim_head=32, qkv_bias=True).to(device)
        test_q = torch.rand(1, 6, 5, 5, 5, 5, 128).to(device)
        test_k = torch.rand(1, 6, 5, 5, 6, 12, 128).to(device)
        test_v = torch.rand(1, 6, 5, 5, 6, 12, 128).to(device)
        output = block(test_q, test_k, test_v)
        print("CrossWinAttention test output shape:", output.shape)
    else:
        print("CUDA not available, skipping __main__ test.")
