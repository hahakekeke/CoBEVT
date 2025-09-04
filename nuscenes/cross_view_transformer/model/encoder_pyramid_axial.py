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

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)     # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                  # 3 h w
    indices = indices[None]                                               # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
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


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
            self,
            dim: int,
            sigma: int,
            bev_height: int,
            bev_width: int,
            h_meters: int,
            w_meters: int,
            offset: int,
            upsample_scales: list,
    ):
        super().__init__()
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters,
                            offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()  # 3 3

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
            sigma * torch.randn(dim,
                                bev_height//upsample_scales[0],
                                bev_width//upsample_scales[0]))

    def get_prior(self):
        return self.learned_features

# ==========================================================================================
# BEVFormer v2 Components
# ==========================================================================================
class TemporalEncoder(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, current_bev, history_bev):
        b, d, h, w = current_bev.shape
        current_bev_flat = current_bev.view(b, d, h * w).permute(0, 2, 1) # (b, h*w, d)
        
        if not history_bev:
            history_bev_cat = current_bev_flat
        else:
            history_bev_flat = [hist.view(b, d, h * w).permute(0, 2, 1) for hist in history_bev]
            history_bev_cat = torch.cat(history_bev_flat + [current_bev_flat], dim=1)

        attn_out, _ = self.attn(current_bev_flat, history_bev_cat, history_bev_cat)
        out = self.norm1(current_bev_flat + attn_out)
        out = self.norm2(out + self.ffn(out))
        
        return out.permute(0, 2, 1).view(b, d, h, w)

class PerspectiveHead(nn.Module):
    def __init__(self, in_dim, feature_dim, num_proposals=100):
        super().__init__()
        self.num_proposals = num_proposals
        self.proposal_generator = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proposal_embedding = nn.Linear(feature_dim, feature_dim)

    def forward(self, image_features):
        proposals = self.proposal_generator(image_features)
        proposals = proposals.view(proposals.size(0), -1)
        proposals = proposals[:self.num_proposals] 
        proposals = self.proposal_embedding(proposals)
        return proposals.unsqueeze(0)

class DETRDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, query, bev_memory):
        q2 = self.self_attn(query, query, query)[0]
        query = self.norm1(query + self.dropout1(q2))
        
        q2 = self.cross_attn(query, bev_memory, bev_memory)[0]
        query = self.norm2(query + self.dropout2(q2))
        
        q2 = self.ffn(query)
        query = self.norm3(query + self.dropout3(q2))
        
        return query
# ==========================================================================================

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 25
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, _, height, width, device, h = *x.shape, x.device, self.heads
        x = rearrange(x, 'b d h w -> b (h w) d')
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b m (h w) d -> b h w (m d)',
                        h = height, w = width)
        out = self.to_out(out)
        return rearrange(out, 'b h w d -> b d h w')


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
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        index: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        q_win_size: list,
        feat_win_size: list,
        heads: list,
        dim_head: list,
        bev_embedding_flag: list,
        rel_pos_emb: bool = False,
        no_image_features: bool = False,
        skip: bool = True,
        norm=nn.LayerNorm,
    ):
        super().__init__()
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)
        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))
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
        self,
        index: int,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
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
        
        skip_conn = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                                w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query = self.cross_win_attend_1(query, key, val, skip=skip_conn)
        query = rearrange(query, 'b x y w1 w2 d  -> b (x w1) (y w2) d')

        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)

        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        
        key = rearrange(key, 'b n x y w1 w2 d -> b n w1 w2 x y d')
        val = rearrange(val, 'b n x y w1 w2 d -> b n w1 w2 x y d')
        
        skip_conn_2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                                w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query = self.cross_win_attend_2(query, key, val, skip=skip_conn_2)
        query = rearrange(query, 'b x y w1 w2 d  -> b (x w1) (y w2) d')

        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        query = rearrange(query, 'b H W d -> b d H W')
        return query

class PyramidAxialEncoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            cross_view_swap: dict,
            bev_embedding: dict,
            self_attn: dict,
            dim: list,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
            num_bevformer_layers: int = 6,
            num_history: int = 2,
    ):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        self.down = (lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)) if scale < 1.0 else (lambda x: x)
        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()
        downsample_layers = list()
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
                # --- 🚨 최종 에러 수정 부분 🚨 ---
                # 원인: 기존 downsample_layers의 채널 계산 로직이 완전히 잘못되어 있었음.
                #       PixelUnshuffle은 채널을 4배로 늘리는데, 이를 전혀 반영하지 않아 차원 불일치 발생.
                #       이것이 결국 BEV 피처맵 크기를 잘못 계산하게 하여 최종 `AssertionError`를 유발.
                # 해결: 버그가 있는 복잡한 블록을 표준적인 Strided Convolution 기반의 다운샘플링 블록으로 교체.
                #       이는 공간적 차원을 1/2로 줄이고 채널을 dim[i] -> dim[i+1]로 정확하게 변경함.
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dim[i+1]),
                    nn.ReLU(inplace=True)
                ))
                # --- 여기까지 ---

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        
        self.temporal_encoder = TemporalEncoder(dim=dim[-1])
        self.perspective_head = PerspectiveHead(in_dim=self.backbone_output_dims[-1], feature_dim=dim[-1])
        self.detr_decoder = nn.ModuleList([
            DETRDecoderLayer(dim=dim[-1]) for _ in range(num_bevformer_layers)
        ])
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

        use_bevformer_path = False
        if object_count is not None and torch.any(object_count >= 30):
            use_bevformer_path = True
        
        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature_unflatten = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature_unflatten, I_inv, E_inv)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

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
    import os
    import re
    import yaml
    def load_yaml(file):
        with open(file, 'r') as stream:
            loader = yaml.SafeLoader
            # The following resolver is now part of SafeLoader by default in recent PyYAML versions
            # but is kept for compatibility.
            loader.add_implicit_resolver(
                u'tag:yaml.org,2002:float',
                re.compile(u'''^(?:
                [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
                list(u'-+0123456789.'))
            param = yaml.load(stream, Loader=loader)
        if "yaml_parser" in param:
            param = eval(param["yaml_parser"])(param)
        return param

    # This test block cannot be fully run without a real backbone and config files.
    # The following is a conceptual test for CrossWinAttention.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running conceptual test on CUDA.")
        
        block = CrossWinAttention(dim=128,
                                    heads=4,
                                    dim_head=32,
                                    qkv_bias=True,).to(device)
        # A valid test case should have matching numbers of windows.
        # q: (b, n, 5, 5, 5, 5, d) -> 5x5=25 windows
        # k: (b, n, 5, 5, 6, 12, d) -> 5x5=25 windows
        test_q = torch.rand(1, 6, 5, 5, 5, 5, 128).to(device)
        test_k = torch.rand(1, 6, 5, 5, 6, 12, 128).to(device)
        test_v = torch.rand(1, 6, 5, 5, 6, 12, 128).to(device)

        output = block(test_q, test_k, test_v)
        print("CrossWinAttention test output shape:", output.shape)
    else:
        print("CUDA not available, skipping __main__ test.")
