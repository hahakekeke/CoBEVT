# encoder_pyramid_axial.py
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
    indices = indices[None]  # 1 3 h w
    return indices

def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """ copied from ..data.common but want to keep models standalone """
    sh = h / h_meters
    sw = w / w_meters
    return [
        [ 0., -sw, w/2.],
        [-sh, 0., h*offset+h/2.],
        [ 0., 0., 1.]
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
        self.kwargs = { 'stride': stride, 'padding': padding, }
    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))

class BEVEmbedding(nn.Module):
    def __init__(
        self, dim: int, sigma: int, bev_height: int, bev_width: int, h_meters: int, w_meters: int, offset: int, upsample_scales: list,
    ):
        """
        Prior BEV embedding (learned) and stored projected grids for each upsample scale.
        """
        super().__init__()
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()  # 3 3
        for i, scale in enumerate(upsample_scales):
            h = bev_height // scale
            w = bev_width // scale
            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]
            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)
            self.register_buffer('grid%d'%i, grid, persistent=False)  # 3 h w

        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height//upsample_scales[0], bev_width//upsample_scales[0])
        )  # d h w

    def get_prior(self):
        return self.learned_features

class Attention(nn.Module):
    def __init__(
        self, dim, dim_head = 32, dropout = 0., window_size = 25
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

        # relative positional bias
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
        out = rearrange(out, 'b m (h w) d -> b h w (m d)', h = height, w = width)
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

    def forward(self, q, k, v, skip=None, attn_scale_factor: float = 1.0):
        """
        q: (b n X Y W1 W2 d)
        k: (b n x y w1 w2 d)
        v: (b n x y w1 w2 d)
        """
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

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)
        # apply external scaling factor (dynamic sharpening)
        if attn_scale_factor != 1.0:
            dot = dot * attn_scale_factor

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        att = dot.softmax(dim=-1)

        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, ' b (x y) (n w1 w2) d -> b n x y w1 w2 d', x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)

        z = self.proj(a)
        # reduce n dimension
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
            nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False)
        )
        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False)
            )

        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
            self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
            self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.q_win_size = list(q_win_size[index])
        self.feat_win_size = list(feat_win_size[index])
        self.rel_pos_emb = rel_pos_emb

        # fixed attention blocks (we keep module shapes but add dynamic behavior at runtime)
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)

        # global kv for heavy global-attention (image->bev)
        self.global_k = nn.Linear(dim, dim)
        self.global_v = nn.Linear(dim, dim)
        self.global_proj = nn.Linear(dim, dim)

        self.skip = skip

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

        # store base window sizes for reference
        self.base_q_win = tuple(self.q_win_size)
        self.base_feat_win = tuple(self.feat_win_size)

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def _compute_scene_complexity(self, object_count):
        """normalize and aggregate object_count to a scalar complexity measure"""
        if object_count is None:
            return 1.0  # minimal scale
        # Attempt typical shapes: (b, n_types) or (b,) or (b, n_cameras, ...) etc.
        try:
            # sum over types/cameras, then mean over batch
            if object_count.dim() == 1:
                compl = float(object_count.sum().item())
            else:
                compl = float(object_count.sum(dim=1).mean().item())
            # normalize to a reasonable scalar
            compl = max(0.0, compl)
            # smooth mapping: scale factor roughly 1 + compl/10
            return 1.0 + (compl / 10.0)
        except Exception:
            return 1.0

    def forward(
        self, index: int, x: torch.FloatTensor, bev: BEVEmbedding, feature: torch.FloatTensor,
        I_inv: torch.FloatTensor, E_inv: torch.FloatTensor, object_count: Optional[torch.Tensor] = None,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)
        object_count: optional tensor to indicate scene complexity
        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape
        _, _, H, W = x.shape

        # compute scene complexity scalar
        complexity_scale = self._compute_scene_complexity(object_count)
        # attn_scale_factor sharpens attention when complexity increases (>1)
        attn_scale_factor = 1.0 + (complexity_scale - 1.0) * 0.6

        # compute adaptive window sizes: more objects -> smaller q windows (more local detail)
        # factor ranges roughly 1..4
        factor = int(min(max(round(complexity_scale), 1), 4))
        q_win_h = max(1, self.base_q_win[0] // factor)
        q_win_w = max(1, self.base_q_win[1] // factor)
        feat_win_h = max(1, self.base_feat_win[0] // factor)
        feat_win_w = max(1, self.base_feat_win[1] // factor)

        # Prepare positional embeddings
        pixel = self.image_plane  # 1 1 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]  # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]  # (b n) 4 1
        c_embed = self.cam_embed(c_flat) if self.bev_embed_flag else 0.0

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
        d = E_inv @ cam
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)
        d_embed = self.img_embed(d_flat) if self.bev_embed_flag else 0.0
        img_embed = (d_embed - c_embed) if self.bev_embed_flag else 0.0
        if self.bev_embed_flag:
            img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        # bev embedding
        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        elif index == 3:
            world = bev.grid3[:2]
        else:
            # fallback: use grid0 if index out of expected range
            world = bev.grid0[:2]

        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])  # 1 d H W
            bev_embed = w_embed - c_embed  # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)
        else:
            query_pos = None

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

        # pad divisible using adaptive feat windows
        key = self.pad_divisble(key, feat_win_h, feat_win_w)
        val = self.pad_divisble(val, feat_win_h, feat_win_w)

        # ---- Local-to-local cross-attention (windowed), using adaptive q/feat windows ----
        # partition query into windows with adaptive q_win sizes
        query_windows = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=q_win_h, w2=q_win_w)
        key_windows = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=feat_win_h, w2=feat_win_w)
        val_windows = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=feat_win_h, w2=feat_win_w)

        # perform first cross-window attention with dynamic attn scaling
        skip_for_first = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=q_win_h, w2=q_win_w) if self.skip else None
        q_out = self.cross_win_attend_1(
            query_windows, key_windows, val_windows,
            skip=skip_for_first, attn_scale_factor=attn_scale_factor
        )
        # reverse windows to feature
        q_out = rearrange(q_out, 'b x y w1 w2 d -> b (x w1) (y w2) d')

        # MLP refinement 1 (repeat more when complex scenes)
        q_out = q_out + self.mlp_1(self.prenorm_1(q_out))
        x_skip = q_out

        # replicate for n cameras
        query_rep = repeat(q_out, 'b x y d -> b n x y d', n=n)

        # ---- Local-to-global cross-attention (heavy) ----
        # Build global K/V across cameras and spatial positions (flatten everything) -> heavy but powerful
        # Flatten key/value from (b n d h w) -> (b, L, d)
        global_k_flat = rearrange(key, 'b n d h w -> b (n h w) d')
        global_v_flat = rearrange(val, 'b n d h w -> b (n h w) d')

        # project global kv
        GK = self.global_k(global_k_flat)  # b L d
        GV = self.global_v(global_v_flat)  # b L d

        # prepare query tokens for global attention: flatten BEV query positions (b x*y d)
        q_for_global = rearrange(query_rep, 'b n (x w1) (y w2) d -> b (n x y) d')  # b Q d
        # compute scaled dot-product (heavy)
        qg = q_for_global.unsqueeze(2)  # b Q 1 d
        GK_t = GK.unsqueeze(1)  # b 1 L d
        attg = torch.matmul(qg, GK_t.transpose(-2, -1)).squeeze(2)  # b Q L
        # normalize and softmax
        attg = F.softmax(attg / max(1.0, (GK.shape[-1] ** 0.5)), dim=-1)
        # aggregate global values
        vg = torch.matmul(attg, GV)  # b Q d
        vg = self.global_proj(vg)  # b Q d
        # restore shape and add to query
        vg = rearrange(vg, 'b (n x y) d -> b n x y d', n=n, x=query_rep.shape[2], y=query_rep.shape[3])
        query_rep = query_rep + vg

        # ---- Second cross-window attention (local-to-global mixing) ----
        # repartition into windows used by second attention (use adaptive q_win)
        query_2 = rearrange(query_rep, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=q_win_h, w2=q_win_w)
        key_2 = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key_2 = rearrange(key_2, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=feat_win_h, w2=feat_win_w)
        val_2 = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val_2 = rearrange(val_2, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=feat_win_h, w2=feat_win_w)

        skip_for_second = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=q_win_h, w2=q_win_w) if self.skip else None
        q_out2 = self.cross_win_attend_2(query_2, key_2, val_2, skip=skip_for_second, attn_scale_factor=attn_scale_factor)
        q_out2 = rearrange(q_out2, 'b x y w1 w2 d -> b (x w1) (y w2) d')

        q_out2 = q_out2 + self.mlp_2(self.prenorm_2(q_out2))

        # Final normalization & reshape to (b, d, H, W)
        q_out2 = self.postnorm(q_out2)
        q_out2 = rearrange(q_out2, 'b H W d -> b d H W')

        # Additional iterative refinement: repeat MLP refinement when complexity high
        extra_refine = max(0, int(min(round(complexity_scale) - 1, 2)))
        for _ in range(extra_refine):
            flat = rearrange(q_out2, 'b d H W -> b (H W) d')
            flat = flat + self.mlp_2(self.prenorm_2(flat))
            q_out2 = rearrange(flat, 'b (H W) d -> b d H W', H=q_out2.shape[2], W=q_out2.shape[3])

        return q_out2

class PyramidAxialEncoder(nn.Module):
    def __init__(
        self, backbone, cross_view: dict, cross_view_swap: dict, bev_embedding: dict, self_attn: dict,
        dim: list, middle: List[int] = [2, 2], scale: float = 1.0,
    ):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        # allow increased depth: default middle increased to add capacity
        assert len(self.backbone.output_shapes) == len(middle)
        cross_views = list()
        layers = list()
        downsample_layers = list()
        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            # make cross view swap attention with potential heavier dims
            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)
            # increase number of bottlenecks per stage for higher capacity
            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(max(2, num_layers))])
            layers.append(layer)
            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(dim[i], dim[i] // 2, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.PixelUnshuffle(2),
                        nn.Conv2d(dim[i+1], dim[i+1], 3, padding=1, bias=False),
                        nn.BatchNorm2d(dim[i+1]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dim[i+1], dim[i+1], 1, padding=0, bias=False),
                        nn.BatchNorm2d(dim[i+1])
                    )
                ))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)

        # a heavy final self-attention to refine BEV (larger head/dim to increase capacity)
        final_dim = dim[-1]
        attn_heads = max(1, final_dim // 64)
        self.self_attn = Attention(final_dim, dim_head=64, dropout=0.1, window_size=33)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()

        # get object_count if present
        object_count = batch.get('object_count', None)

        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        # final self-attention refinement (heavy)
        x = self.self_attn(x)
        return x

if __name__ == "__main__":
    import os
    import re
    import yaml

    def load_yaml(file):
        stream = open(file, 'r')
        loader = yaml.Loader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?: [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)? |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+) |\\.[0-9_]+(?:[eE][-+][0-9]+)? |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]* |[-+]?\\.(?:inf|Inf|INF) |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.')
        )
        param = yaml.load(stream, Loader=loader)
        if "yaml_parser" in param:
            param = eval(param["yaml_parser"])(param)
        return param

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    block = CrossWinAttention(dim=128, heads=4, dim_head=32, qkv_bias=True,)
    block.cuda()
    test_q = torch.rand(1, 6, 5, 5, 5, 5, 128).cuda()
    test_k = test_v = torch.rand(1, 6, 5, 5, 6, 12, 128).cuda()
    output = block(test_q, test_k, test_v)
    print(output.shape)
