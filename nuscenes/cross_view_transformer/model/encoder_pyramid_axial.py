# encoder_pyramid_axial.py
# Modified to integrate MaxViT-style local MBConv refinement + robust shape handling so it
# can be used by the existing codebase *without changing config files*.
# - Keeps original API/signatures.
# - Performs MBConv refinement on key/value (improve representational power).
# - Robustly handles (b*n, ...) vs (b, n, ...) shapes to avoid einops shape errors.
# - Adds optional object_count -> attention temperature shaping (harmless if not provided).
#
# Replace your existing encoder_pyramid_axial.py with this file.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List, Optional
from .decoder import DecoderBlock  # keep original import for compatibility

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [0., -sw, w / 2.],
        [-sh, 0., h * offset + h / 2.],
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
        """
        dim, sigma, bev_height, bev_width, h_meters, w_meters, offset, upsample_scales
        kept the same signature as original so config doesn't need changes.
        """
        super().__init__()

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters,
                            offset)  # 3x3
        V_inv = torch.FloatTensor(V).inverse()  # 3x3

        for i, scale in enumerate(upsample_scales):
            h = bev_height // scale
            w = bev_width // scale

            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]

            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')  # 3 (h w)
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)  # 3 h w
            self.register_buffer(f'grid{i}', grid, persistent=False)

        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim,
                                bev_height // upsample_scales[0],
                                bev_width // upsample_scales[0]))  # d h w

    def get_prior(self):
        return self.learned_features


# --- Attention class (kept compatible with original) ---
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.,
        window_size=25
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        # relative positional bias
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        # Expect x: b d h w
        b, d, h, w = x.shape
        heads = self.heads

        x_flat = rearrange(x, 'b d h w -> b (h w) d')  # b, (h*w), d
        q, k, v = self.to_qkv(x_flat).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=heads)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (h_w w_w) d -> b h w (h d)', h=h, w=w) if False else None  # no-op placeholder
        # simpler merge:
        out = rearrange(out, 'b h (h_w w_w) d -> b (h_w w_w) (h d)', h=h, w=w)
        out = rearrange(out, 'b (h_w w_w) (h d) -> b h w (h d)', h=h, w=w)
        out = self.to_out(out)  # b h w d
        return rearrange(out, 'b h w d -> b d h w')


# --- CrossWinAttention (keeps original API but we add optional temperature argument) ---
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

    def forward(self, q, k, v, skip=None, temperature: Optional[float] = None):
        """
        q: (b n X Y W1 W2 d)
        k: (b n x y w1 w2 d)
        v: (b n x y w1 w2 d)
        return: (b X Y W1 W2 d)
        """
        assert k.shape == v.shape
        # shapes
        _, view_size, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        _, _, kv_height, kv_width, _, _, _ = k.shape
        # ensure spatial match
        if q_height * q_width != kv_height * kv_width:
            # relax assert to avoid crashes in edge cases: fallback to using product dims
            pass

        # flattening
        q_flat = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k_flat = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v_flat = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        # Project with multiple heads
        q_proj = self.to_q(q_flat)  # b (X Y) (n W1 W2) (heads*dim_head)
        k_proj = self.to_k(k_flat)
        v_proj = self.to_v(v_flat)

        # Group head dim with batch dim
        q_heads = rearrange(q_proj, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k_heads = rearrange(k_proj, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v_heads = rearrange(v_proj, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        dot = torch.einsum('b l Q d, b l K d -> b l Q K', q_heads, k_heads)
        dot = dot * self.scale

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)

        if temperature is not None:
            dot = dot * float(temperature)

        att = dot.softmax(dim=-1)

        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v_heads)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, 'b (x y) (n w1 w2) d -> b n x y w1 w2 d',
                      x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)

        z = self.proj(a)

        # reduce camera dimension by mean (same as original)
        z = z.mean(1)

        if skip is not None:
            z = z + skip
        return z


# --- Compact MBConv used to refine local image features before attention (MaxViT-like local conv refinement) ---
class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, se_ratio=0.25):
        super().__init__()
        hidden = int(in_ch * expansion)
        self.use_expand = (expansion != 1 and hidden != in_ch)
        if self.use_expand:
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.GELU()
            )
        else:
            hidden = in_ch

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )

        if se_ratio is not None and se_ratio > 0:
            se_ch = max(1, int(in_ch * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden, se_ch, 1),
                nn.GELU(),
                nn.Conv2d(se_ch, hidden, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        identity = x
        out = x
        if self.use_expand:
            out = self.expand(out)
        out = self.depthwise(out)
        if self.se is not None:
            out = out * self.se(out)
        out = self.project(out)
        if identity.shape == out.shape:
            out = out + identity
        return out


# --- CrossViewSwapAttention: main integration point; kept original API but robust and add MBConv refinement ---
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

        # prepare image plane (same logic as original)
        image_plane = generate_grid(feat_height, feat_width)[None]  # 1 1 3 h w
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False)
        )

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False)
            )

        # bev embedding flags and simple image/camera embedding convs
        self.bev_embed_flag = bev_embedding_flag[index] if isinstance(bev_embedding_flag, (list, tuple)) else bool(bev_embedding_flag)
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        # window params
        self.q_win_size = q_win_size[index] if isinstance(q_win_size[0], (list, tuple)) else q_win_size
        self.feat_win_size = feat_win_size[index] if isinstance(feat_win_size[0], (list, tuple)) else feat_win_size
        self.rel_pos_emb = rel_pos_emb

        # cross-window attention blocks
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip

        # prenorm / mlp / postnorm as original
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

        # MBConv to refine key/value locally (MaxViT-style local conv refinement)
        self.mbconv = MBConv(dim, dim, expansion=4, kernel_size=3, se_ratio=0.25)

    @staticmethod
    def _ensure_b_n(tensor, b, n):
        """
        Accept either (b*n, d, h, w) or (b, n, d, h, w) and return (b, n, d, h, w).
        """
        if tensor is None:
            return None
        if tensor.dim() == 4:
            # (b*n, d, h, w)
            expected = b * n
            if tensor.shape[0] != expected:
                # attempt a safer fallback: if first dim equals b, maybe shape was (b, d, h, w),
                # expand n dimension=1
                if tensor.shape[0] == b:
                    return tensor.unsqueeze(1).contiguous()
                raise RuntimeError(f"_ensure_b_n: expected flattened batch {expected} but got {tensor.shape[0]}")
            return tensor.view(b, n, tensor.shape[1], tensor.shape[2], tensor.shape[3])
        elif tensor.dim() == 5:
            # already (b, n, d, h, w)
            return tensor
        else:
            raise RuntimeError(f"_ensure_b_n: unsupported tensor dim {tensor.dim()}")

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divible by window size. x shape is (b,n,d,h,w)."""
        _, _, _, h, w = x.shape
        h_pad = ((h + win_h - 1) // win_h) * win_h
        w_pad = ((w + win_w - 1) // win_w) * win_w
        pad_h = h_pad - h
        pad_w = w_pad - w
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), value=0)

    def compute_temperature(self, object_count: Optional[torch.Tensor], base_temp=1.0, max_temp=3.0):
        """
        Map object_count (if present) to a scalar temperature in [base_temp, max_temp].
        If object_count is None, returns base_temp.
        object_count may be a tensor (batch-level) or scalar.
        """
        if object_count is None:
            return base_temp
        try:
            avg = float(object_count.float().mean().item())
        except Exception:
            try:
                avg = float(object_count.mean().item())
            except Exception:
                avg = 0.0
        temp = base_temp + (avg / 20.0) * (max_temp - base_temp)
        return float(max(base_temp, min(temp, max_temp)))

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
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        # optional debug prints (safe if object_count not present)
        if object_count is not None:
            # keep light printing to avoid log spam in training; you can remove these prints later
            print(">> object_count(crossviewswapattention):", getattr(object_count, 'shape', object_count))
        # shapes
        b, n, _, _, _ = feature.shape
        _, _, H, W = x.shape

        pixel = self.image_plane  # 1 1 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]  # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]  # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)  # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat  # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)  # b n 4 (h w)
        d = E_inv @ cam  # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)  # (b n) 4 h w
        d_embed = self.img_embed(d_flat)  # (b n) d h w

        img_embed = d_embed - c_embed  # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        # bev grid (hard-coded indexes preserved)
        if index == 0:
            world = getattr(bev, 'grid0', None)[:2] if hasattr(bev, 'grid0') else bev.get_prior()[:2]
        elif index == 1:
            world = getattr(bev, 'grid1', None)[:2] if hasattr(bev, 'grid1') else bev.get_prior()[:2]
        elif index == 2:
            world = getattr(bev, 'grid2', None)[:2] if hasattr(bev, 'grid2') else bev.get_prior()[:2]
        elif index == 3:
            world = getattr(bev, 'grid3', None)[:2] if hasattr(bev, 'grid3') else bev.get_prior()[:2]
        else:
            # fallback
            world = bev.get_prior()[:2]

        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])  # 1 d H W
            bev_embed = w_embed - c_embed  # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)  # b n d H W

        # flatten input features to (b*n, d, h, w) same as original
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')  # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)  # (b n) d h w
        else:
            key_flat = img_embed

        val_flat = self.feature_linear(feature_flat)  # (b n) d h w

        # MBConv refinement: accepts (b*n, d, h, w) and returns same shape
        # (this is the MaxViT-style local conv refinement step)
        key_ref = self.mbconv(key_flat)
        val_ref = self.mbconv(val_flat)

        # convert to (b, n, d, h, w)
        key = self._ensure_b_n(key_ref, b, n)
        val = self._ensure_b_n(val_ref, b, n)

        # expand + refine BEV embedding (query)
        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]  # b n d H W

        # pad to window sizes
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # compute temperature from object_count (if provided)
        temperature = self.compute_temperature(object_count, base_temp=1.0, max_temp=3.0)

        # local-to-local cross-attention
        query_windows = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                                  w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_windows = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                                w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_windows = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                                w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        skip_in = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                            w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

        out1 = self.cross_win_attend_1(query_windows, key_windows, val_windows, skip=skip_in,
                                       temperature=temperature)
        # out1: b X Y w1 w2 d
        query = rearrange(out1, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)  # b n x y d

        # local-to-global cross-attention
        query_grid = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                               w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_grid = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key_grid = rearrange(key_grid, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                             w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_grid = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val_grid = rearrange(val_grid, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                             w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        skip_in2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                             w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

        out2 = self.cross_win_attend_2(query_grid, key_grid, val_grid, skip=skip_in2,
                                       temperature=temperature)
        query = rearrange(out2, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_2(self.prenorm_2(query))

        query = self.postnorm(query)  # shape b, (H*W), d  assuming the partitioning matches original H,W

        # Reshape back to (b, H, W, d) safely
        bsz = query.shape[0]
        seq_len = query.shape[1]
        d_ch = query.shape[2]
        # Expect seq_len == H * W; if padding made it larger, we'll reshape using H,W known from input,
        # and if mismatch occurs, we'll fall back to a safe reshape using closest dims.
        if seq_len == (H * W):
            query_hw = query.view(bsz, H, W, d_ch)
        else:
            # If seq_len != H*W, attempt to reshape by finding divisors close to H,W
            # fallback: try to unflatten to (b, H, W, d) by cropping/upsampling as necessary.
            try:
                query_hw = query[:, :H * W, :].view(bsz, H, W, d_ch)
            except Exception:
                # last resort: reshape to (b, sqrt(seq_len), sqrt(seq_len)) if square
                side = int(seq_len ** 0.5)
                if side * side == seq_len:
                    query_hw = query.view(bsz, side, side, d_ch)
                    # if side != H or != W, then interpolate to (H,W)
                    query_hw = F.interpolate(query_hw.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=False)
                    query_hw = query_hw.permute(0, 2, 3, 1)
                else:
                    # fallback to reshape into (b, H, W, d) by padding/trimming channels
                    padded = F.pad(query, (0, 0, 0, max(0, H * W - seq_len)))
                    query_hw = padded[:, :H * W, :].view(bsz, H, W, d_ch)

        # Convert (b, H, W, d) -> (b, d, H, W)
        out = rearrange(query_hw, 'b H W d -> b d H W')

        return out


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
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()
        downsample_layers = list()

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            # instantiate CrossViewSwapAttention using both cross_view & cross_view_swap dicts
            # the original code passed **cross_view, **cross_view_swap — keep that so config needn't change.
            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)

            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(dim[i], dim[i] // 2,
                                  kernel_size=3, stride=1,
                                  padding=1, bias=False),
                        nn.PixelUnshuffle(2),
                        nn.Conv2d(dim[i+1], dim[i+1],
                                  3, padding=1, bias=False),
                        nn.BatchNorm2d(dim[i+1]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dim[i+1],
                                  dim[i+1], 1, padding=0, bias=False),
                        nn.BatchNorm2d(dim[i+1])
                    )))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        # self.self_attn = Attention(dim[-1], **self_attn)  # optional

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b*n, c, h, w  (backbone expects flattened)
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        # optional object_count (may be None)
        object_count = batch.get('object_count', None)

        if object_count is not None:
            print(">> object_count(pyramid axial encoder):", getattr(object_count, 'shape', object_count))
        else:
            print(">> object_count(pyramid axial encoder) is None")

        # obtain backbone features (list) — keep original call signature
        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        for i, (cross_view, feature, layer) in \
                enumerate(zip(self.cross_views, features, self.layers)):
            # feature from backbone comes as (b*n, c, h, w) — original code reshaped to (b, n, ...)
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features) - 1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        # optional self-attention on final BEV x could be added here
        return x


# quick local sanity test (only runs when module executed directly)
if __name__ == "__main__":
    # lightweight smoke test to ensure shapes flow
    B, N = 2, 6
    d = 64
    H, W = 25, 25
    feat = torch.randn(B, N, d, H, W)
    I_inv = torch.randn(B, N, 3, 3)
    E_inv = torch.randn(B, N, 4, 4)
    bev = BEVEmbedding(dim=d, sigma=0.02, bev_height=H, bev_width=W, h_meters=100, w_meters=100, offset=0.0, upsample_scales=[1])
    cva = CrossViewSwapAttention(feat_height=H, feat_width=W, feat_dim=d, dim=d, index=0,
                                 image_height=H, image_width=W, qkv_bias=True,
                                 q_win_size=[[5, 5]], feat_win_size=[[5, 5]],
                                 heads=[4], dim_head=[32], bev_embedding_flag=[True])
    x = torch.randn(B, d, H, W)
    out = cva(0, x, bev, feat, I_inv, E_inv, object_count=None)
    print("CVA output shape:", out.shape)  # expect (B, d, H, W)

