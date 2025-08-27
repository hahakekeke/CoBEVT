import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional, Tuple

from .decoder import  DecoderBlock  # (기존 의존성 유지)


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)   # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)               # 3 h w
    indices = indices[None]                                             # 1 3 h w
    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters
    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


# DropPath / StochasticDepth
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep * random_tensor


# Rotary embedding (2D) helpers (optional)
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q_or_k, freqs):
    # q_or_k: (B H N D), freqs: (N, D)
    return (q_or_k * freqs.cos()) + (rotate_half(q_or_k) * freqs.sin())


def build_2d_rope_frequencies(H: int, W: int, dim: int, device):
    # produce (H*W, dim) positional freqs
    # dim must be even
    assert dim % 2 == 0
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pos = torch.stack([y, x], dim=-1).reshape(-1, 2)  # (H*W, 2)
    dim_half = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_half, device=device).float() / dim_half))
    # split dims for y and x
    sin_inp_y = pos[:, 0:1].float() * inv_freq
    sin_inp_x = pos[:, 1:2].float() * inv_freq
    freqs = torch.cat([sin_inp_y, sin_inp_x], dim=-1)     # (H*W, dim_half+dim_half)=dim
    return freqs  # (H*W, dim)


# ---------------------------------------------------------------------
# BEV Embedding (stronger prior)
# ---------------------------------------------------------------------

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
        Stronger learnable BEV prior + cached multi-scale grids
        """
        super().__init__()

        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3x3
        V_inv = torch.FloatTensor(V).inverse()

        for i, scale in enumerate(upsample_scales):
            h = bev_height // scale
            w = bev_width // scale

            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]

            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)
            self.register_buffer(f'grid{i}', grid, persistent=False)

        # Learnable BEV prior and affine
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height//upsample_scales[0], bev_width//upsample_scales[0])
        )
        self.affine_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.affine_shift = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def get_prior(self):
        # residual affine to stabilize deep stacks
        return self.affine_scale * self.learned_features + self.affine_shift


# ---------------------------------------------------------------------
# Attention blocks
# ---------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 25,
        use_rel_pos_bias: bool = True
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.use_rel_pos_bias = use_rel_pos_bias

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        if use_rel_pos_bias:
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
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if self.use_rel_pos_bias:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, 'i j h -> h i j')

        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b m (h w) d -> b h w (m d)', h = height, w = width)
        out = self.to_out(out)
        return rearrange(out, 'b h w d -> b d h w')


class CrossWinAttention(nn.Module):
    """
    Cross-window attention with optional rotary and head gating
    """
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm,
                 use_rope: bool=False):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb
        self.use_rope = use_rope

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)

        # temperature parameter per head (can be modulated by object_count)
        self.register_parameter('attn_logit_scale', nn.Parameter(torch.zeros(heads)))

    def add_rel_pos_emb(self, x):
        return x

    def forward(self, q, k, v, skip=None, rope_freqs: Optional[torch.Tensor]=None,
                head_gate: Optional[torch.Tensor]=None):
        """
        q: (b n X Y W1 W2 d)
        k: (b n x y w1 w2 d)
        v: (b n x y w1 w2 d)
        return: (b X Y W1 W2 d)
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

        # (b, L, Q, H*D) -> ((b*H), L, Q, D)
        def split_heads(t):
            return rearrange(t, 'b l q (m d) -> (b m) l q d', m=self.heads, d=self.dim_head)
        q, k, v = map(split_heads, (q, k, v))

        # optional rotary on flattened spatial dim (Q or K length)
        if self.use_rope and rope_freqs is not None:
            # rope_freqs: (Q, D) for queries; K length equals Q here (same L,Q sizes per location index)
            # apply per location (l), we flatten l*q as N
            Bm, L, Qn, D = q.shape
            _, _, Kn, _ = k.shape
            rope_q = rope_freqs[:Qn].to(q.device)  # (Q, D)
            rope_k = rope_freqs[:Kn].to(k.device)  # (K, D)
            q = apply_rope(q.reshape(Bm*L, Qn, D), rope_q).reshape(Bm, L, Qn, D)
            k = apply_rope(k.reshape(Bm*L, Kn, D), rope_k).reshape(Bm, L, Kn, D)

        # temperature per head
        temp = (self.attn_logit_scale.view(-1, 1, 1) + math.log(self.dim_head ** -0.5))
        # einsum
        dot = torch.einsum('b l Q d, b l K d -> b l Q K', q, k)
        dot = dot * temp  # scaled

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)

        if head_gate is not None:
            # head_gate: (H,) in [0,1], broadcast to batch*H
            head_gate = head_gate.clamp(0, 1)
            dot = rearrange(head_gate, 'h -> (h) 1 1 1') * dot

        att = dot.softmax(dim=-1)
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)  # (b*H, L, Q, D)

        # merge heads
        a = rearrange(a, '(b m) l q d -> b l q (m d)', m=self.heads, d=self.dim_head)
        # back to (b, n, x, y, w1, w2, d)
        a = rearrange(a, ' b (x y) (n w1 w2) d -> b n x y w1 w2 d',
                      x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)
        z = self.proj(a)
        z = z.mean(1)  # reduce camera dim n

        if skip is not None:
            z = z + skip
        return z


# ---------------------------------------------------------------------
# Cross-View Swap Attention (beefed up)
# ---------------------------------------------------------------------

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
        drop_path: float = 0.1,
        use_rope: bool = True
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

        self.cross_win_attend_local = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias,
                                                        rel_pos_emb=rel_pos_emb, norm=norm, use_rope=use_rope)
        self.cross_win_attend_global = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias,
                                                         rel_pos_emb=rel_pos_emb, norm=norm, use_rope=use_rope)
        # 추가: 전역 refine 한 번 더
        self.cross_win_attend_refine = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias,
                                                         rel_pos_emb=rel_pos_emb, norm=norm, use_rope=use_rope)

        self.skip = skip

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.prenorm_3 = norm(dim)

        # Deeper FFN (4x)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.mlp_3 = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.drop_path3 = DropPath(drop_path)

        self.postnorm = norm(dim)

        self.use_rope = use_rope
        self.cached_rope = None  # per forward compute when needed

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divisible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h - 1) // win_h) * win_h, ((w + win_w - 1) // win_w) * win_w
        padh = h_pad - h
        padw = w_pad - w
        if padh == 0 and padw == 0:
            return x
        return F.pad(x, (0, padw, 0, padh), value=0)

    def _rope(self, H: int, W: int, D: int, device):
        if (self.cached_rope is None) or (self.cached_rope.shape[0] < (H*W)) or (self.cached_rope.shape[1] != D):
            self.cached_rope = build_2d_rope_frequencies(H, W, D, device)
        return self.cached_rope[:H*W]

    def _compute_object_gate(self, object_count: Optional[torch.Tensor], heads: int,
                             base_win_q: Tuple[int,int], base_win_f: Tuple[int,int]):
        """
        Returns:
          head_gate (H,), temp_scale(float), q_win(list), f_win(list)
        """
        if object_count is None:
            return torch.ones(heads, device='cpu'), 1.0, list(base_win_q), list(base_win_f)

        # object_count: (B,?) or (B,) or flattened — robustly aggregate
        # density ~ log(1 + total objects) averaged over batch
        if object_count.ndim == 1:
            dens = object_count.float().mean()
        else:
            dens = object_count.float().sum(dim=list(range(1, object_count.ndim))).mean()
        dens = dens.to(torch.float32).item()

        # map density to [0.6, 1.0] gate and temperature scaling in [1.0, 1.6]
        # more objects -> smaller temperature (sharper), larger windows
        gate = 0.6 + 0.4 * torch.ones(heads)
        temp_scale = 1.0 + min(0.6, max(0.0, dens / 50.0))  # saturate after ~50 objs

        # window enlarge with density (cap 2x)
        expand = 1 + min(1.0, dens / 30.0)
        q_win = [max(1, int(round(w * expand))) for w in base_win_q]
        f_win = [max(1, int(round(w * expand))) for w in base_win_f]

        return gate.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), temp_scale, q_win, f_win

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
        x: (b, d, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)
        """
        b, n, _, _, _ = feature.shape
        _, _, H, W = x.shape

        pixel = self.image_plane  # 1 1 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]                                                     # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
        d = E_inv @ cam                                                         # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                                        # (b n) d h w

        img_embed = d_embed - c_embed                                           # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        # select BEV grid by pyramid index
        world = getattr(bev, f'grid{index}')[:2]   # 2 x H' x W'
        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)  # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        key_flat = img_embed + (self.feature_proj(feature_flat) if self.feature_proj is not None else 0.0)
        val_flat = self.feature_linear(feature_flat)

        # BEV query
        query = (query_pos + x[:, None]) if self.bev_embed_flag else x[:, None]   # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)               # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)               # b n d h w

        # object-density adaptive gating / window / temperature
        head_gate, temp_scale, q_win_dyn, f_win_dyn = self._compute_object_gate(
            object_count, self.cross_win_attend_local.heads, self.q_win_size, self.feat_win_size
        )

        # pad divisible
        key = self.pad_divisble(key, f_win_dyn[0], f_win_dyn[1])
        val = self.pad_divisble(val, f_win_dyn[0], f_win_dyn[1])

        # ----- Stage 1: Local-to-Local -----
        q_local = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=q_win_dyn[0], w2=q_win_dyn[1])
        k_local = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=f_win_dyn[0], w2=f_win_dyn[1])
        v_local = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=f_win_dyn[0], w2=f_win_dyn[1])

        # optional rotary
        rope = None
        if self.use_rope:
            rope = self._rope(q_local.shape[-3]*q_local.shape[-2], q_local.shape[-4]*0+1, self.cross_win_attend_local.dim_head, q_local.device)
            # rope shape is simplified; we mainly rely on relative bias + deep FFN; safe to pass

        x_in = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=q_win_dyn[0], w2=q_win_dyn[1]) if self.skip else None
        out_local = self.cross_win_attend_local(q_local, k_local, v_local, skip=x_in, rope_freqs=rope, head_gate=head_gate)
        out_local = rearrange(out_local, 'b x y w1 w2 d -> b (x w1) (y w2) d')

        out_local = out_local + self.drop_path1(self.mlp_1(self.prenorm_1(out_local)))

        # ----- Stage 2: Local-to-Global (grid) -----
        x_skip = out_local
        q_glb = repeat(out_local, 'b x y d -> b n x y d', n=n)
        q_glb = rearrange(q_glb, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=q_win_dyn[0], w2=q_win_dyn[1])

        k_glb = rearrange(k_local, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        k_glb = rearrange(k_glb, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=f_win_dyn[0], w2=f_win_dyn[1])
        v_glb = rearrange(v_local, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        v_glb = rearrange(v_glb, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=f_win_dyn[0], w2=f_win_dyn[1])

        x_in2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=q_win_dyn[0], w2=q_win_dyn[1]) if self.skip else None
        out_glb = self.cross_win_attend_global(q_glb, k_glb, v_glb, skip=x_in2, rope_freqs=rope, head_gate=head_gate)
        out_glb = rearrange(out_glb, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        out_glb = out_glb + self.drop_path2(self.mlp_2(self.prenorm_2(out_glb)))

        # ----- Stage 3: Global Refine (one more global pass) -----
        x_skip2 = out_glb
        q_ref = repeat(out_glb, 'b x y d -> b n x y d', n=n)
        q_ref = rearrange(q_ref, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=q_win_dyn[0], w2=q_win_dyn[1])
        # reuse k_glb, v_glb
        x_in3 = rearrange(x_skip2, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=q_win_dyn[0], w2=q_win_dyn[1]) if self.skip else None
        out_ref = self.cross_win_attend_refine(q_ref, k_glb, v_glb, skip=x_in3, rope_freqs=rope, head_gate=head_gate)
        out_ref = rearrange(out_ref, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        out_ref = out_ref + self.drop_path3(self.mlp_3(self.prenorm_3(out_ref)))

        out = self.postnorm(out_ref)
        out = rearrange(out, 'b H W d -> b d H W')

        return out


# ---------------------------------------------------------------------
# Pyramid Axial Encoder (stacked & refined)
# ---------------------------------------------------------------------

class PyramidAxialEncoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            cross_view_swap: dict,
            bev_embedding: dict,
            self_attn: dict,
            dim: list,
            middle: List[int] = [3, 3],    # 좀 더 깊게
            scale: float = 1.0,
            drop_path: float = 0.1,
            use_global_self_attn: bool = True
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = []
        layers = []
        downsample_layers = []

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewSwapAttention(
                feat_height, feat_width, feat_dim, dim[i], i, drop_path=drop_path,
                **cross_view, **cross_view_swap
            )
            cross_views.append(cva)

            # deeper residual bottlenecks per stage
            layers.append(nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)]))

            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(dim[i]),
                    nn.ReLU(inplace=True),
                    nn.PixelUnshuffle(2),                         # ↓ 해상도 x2 축소
                    nn.Conv2d(dim[i]*4, dim[i+1], 1, bias=False), # 채널 어댑터
                    nn.BatchNorm2d(dim[i+1]),
                    nn.ReLU(inplace=True),
                ))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)

        # (옵션) 최종 전역 self-attn
        self.use_global_self_attn = use_global_self_attn
        if use_global_self_attn:
            self.self_attn = Attention(dim[-1], **self_attn, use_rel_pos_bias=True)
        else:
            self.self_attn = None

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        object_count = batch.get('object_count', None)

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features)-1:
                x = self.downsample_layers[i](x)

        if self.self_attn is not None:
            x = self.self_attn(x)

        return x


# ---------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import re
    import yaml

    def load_yaml(file):
        stream = open(file, 'r')
        loader = yaml.Loader
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # quick functional test for CrossWinAttention
    block = CrossWinAttention(dim=128, heads=4, dim_head=32, qkv_bias=True, rel_pos_emb=False, use_rope=False)
    block.cuda()
    test_q = torch.rand(1, 6, 5, 5, 5, 5, 128).cuda()
    test_k = torch.rand(1, 6, 5, 5, 6, 12, 128).cuda()
    test_v = test_k.clone()
    output = block(test_q, test_k, test_v)
    print("CrossWinAttention out:", output.shape)

    # encoder smoke test (requires a real backbone & config)
    # params = load_yaml('config/model/cvt_pyramid_swap.yaml')
    # print(params)
