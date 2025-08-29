# encoder_pyramid_axial.py (MaxViT-like modifications for CoBEVT)
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List
from .decoder import DecoderBlock

from typing import Optional

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)  # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)  # 3 h w
    indices = indices[None]  # 1 3 h w

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
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.
        """
        super().__init__()

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters,
                            offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()  # 3 3

        for i, scale in enumerate(upsample_scales):
            # each decoder block upsamples the bev embedding by a factor of 2
            h = bev_height // scale
            w = bev_width // scale

            # bev coordinates
            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]

            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')  # 3 (h w)
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)  # 3 h w
            # egocentric frame
            self.register_buffer('grid%d' % i, grid, persistent=False)

        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim,
                                bev_height // upsample_scales[0],
                                bev_width // upsample_scales[0]))  # d h w

    def get_prior(self):
        return self.learned_features


# ---------------------------
# Helper modules (MBConv, FFN, Window/Grid Attention wrappers)
# ---------------------------

class MBConv(nn.Module):
    """Simplified MBConv (Expansion -> Depthwise -> SE -> Projection)"""
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, se_ratio=0.25, act=nn.GELU):
        super().__init__()
        hidden = in_ch * expansion
        self.use_expand = (expansion != 1)
        if self.use_expand:
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                act()
            )
        else:
            hidden = in_ch

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            act()
        )

        # Squeeze-and-Excitation
        if se_ratio is not None and 0 < se_ratio <= 1:
            se_ch = max(1, int(in_ch * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden, se_ch, 1),
                act(),
                nn.Conv2d(se_ch, hidden, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.activation = act()
        self.out_ch = out_ch
        self.in_ch = in_ch

    def forward(self, x):
        identity = x
        out = x
        if self.use_expand:
            out = self.expand(out)
        out = self.depthwise(out)
        if self.se is not None:
            out = out * self.se(out)
        out = self.project(out)
        # Residual if same channels
        if identity.shape == out.shape:
            out = out + identity
        return out


class FFN(nn.Module):
    """MLP for features (applies last dim MLP on flattened HxW-> ... )"""
    def __init__(self, dim, expansion=4, dropout=0.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # expects x: b, seq, dim OR b, H, W, dim
        if x.dim() == 4 and x.shape[1] != x.shape[-1]:
            # b, H, W, d -> flatten last two dims
            b, H, W, d = x.shape
            x = x.view(b, H * W, d)
            out = self.net(x)
            out = out.view(b, H, W, d)
            return out
        else:
            return self.net(x)


class WindowAttention(nn.Module):
    """Wrapper using existing Attention style (self-attention with rel pos bias).
       This will operate on windows shaped as (b, n_windows, win_h, win_w, dim)
    """
    def __init__(self, dim, dim_head=32, dropout=0., window_size=7):
        super().__init__()
        self.att = Attention(dim=dim, dim_head=dim_head, dropout=dropout, window_size=window_size)

    def forward(self, x):
        # x: b, n_win, win_h, win_w, d  -> our Attention expects b d h w input previously,
        # but in this file Attention.forward expects x: b d h w. We will adapt: permute.
        b, nwin, wh, ww, d = x.shape
        x = x.view(b * nwin, d, wh, ww)
        out = self.att(x)  # returns b*nwin, d, wh, ww
        out = out.view(b, nwin, d, wh, ww)
        # convert to b, nwin, wh, ww, d
        out = out.permute(0, 1, 3, 4, 2)
        return out


class GridAttention(nn.Module):
    """Global grid attention operating on grid patches assembled as input."""
    def __init__(self, dim, dim_head=32, dropout=0., grid_size=7):
        super().__init__()
        self.att = Attention(dim=dim, dim_head=dim_head, dropout=dropout, window_size=grid_size)

    def forward(self, x):
        # x: b, n_grid, g_h, g_w, d -> same handling as WindowAttention
        b, ngrid, gh, gw, d = x.shape
        x = x.view(b * ngrid, d, gh, gw)
        out = self.att(x)
        out = out.view(b, ngrid, d, gh, gw)
        out = out.permute(0, 1, 3, 4, 2)
        return out


# ---------------------------
# Existing Attention class (kept - but note this file already has one)
# We'll reuse the Attention defined in original file; ensure defined earlier.
# The file already defined Attention class earlier in the project (not repeated here)
# ---------------------------

# The file's original Attention class (windowed attention with rel pos bias)
# is used above inside WindowAttention and GridAttention.

# ---------------------------
# CrossWinAttention (slightly modified to accept temperature scalar)
# ---------------------------

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
        _, view_size, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        _, _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

        # flattening
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (X Y) (n W1 W2) (heads dim_head)
        k = self.to_k(k)                                # b (X Y) (n w1 w2) (heads dim_head)
        v = self.to_v(v)                                # b (X Y) (n w1 w2) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = torch.einsum('b l Q d, b l K d -> b l Q K', q, k)  # b (X Y) (n W1 W2) (n w1 w2)
        dot = dot * self.scale

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)

        # Apply temperature scaling if provided: higher temperature -> sharper softmax
        if temperature is not None:
            # temperature > 1 will increase logits -> sharper; <1 softer
            dot = dot * temperature

        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)  # b (X Y) (n W1 W2) d
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, ' b (x y) (n w1 w2) d -> b n x y w1 w2 d',
            x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)

        # Combine multiple heads
        z = self.proj(a)

        # reduce n: (b n X Y W1 W2 d) -> (b X Y W1 W2 d)
        z = z.mean(1)  # for sequential usage, we cannot reduce it!

        # Optional skip connection
        if skip is not None:
            z = z + skip
        return z


# ---------------------------
# CrossViewSwapAttention (reworked: MBConv + block-attn + FFN + grid-attn + FFN)
# ---------------------------

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
        rel_pos_emb: bool = False,  # to-do
        no_image_features: bool = False,
        skip: bool = True,
        norm=nn.LayerNorm,
    ):
        super().__init__()

        # 1 1 3 h w
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

        # Cross attention modules (per stage)
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip

        # MaxViT-like internal components:
        # MBConv operates on combined feature maps to strengthen local representations
        self.mbconv = MBConv(in_ch=dim, out_ch=dim, expansion=4, kernel_size=3, se_ratio=0.25)

        # Local window and grid attention wrappers (for optional extra refining)
        self.win_att = WindowAttention(dim=dim, dim_head=dim_head[index], dropout=0., window_size=self.q_win_size[0])
        self.grid_att = GridAttention(dim=dim, dim_head=dim_head[index], dropout=0., grid_size=self.feat_win_size[0])

        # FFNs
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

        # Extra FFN after MBConv+attentions for more capacity
        self.post_mlp = FFN(dim, expansion=2, dropout=0.0)

        # projection for cross-attention output if needed
        self.proj_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def compute_temperature(self, object_count: Optional[torch.Tensor], base_temp=1.0, max_temp=3.0):
        """
        Compute temperature from object_count tensor.
        If object_count is None -> temperature = base_temp
        Else -> map average object count to [base_temp, max_temp]
        """
        if object_count is None:
            return base_temp
        # object_count shape likely (b, ) or (num_classes,), try robust handling
        try:
            # average across classes and batch
            avg_count = float(object_count.float().mean().item())
        except Exception:
            try:
                avg_count = float(object_count.mean().item())
            except Exception:
                avg_count = 0.0
        # map to temperature (simple linear mapping with clamp)
        # assume typical object counts in dataset < 20; scale accordingly
        temp = base_temp + (avg_count / 20.0) * (max_temp - base_temp)
        temp = max(base_temp, min(temp, max_temp))
        return float(temp)

    def forward(
        self,
        index: int,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        object_count: Optional[torch.Tensor] = None,  # object_count
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """

        # 디버깅 (원하면 제거)
        if object_count is not None:
            # object_count could be per-batch or per-class; print shape for debugging
            print(">> object_count(crossviewswapattention):", object_count.shape, object_count)
        else:
            print(">> object_count(crossviewswapattention) is None")

        b, n, _, _, _ = feature.shape
        _, _, H, W = x.shape

        pixel = self.image_plane  # 1 1 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]  # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]  # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)  # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')  # 1 1 3 (h w)
        cam = I_inv @ pixel_flat  # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)  # b n 4 (h w)
        d = E_inv @ cam  # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)  # (b n) 4 h w
        d_embed = self.img_embed(d_flat)  # (b n) d h w

        img_embed = d_embed - c_embed  # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)  # (b n) d h w

        # select bev grid for given index
        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        elif index == 3:
            world = bev.grid3[:2]
        else:
            # fallback: pick closest available
            world = bev.grid0[:2]

        if self.bev_embed_flag:
            # 2 H W
            w_embed = self.bev_embed(world[None])  # 1 d H W
            bev_embed = w_embed - c_embed  # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)  # (b n) d H W
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)  # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')  # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)  # (b n) d h w
        else:
            key_flat = img_embed  # (b n) d h w

        val_flat = self.feature_linear(feature_flat)  # (b n) d h w

        # Expand + refine the BEV embedding
        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]  # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w

        # === MBConv local refinement on values (acts like MaxViT's MBConv pre-attention) ===
        # val_flat: b n d h w -> process per-camera separately
        v_shape = val.shape  # b n d h w
        val_reshaped = val.view(b * n, val.shape[2], val.shape[3], val.shape[4])  # (b*n) d h w
        val_refined = self.mbconv(val_reshaped)  # (b*n) d h w
        val = val_refined.view(b, n, val_refined.shape[1], val_refined.shape[2], val_refined.shape[3])

        # optionally refine key similarly
        key_reshaped = key.view(b * n, key.shape[2], key.shape[3], key.shape[4])
        key_refined = self.mbconv(key_reshaped)
        key = key_refined.view(b, n, key_refined.shape[1], key_refined.shape[2], key_refined.shape[3])

        # normalize features
        key = key / (key.norm(dim=2, keepdim=True) + 1e-7)
        val = val / (val.norm(dim=2, keepdim=True) + 1e-7)

        # pad divisible
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # Compute temperature from object_count to modulate attention sharpness
        temperature = self.compute_temperature(object_count, base_temp=1.0, max_temp=3.0)

        # local-to-local cross-attention
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition

        # perform cross-window attention (camera-wise aggregation)
        # use cross_win_attend_1 and pass temperature to sharpen/soften
        query_windows = self.cross_win_attend_1(query, key, val,
                                                skip=rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                                                               w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None,
                                                temperature=temperature)

        # reverse window to feature: query_windows is b x y w1 w2 d -> flatten
        query = rearrange(query_windows, 'b x y w1 w2 d  -> b (x w1) (y w2) d')  # reverse window to feature

        # post-attn MLP
        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)  # b n x y d

        # local-to-global cross-attention
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition

        # prepare key/val for grid attention (grid partition)
        key_grid = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')  # reverse window to feature
        key_grid = rearrange(key_grid, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                             w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition
        val_grid = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')  # reverse window to feature
        val_grid = rearrange(val_grid, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                             w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition

        query_grid_out = self.cross_win_attend_2(query,
                                                  key_grid,
                                                  val_grid,
                                                  skip=rearrange(x_skip,
                                                                 'b (x w1) (y w2) d -> b x y w1 w2 d',
                                                                 w1=self.q_win_size[0],
                                                                 w2=self.q_win_size[1])
                                                  if self.skip else None,
                                                  temperature=temperature)
        # reverse grid to feature
        query = rearrange(query_grid_out, 'b x y w1 w2 d  -> b (x w1) (y w2) d')

        query = query + self.mlp_2(self.prenorm_2(query))

        # final norm & reshape to b d H W
        query = self.postnorm(query)
        # optional extra FFN (channel-wise)
        # query: b, (H W), d -> apply channel-FFN
        bsz = query.shape[0]
        total_hw = query.shape[1]
        d_ch = query.shape[2]
        # run FFN on sequence dimension
        query_seq = query.view(bsz, total_hw, d_ch)
        query_seq = self.post_mlp(query_seq)
        query = query_seq.view(bsz, int(total_hw ** 0.5), int(total_hw ** 0.5), d_ch)  # assume square

        # rearrange to b d H W
        Hnew = query.shape[1]
        Wnew = query.shape[2]
        query = rearrange(query, 'b H W d -> b d H W')

        # projection
        query = self.proj_out(query.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # b d H W

        return query


# ---------------------------
# PyramidAxialEncoder (mostly unchanged, but uses new CrossViewSwapAttention)
# ---------------------------

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
        # note: we intentionally keep self_attn optional / commented

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w -> (b*n) c H W for backbone
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        # ✅ 여기서 object_count 가져오기
        object_count = batch.get('object_count', None)

        # 디버깅
        if object_count is not None:
            print(">> object_count(pyramid axial encoder):", object_count.shape, object_count)
        else:
            print(">> object_count(pyramid axial encoder) is None")

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        for i, (cross_view, feature, layer) in \
                enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        return x


# Test / debug main (kept for local test)
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

    # quick sanity tests
    block = CrossWinAttention(dim=128,
                              heads=4,
                              dim_head=32,
                              qkv_bias=True,)
    block.cuda()
    test_q = torch.rand(1, 6, 5, 5, 5, 5, 128)
    test_k = test_v = torch.rand(1, 6, 5, 5, 6, 12, 128)
    test_q = test_q.cuda()
    test_k = test_k.cuda()
    test_v = test_v.cuda()

    output = block(test_q, test_k, test_v)
    print(output.shape)

    image = torch.rand(1, 6, 128, 28, 60)            # b n c h w
    I_inv = torch.rand(1, 6, 3, 3)           # b n 3 3
    E_inv = torch.rand(1, 6, 4, 4)           # b n 4 4

    feature = torch.rand(1, 6, 128, 25, 25)

    x = torch.rand(1, 128, 25, 25)                     # b d H W

    # quick CrossViewSwapAttention instantiation test
    cva = CrossViewSwapAttention(
        feat_height=25,
        feat_width=25,
        feat_dim=128,
        dim=128,
        index=0,
        image_height=28,
        image_width=60,
        qkv_bias=True,
        q_win_size=[5, 5],
        feat_win_size=[5, 5],
        heads=[4],
        dim_head=[32],
        bev_embedding_flag=[True, False, False, False],
    )
    cva.cuda()
    batch = {
        'image': image,
        'intrinsics': I_inv,
        'extrinsics': E_inv,
        'object_count': torch.tensor([10.0])  # dummy
    }

    bev_emb = BEVEmbedding(dim=128, sigma=0.02, bev_height=25, bev_width=25, h_meters=100, w_meters=100, offset=0.0,
                           upsample_scales=[1,])
    bev_emb.register_buffer('grid0', bev_emb.grid0)
    out = cva(0, x.cuda(), bev_emb, feature.cuda(), I_inv.cuda(), E_inv.cuda(), object_count=batch['object_count'])
    print("CVA out:", out.shape)
