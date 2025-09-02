import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional
from .decoder import DecoderBlock  # decoder 연동 고려(현재 파일 내부 사용 없음, import만 유지)

# ---------------------------
# Utility & Basic Blocks
# ---------------------------

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)  # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)               # 3 h w
    indices = indices[None]                                             # 1 3 h w
    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters
    return [
        [0., -sw,           w/2.],
        [-sh,  0.,  h*offset + h/2.],
        [0.,   0.,         1.]
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
        self.kwargs = {'stride': stride, 'padding': padding}

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
        dim: embedding size
        sigma: scale for initializing embedding
        """
        super().__init__()

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3x3
        V_inv = torch.FloatTensor(V).inverse()  # 3x3

        for i, scale in enumerate(upsample_scales):
            # each decoder block upsamples the bev embedding by a factor of 2
            h = bev_height // scale
            w = bev_width // scale

            # bev coordinates
            grid = generate_grid(h, w).squeeze(0)  # 3 h w
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]

            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')  # 3 (h w)
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)  # 3 h w
            # egocentric frame
            self.register_buffer(f'grid{i}', grid, persistent=False)

        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height // upsample_scales[0], bev_width // upsample_scales[0])
        )  # d h w

    def get_prior(self):
        return self.learned_features


# ---------------------------
# Attention Blocks
# ---------------------------

class Attention(nn.Module):
    """Axial/global attention used for BEV self-attention (refined)."""
    def __init__(self, dim, dim_head=32, dropout=0., window_size=25):
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
        batch, _, height, width = x.shape
        h = self.heads

        x = rearrange(x, 'b d h w -> b (h w) d')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b m (h w) d -> b h w (m d)', h=height, w=width)
        out = self.to_out(out)
        return rearrange(out, 'b h w d -> b d h w')


class CrossWinAttention(nn.Module):
    """
    Cross-window attention along camera views (stronger version)
    - learned temperature
    - learned per-camera weights (instead of simple mean)
    """
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm, dropout=0.0):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.dropout = nn.Dropout(dropout)

        # learned temperature (per head)
        self.logit_scale = nn.Parameter(torch.zeros(heads))  # starts at 1.0 after exp
        # learned per-camera weights: small MLP to score each camera token group
        self.cam_score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def add_rel_pos_emb(self, x):
        return x

    def forward(self, q, k, v, skip=None):
        """
        q: (b n X Y W1 W2 d)
        k: (b n x y w1 w2 d)
        v: (b n x y w1 w2 d)
        return: (b X Y W1 W2 d)
        """
        assert k.shape == v.shape
        b, n, q_height, q_width, q_win_h, q_win_w, d_model = q.shape
        _, _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

        # flattening
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        # Project with multiple heads
        q = self.to_q(q)  # b (X Y) (n W1 W2) (heads dim_head)
        k = self.to_k(k)
        v = self.to_v(v)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras (with learned temperature per head)
        scale = (self.dim_head ** -0.5) * (self.logit_scale.exp().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        dot = scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)  # b_head (X Y) (n W1 W2) (n w1 w2)
        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        att = dot.softmax(dim=-1)
        att = self.dropout(att)

        # Combine values (image level features)
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)  # b_head (X Y) (n W1 W2) d

        # merge heads back
        a = rearrange(a, '(b m) l q d -> b l q (m d)', m=self.heads, b=b)

        # reshape to camera groups again to compute per-camera weights
        # current q groups tokens across (n * W1 * W2). We compute weights per camera by averaging within each camera group.
        # First, unroll q locations:
        a_cam = rearrange(a, 'b l (n w1 w2) d -> b l n (w1 w2) d', n=n, w1=q_win_h, w2=q_win_w)
        a_cam_mean = a_cam.mean(dim=3)  # b l n d

        # score per camera and softmax-normalize
        cam_weights = self.cam_score(a_cam_mean)  # b l n 1
        cam_weights = torch.softmax(cam_weights, dim=2)  # b l n 1

        # apply weights back to the fine tokens
        a_weighted = (a_cam * cam_weights.unsqueeze(3)).sum(dim=2)  # b l (w1 w2) d
        a_weighted = rearrange(a_weighted, 'b l (w1 w2) d -> b l w1 w2 d', w1=q_win_h, w2=q_win_w)

        # project
        z = self.proj(a_weighted)  # b (X Y) W1 W2 d

        # Optional skip connection
        if skip is not None:
            z = z + skip
        return z


class BEVTransformerBlock(nn.Module):
    """
    Windowed BEV self-attention block to strengthen global reasoning.
    """
    def __init__(self, dim, window=(25, 25), heads=8, dim_head=32, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.win_h, self.win_w = window
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, dim_head=dim_head, dropout=dropout, window_size=max(self.win_h, self.win_w))
        self.drop_path = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def window_partition(self, x):
        # x: (b, d, H, W) -> (b, num_win_h, num_win_w, d, wh, ww)
        b, d, H, W = x.shape
        pad_h = (self.win_h - H % self.win_h) % self.win_h
        pad_w = (self.win_w - W % self.win_w) % self.win_w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[-2:]
        x = rearrange(x, 'b d (nh wh) (nw ww) -> b nh nw d wh ww', wh=self.win_h, ww=self.win_w)
        return x, H, W, pad_h, pad_w

    def window_reverse(self, x, H, W, pad_h, pad_w):
        # x: (b, nh, nw, d, wh, ww)
        x = rearrange(x, 'b nh nw d wh ww -> b d (nh wh) (nw ww)')
        if pad_h or pad_w:
            x = x[:, :, :H - pad_h, :W - pad_w]
        return x

    def forward(self, x):
        # windowed self-attention
        x_win, H, W, ph, pw = self.window_partition(x)
        b, nh, nw, d, wh, ww = x_win.shape

        # attn expects (b, d, h, w)
        x_win = rearrange(x_win, 'b nh nw d wh ww -> (b nh nw) d wh ww')
        res = x_win
        x_win = rearrange(x_win, 'b d h w -> b (h w) d')
        x_win = self.norm1(x_win)
        x_win = rearrange(x_win, 'b n d -> b d 1 n')  # fake H=1 for Attention
        x_win = rearrange(x_win, 'b d h n -> b d h n')  # no-op, just to keep dims explicit
        x_win = rearrange(x_win, 'b d 1 n -> b d 1 n')  # keep

        # Use original Attention on (b, d, wh, ww)
        x_win = rearrange(x_win, '(b) d 1 (h w) -> b d h w', b=res.shape[0], h=res.shape[2], w=res.shape[3])
        x_win = self.attn(x_win)
        x_win = self.drop_path(x_win) + res

        # MLP
        y = rearrange(x_win, 'b d h w -> b (h w) d')
        y = self.norm2(y)
        y = self.mlp(y)
        y = rearrange(y, 'b (h w) d -> b d h w', h=res.shape[2], w=res.shape[3])

        x_win = x_win + y
        x_win = self.window_reverse(rearrange(x_win, '(b) d wh ww -> b 1 1 d wh ww', b=b), H, W, ph, pw)
        return x_win


class CrossViewSwapAttention(nn.Module):
    """
    강화된 Cross-View Attention:
      - 이미지 기하 임베딩(img_embed) + 특징(feature) 둘 다 활용
      - 로컬-투-로컬, 로컬-투-글로벌 2단 주의
      - 카메라 가중치 학습(learned per-camera weighting)
      - residual / MLP / LayerNorm / gating
    """
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
        dropout: float = 0.1,
        residual_scale: float = 1.0,
    ):
        super().__init__()

        # 1 1 3 h w (pixel grid in image plane)
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, dim, 1, bias=False)
        )

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_dim, dim, 1, bias=False)
            )

        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.q_win_size = q_win_size[index]
        self.feat_win_size = feat_win_size[index]
        self.rel_pos_emb = rel_pos_emb

        # 강화된 cross attention (learned temp + per-camera weighting)
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias, dropout=dropout)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias, dropout=dropout)
        self.skip = skip
        self.residual_scale = residual_scale

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

        # gating for skip connection blending (helps when decoder changes later)
        self.skip_gate_1 = nn.Parameter(torch.tensor(0.5))
        self.skip_gate_2 = nn.Parameter(torch.tensor(0.5))

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divisible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h - 1) // win_h) * win_h, ((w + win_w - 1) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def forward(
        self,
        index: int,
        x: torch.FloatTensor,
        bev: 'BEVEmbedding',
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        object_count: Optional[torch.Tensor] = None,
    ):
        """
        x: (b, c, H, W)             # current BEV feature (query seed)
        feature: (b, n, dim_in, h, w)  # image features per camera
        I_inv: (b, n, 3, 3), E_inv: (b, n, 4, 4)
        Returns: (b, d, H, W)
        """

        b, n, _, _, _ = feature.shape
        _, _, H, W = x.shape

        pixel = self.image_plane  # 1 1 3 h w
        _, _, _, h, w = pixel.shape

        # camera centers
        c = E_inv[..., -1:]  # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]  # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)  # (b n) d 1 1

        # image direction embedding
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')  # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                               # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)    # b n 4 (h w)
        d = E_inv @ cam                                        # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)  # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                       # (b n) d h w

        img_embed = d_embed - c_embed
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)  # (b n) d h w

        # select BEV grid for this stage
        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        else:
            world = bev.grid3[:2]

        # positional BEV embedding (optional)
        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])  # 1 d H W
            bev_embed = w_embed - c_embed          # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)  # b n d H W

        # prepare key/value
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')  # (b n) dim_in h w
        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)  # (b n) d h w
        else:
            key_flat = img_embed
        val_flat = self.feature_linear(feature_flat)  # (b n) d h w

        # Expand + refine the BEV embedding
        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]  # b n d H W

        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w

        # pad divisible
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # ---------- stage 1: local-to-local ----------
        q1 = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                       w1=self.q_win_size[0], w2=self.q_win_size[1])
        k1 = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        v1 = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        skip1 = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

        out1 = self.cross_win_attend_1(q1, k1, v1, skip=skip1)
        out1 = rearrange(out1, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        out1 = out1 + self.mlp_1(self.prenorm_1(out1))
        x_skip = out1
        out1 = repeat(out1, 'b x y d -> b n x y d', n=n)

        # ---------- stage 2: local-to-global ----------
        q2 = rearrange(out1, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                       w1=self.q_win_size[0], w2=self.q_win_size[1])
        k2 = rearrange(k1, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        k2 = rearrange(k2, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        v2 = rearrange(v1, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        v2 = rearrange(v2, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        skip2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

        out2 = self.cross_win_attend_2(q2, k2, v2, skip=skip2)
        out2 = rearrange(out2, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        out2 = out2 + self.mlp_2(self.prenorm_2(out2))
        out2 = self.postnorm(out2)

        # fuse back to (b, d, H, W)
        out = rearrange(out2, 'b H W d -> b d H W')

        # gated residual to control stability
        if self.skip:
            out = out + self.residual_scale * x * torch.sigmoid(self.skip_gate_2)

        return out


# ---------------------------
# Encoder (PyramidAxialEncoder)
# ---------------------------

class PyramidAxialEncoder(nn.Module):
    """
    정확도 강화 버전의 Encoder.
    - 각 stage에서 Cross-View 강화 + BEV Self-Attention (전역 문맥)
    - 향후 decoder를 위한 multi-scale BEV pyramid 출력
    - return_dict=True: {'bev', 'pyramid', 'backbone_feats'} 반환 (권장)
      return_dict=False: 기존과 동일하게 최종 텐서만 반환
    """
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
        return_dict: bool = True,
        bev_sa_per_stage: int = 1,     # BEV self-attn blocks per stage
        bev_sa_dropout: float = 0.1,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        self.return_dict = return_dict

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()
        downsample_layers = list()
        bev_sa_blocks = list()  # per-stage BEV self-attn stacks

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewSwapAttention(
                feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap
            )
            cross_views.append(cva)

            # lightweight conv refinement (ResNet bottlenecks)
            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)

            # BEV self-attention blocks for global reasoning at each stage
            sa_stack = nn.Sequential(*[
                BEVTransformerBlock(dim=dim[i], window=(25, 25), heads=max(4, dim[i] // 32),
                                    dim_head=32, mlp_ratio=2.0, dropout=bev_sa_dropout)
                for _ in range(bev_sa_per_stage)
            ])
            bev_sa_blocks.append(sa_stack)

            # Downsample BEV feature to next stage resolution (channel align)
            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i] // 2, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(dim[i + 1], dim[i + 1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim[i + 1], dim[i + 1], 1, padding=0, bias=False),
                    nn.BatchNorm2d(dim[i + 1])
                ))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.bev_sa_blocks = nn.ModuleList(bev_sa_blocks)

        # Optional final BEV self-attention at the deepest level
        self.use_final_self_attn = self_attn is not None and len(self_attn) > 0
        if self.use_final_self_attn:
            d_last = dim[-1]
            self.final_self_attn = Attention(d_last, **self_attn)
        else:
            self.final_self_attn = None

    def forward(self, batch):
        """
        Returns:
          if return_dict:
            {
              'bev': final_bev,            # (b, C, H, W)
              'pyramid': [bev_s0, ...],    # list of per-stage BEV (high->low)
              'backbone_feats': features   # list of backbone multi-scale (for advanced decoders)
            }
          else:
            final_bev
        """
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)   # (b*n, c, h, w)
        I_inv = batch['intrinsics'].inverse()  # (b, n, 3, 3)
        E_inv = batch['extrinsics'].inverse()  # (b, n, 4, 4)

        object_count = batch.get('object_count', None)  # optional

        # backbone multi-scale features
        features = [self.down(y) for y in self.backbone(self.norm(image))]

        # seed BEV prior
        x = self.bev_embedding.get_prior()     # (d, H, W)
        x = repeat(x, '... -> b ...', b=b)     # (b, d, H, W)

        # collect pyramid for decoder
        bev_pyramid = []

        for i, (cross_view, feature, layer, sa_stack) in enumerate(
            zip(self.cross_views, features, self.layers, self.bev_sa_blocks)
        ):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            # camera-to-BEV cross-view
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)

            # conv refinement
            x = layer(x)

            # BEV self-attention (global reasoning)
            x = sa_stack(x)

            # save this stage output for decoder skip
            bev_pyramid.append(x)

            # downsample for next stage
            if i < len(features) - 1:
                x = self.downsample_layers[i](x)

        # optional final self-attention at deepest level
        if self.final_self_attn is not None:
            x = self.final_self_attn(x)

        if self.return_dict:
            return {
                'bev': x,
                'pyramid': bev_pyramid,       # high -> low
                'backbone_feats': [rearrange(f, '(b n) ... -> b n ...', b=b, n=n) for f in features],
            }
        else:
            return x


# ---------------------------
# Quick sanity for module
# ---------------------------

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

    # Minimal check for CrossWinAttention shapes
    block = CrossWinAttention(dim=128, heads=4, dim_head=32, qkv_bias=True)
    block.cuda()
    test_q = torch.rand(1, 6, 5, 5, 5, 5, 128).cuda()
    test_k = test_v = torch.rand(1, 6, 5, 5, 6, 12, 128).cuda()
    output = block(test_q, test_k, test_v)
    print('CrossWinAttention out:', output.shape)

    # Dummy inputs
    image = torch.rand(1, 6, 128, 28, 60)   # (b, n, c, h, w)
    I_inv = torch.rand(1, 6, 3, 3)
    E_inv = torch.rand(1, 6, 4, 4)

    feature = torch.rand(1, 6, 128, 25, 25)
    x = torch.rand(1, 128, 25, 25)

    # Example config load (if needed)
    # params = load_yaml('config/model/cvt_pyramid_swap.yaml')
    # print(params)

    # NOTE: You need to instantiate `backbone` with .output_shapes available and callable.
    # And then create encoder like:
    # encoder = PyramidAxialEncoder(backbone, cross_view=..., cross_view_swap=..., bev_embedding=..., self_attn=..., dim=..., middle=..., return_dict=True)
    # batch = {'image': image, 'intrinsics': I_inv, 'extrinsics': E_inv}
    # out = encoder(batch)
    # print(out['bev'].shape, len(out['pyramid']))
