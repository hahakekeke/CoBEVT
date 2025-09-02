# encoder_pyramid_axial.py (수정본)
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
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)             # 3 h w
    indices = indices[None]                                           # 1 3 h w
    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
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
        super().__init__()
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
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
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height // upsample_scales[0], bev_width // upsample_scales[0])
        )

    def get_prior(self):
        return self.learned_features


class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0., window_size=25):
        super().__init__()
        assert (dim % dim_head) == 0
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        b, d, H, W = x.shape
        h = self.heads
        x_flat = rearrange(x, 'b d h w -> b (h w) d')
        q, k, v = self.to_qkv(x_flat).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b m (h w) d -> b h w (m d)', h=H, w=W)
        out = self.to_out(out)
        return rearrange(out, 'b h w d -> b d h w')


class CrossWinAttention(nn.Module):
    """
    Robust CrossWinAttention implementation:
      - q: (b, n, X, Y, qWh, qWw, d)
      - k, v: (b, n, x, y, kWh, kWw, d)  (assert X*Y == x*y)
      - returns: (b, X, Y, qWh, qWw, d)  (averaged over cameras)
    """
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.proj = nn.Linear(heads * dim_head, dim)
        self.dropout = nn.Dropout(dropout)
        self.rel_pos_emb = rel_pos_emb

    def forward(self, q, k, v, skip=None):
        # q: b n X Y qWh qWw d
        b, n, X, Y, qWh, qWw, d = q.shape
        _, _, x, y, kWh, kWw, _ = k.shape
        assert X * Y == x * y, "q windows should match k windows count"

        # Flatten per-window groups:
        # q_flat: (b, L, Mq, d) where L = X*Y, Mq = n * qWh * qWw
        L = X * Y
        Mq = n * qWh * qWw
        Mk = n * kWh * kWw

        q_flat = rearrange(q, 'b n x y wh ww d -> b (x y) (n wh ww) d')
        k_flat = rearrange(k, 'b n x y wh ww d -> b (x y) (n wh ww) d')
        v_flat = rearrange(v, 'b n x y wh ww d -> b (x y) (n wh ww) d')

        # Project
        q_proj = self.to_q(q_flat)  # b L Mq (heads*dim_head)
        k_proj = self.to_k(k_flat)
        v_proj = self.to_v(v_flat)

        # reshape to separate heads: (b, L, M, heads, dim_head) -> (b*heads, L, M, dim_head)
        def to_heads(t):
            b_, L_, M_, hd = t.shape
            t = t.view(b_, L_, M_, self.heads, self.dim_head)
            t = t.permute(0, 3, 1, 2, 4).contiguous()  # b heads L M dim_head
            return t.view(b_ * self.heads, L_, M_, self.dim_head)

        qh = to_heads(q_proj)
        kh = to_heads(k_proj)
        vh = to_heads(v_proj)

        # scaled dot-product per-head
        # qh: (B', L, Mq, d), kh: (B', L, Mk, d)
        attn_logits = torch.einsum('B l q d, B l k d -> B l q k', qh * self.scale, kh)

        if self.rel_pos_emb:
            # keep placeholder (no-op) or implement relative pos here if needed
            pass

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # attn @ v  -> (B', L, Mq, d)
        a_heads = torch.einsum('B l q k, B l k d -> B l q d', attn, vh)

        # merge heads back: (b*heads, L, Mq, dim_head) -> (b, L, Mq, heads*dim_head)
        a_heads = a_heads.view(b, self.heads, L, Mq, self.dim_head)
        a_comb = a_heads.permute(0, 2, 3, 1, 4).contiguous().view(b, L, Mq, self.heads * self.dim_head)

        # linear projection
        z = self.proj(a_comb)  # (b, L, Mq, d)

        # reshape to (b, L, n, qWh, qWw, d) and then to (b, n, X, Y, qWh, qWw, d)
        z = z.view(b, L, n, qWh, qWw, d)
        z = rearrange(z, 'b (x y) n wh ww d -> b n x y wh ww d', x=X, y=Y)

        # average across cameras to match skip shape (skip has no camera dim)
        z_mean = z.mean(dim=1)  # (b, X, Y, qWh, qWw, d)

        # optional skip add: skip expected shape (b, X, Y, qWh, qWw, d)
        if skip is not None:
            # ensure skip and z_mean shapes match exactly
            if skip.shape != z_mean.shape:
                # try permuting if skip in slightly different order
                # (original code uses skip = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d') )
                # z_mean shape: (b, X, Y, qWh, qWw, d)
                # if skip is (b, x, y, w1, w2, d) it's compatible
                raise RuntimeError(f"CrossWinAttention skip shape {skip.shape} mismatch with computed attention {z_mean.shape}")
            z_mean = z_mean + skip

        return z_mean  # shape (b, X, Y, qWh, qWw, d)


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

        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.q_win_size = q_win_size[index]
        self.feat_win_size = feat_win_size[index]
        self.rel_pos_emb = rel_pos_emb

        # use our robust CrossWinAttention
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
        h_pad = ((h + win_h - 1) // win_h) * win_h
        w_pad = ((w + win_w - 1) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
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
        # debug prints preserved
        if object_count is not None:
            print(">> object_count(crossviewswapattention):", object_count.shape, object_count)
        else:
            print(">> object_count(crossviewswapattention) is None")

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

        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        else:
            world = bev.grid3[:2]

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

        # local-to-local
        q1 = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                       w1=self.q_win_size[0], w2=self.q_win_size[1])
        k1 = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        v1 = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        skip1 = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

        out1 = self.cross_win_attend_1(q1, k1, v1, skip=skip1)  # (b, X, Y, w1, w2, d)
        out1 = rearrange(out1, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        out1 = out1 + self.mlp_1(self.prenorm_1(out1))

        x_skip = out1
        out1_rep = repeat(out1, 'b x y d -> b n x y d', n=n)

        # local-to-global
        q2 = rearrange(out1_rep, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                       w1=self.q_win_size[0], w2=self.q_win_size[1])
        k2 = rearrange(k1, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        k2 = rearrange(k2, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        v2 = rearrange(v1, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        v2 = rearrange(v2, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        skip2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

        out2 = self.cross_win_attend_2(q2, k2, v2, skip=skip2)  # (b, X, Y, w1, w2, d)
        out2 = rearrange(out2, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        out2 = out2 + self.mlp_2(self.prenorm_2(out2))
        out2 = self.postnorm(out2)

        out = rearrange(out2, 'b H W d -> b d H W')

        if self.skip:
            out = out + x * 0.0  # keep residual path simple (scale 0 if not desired)

        return out


class BEVTransformerBlock(nn.Module):
    def __init__(self, dim, window=(25, 25), heads=8, dim_head=32, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.win_h, self.win_w = window
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, dim_head=dim_head, dropout=dropout, window_size=max(self.win_h, self.win_w))
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        # x: (b, d, H, W)
        b, d, H, W = x.shape
        pad_h = (self.win_h - H % self.win_h) % self.win_h
        pad_w = (self.win_w - W % self.win_w) % self.win_w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            Hp, Wp = x.shape[-2], x.shape[-1]
        else:
            Hp, Wp = H, W
        # partition windows
        x_win = rearrange(x, 'b d (nh wh) (nw ww) -> (b nh nw) d wh ww', wh=self.win_h, ww=self.win_w)
        # apply attention per window (Attention expects b d h w)
        x_win = self.attn(x_win)
        x_win = rearrange(x_win, '(b nh nw) d wh ww -> b d (nh wh) (nw ww)', b=b, nh=Hp//self.win_h, nw=Wp//self.win_w, wh=self.win_h, ww=self.win_w)
        if pad_h or pad_w:
            x_win = x_win[:, :, :H, :W]
        y = rearrange(x_win, 'b d h w -> b (h w) d')
        y = self.norm2(y)
        y = self.mlp(y)
        y = rearrange(y, 'b (h w) d -> b d h w', h=x.shape[-2], w=x.shape[-1])
        return x + y


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
        return_dict: bool = True,
        bev_sa_per_stage: int = 1,
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
        bev_sa_blocks = list()
        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)
            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)
            sa_stack = nn.Sequential(*[
                BEVTransformerBlock(dim=dim[i], window=(25, 25), heads=max(4, dim[i] // 32),
                                    dim_head=32, mlp_ratio=2.0, dropout=bev_sa_dropout)
                for _ in range(bev_sa_per_stage)
            ])
            bev_sa_blocks.append(sa_stack)
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

        self.use_final_self_attn = self_attn is not None and len(self_attn) > 0
        if self.use_final_self_attn:
            d_last = dim[-1]
            self.final_self_attn = Attention(d_last, **self_attn)
        else:
            self.final_self_attn = None

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        object_count = batch.get('object_count', None)
        if object_count is not None:
            print(">> object_count(pyramid axial encoder):", object_count.shape, object_count)
        else:
            print(">> object_count(pyramid axial encoder) is None")
        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)
        bev_pyramid = []
        for i, (cross_view, feature, layer, sa_stack) in enumerate(
            zip(self.cross_views, features, self.layers, self.bev_sa_blocks)
        ):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            x = sa_stack(x)
            bev_pyramid.append(x)
            if i < len(features) - 1:
                x = self.downsample_layers[i](x)
        if self.final_self_attn is not None:
            x = self.final_self_attn(x)
        if self.return_dict:
            return {
                'bev': x,
                'pyramid': bev_pyramid,
                'backbone_feats': [rearrange(f, '(b n) ... -> b n ...', b=b, n=n) for f in features],
            }
        else:
            return x



if __name__ == "__main__":
    # 간단 shape 체크 (필요 시 사용)
    pass
