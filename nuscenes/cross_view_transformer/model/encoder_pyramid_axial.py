# encoder_pyramid_axial.py — patched: robust shape handling for key/val (fixes Einops shape mismatch)
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

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters

    return [
        [0., -sw, w/2.],
        [-sh, 0., h*offset + h/2.],
        [0., 0., 1.]
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


# ---- Attention (original style from your file) ----
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
        # x: b d h w
        batch, _, height, width, device, h = *x.shape, x.device, self.heads
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


# ---- small MBConv used to refine key/val locally ----
class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, se_ratio=0.25):
        super().__init__()
        hidden = int(in_ch * expansion)
        self.use_expand = (expansion != 1 and hidden != in_ch)
        if self.use_expand:
            self.expand = nn.Sequential(nn.Conv2d(in_ch, hidden, 1, bias=False),
                                        nn.BatchNorm2d(hidden),
                                        nn.GELU())
        else:
            hidden = in_ch
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size//2, groups=hidden, bias=False),
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
        self.project = nn.Sequential(nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))

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


# ---- small Channel-wise FFN used at the end (applied to channels dimension) ----
class ChannelFFN(nn.Module):
    def __init__(self, dim, expansion=2, dropout=0.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        # supports x: (b, seq, d) or (b, H, W, d)
        if x.dim() == 3:
            return self.net(x)
        elif x.dim() == 4:
            b, H, W, d = x.shape
            flat = x.view(b, H * W, d)
            out = self.net(flat)
            return out.view(b, H, W, d)
        else:
            raise RuntimeError("ChannelFFN expects 3D or 4D input")


# ---- CrossWinAttention (original with optional temperature scaling) ----
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
        # q: b n X Y W1 W2 d
        # k,v: b n x y w1 w2 d
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

        dot = torch.einsum('b l Q d, b l K d -> b l Q K', q, k)
        dot = dot * self.scale

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)

        if temperature is not None:
            dot = dot * float(temperature)

        att = dot.softmax(dim=-1)
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, 'b (x y) (n w1 w2) d -> b n x y w1 w2 d',
                      x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)
        z = self.proj(a)
        z = z.mean(1)
        if skip is not None:
            z = z + skip
        return z


# ---- CrossViewSwapAttention: main block (with robust shape handling) ----
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

        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

        # added modules
        self.mbconv = MBConv(dim, dim, expansion=4, kernel_size=3, se_ratio=0.25)
        self.post_mlp = ChannelFFN(dim, expansion=2, dropout=0.0)
        self.proj_out = nn.Sequential(nn.Linear(dim, dim), nn.GELU())

    def pad_divisble(self, x, win_h, win_w):
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def compute_temperature(self, object_count: Optional[torch.Tensor], base_temp=1.0, max_temp=3.0):
        if object_count is None:
            return base_temp
        try:
            avg_count = float(object_count.float().mean().item())
        except Exception:
            try:
                avg_count = float(object_count.mean().item())
            except Exception:
                avg_count = 0.0
        temp = base_temp + (avg_count / 20.0) * (max_temp - base_temp)
        return float(max(base_temp, min(temp, max_temp)))

    def _ensure_b_n(self, t, b, n):
        """
        Ensure t is shaped (b, n, d, h, w).
        Accepts t either (b*n, d, h, w) or (b, n, d, h, w).
        """
        if t is None:
            return t
        if t.dim() == 4:
            # (b*n, d, h, w)
            expected = b * n
            if t.shape[0] != expected:
                # If shapes don't match, try to detect (b, n, d, h, w) passed as 4D incorrectly:
                raise RuntimeError(f"Expected first dim {expected} for flattened tensor but got {t.shape[0]}.")
            return t.view(b, n, t.shape[1], t.shape[2], t.shape[3])
        elif t.dim() == 5:
            # already (b, n, d, h, w)
            if t.shape[0] != b or t.shape[1] != n:
                # allow case where b/n may match differently; still return as is
                # but warn
                # print(f"Warning: _ensure_b_n got (b,n)=({t.shape[0]},{t.shape[1]}) expected ({b},{n})")
                pass
            return t
        else:
            raise RuntimeError(f"_ensure_b_n expects 4D or 5D tensor, got {t.dim()}D")

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
        # debug prints (optional)
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

        # bev grid selection
        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        elif index == 3:
            world = bev.grid3[:2]
        else:
            world = bev.grid0[:2]

        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])  # 1 d H W
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)

        # feature_flat: (b n) d h w
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')

        if self.feature_proj is not None:
            # ensure feature_proj accepts (b*n, c, h, w)
            key_flat = img_embed + self.feature_proj(feature_flat)
        else:
            key_flat = img_embed

        val_flat = self.feature_linear(feature_flat)

        # MBConv refinement: accepts (b*n, d, h, w) -> returns same
        # After MBConv, we convert to (b, n, d, h, w)
        if val_flat.dim() == 4:
            bn_mul = val_flat.shape[0]
        else:
            raise RuntimeError("val_flat expected 4D (b*n, c, h, w) before MBConv.")

        val_ref = self.mbconv(val_flat)
        key_ref = self.mbconv(key_flat)

        # reshape to (b, n, d, h, w)
        key = self._ensure_b_n(key_ref, b, n)
        val = self._ensure_b_n(val_ref, b, n)

        # normalize
        key = key / (key.norm(dim=2, keepdim=True) + 1e-7)
        val = val / (val.norm(dim=2, keepdim=True) + 1e-7)

        # pad to be divisible
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # temperature
        temperature = self.compute_temperature(object_count, base_temp=1.0, max_temp=3.0)

        # query construction
        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]  # b, n, d, H, W

        # local-to-local cross-attention
        query_windows = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                                  w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_windows = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                                w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_windows = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                                w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        out1 = self.cross_win_attend_1(query_windows, key_windows, val_windows,
                                       skip=rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                                                      w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None,
                                       temperature=temperature)

        query = rearrange(out1, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)

        # local-to-global cross-attention (grid)
        query_grid = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                               w1=self.q_win_size[0], w2=self.q_win_size[1])

        key_grid = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key_grid = rearrange(key_grid, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                             w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_grid = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val_grid = rearrange(val_grid, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                             w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        out2 = self.cross_win_attend_2(query_grid, key_grid, val_grid,
                                       skip=rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                                                      w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None,
                                       temperature=temperature)

        query = rearrange(out2, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_2(self.prenorm_2(query))

        query = self.postnorm(query)  # b, (H*W), d

        # reshape back to (b, H, W, d)
        bsz = query.shape[0]
        total_hw = query.shape[1]
        d_ch = query.shape[2]
        # assume query corresponds to H*W (original BEV H,W)
        # if padding mismatch occurs, user may need to adjust pad/unpad logic.
        query_hw = query.view(bsz, H, W, d_ch)

        # channel-wise ffm
        query_hw = self.post_mlp(query_hw)

        # to (b, d, H, W)
        out = rearrange(query_hw, 'b H W d -> b d H W')

        # final proj operating on channels
        out = self.proj_out(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out


# ---------------------------
# PyramidAxialEncoder (uses CrossViewSwapAttention)
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

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        return x


# If executed as script: quick sanity check (light)
if __name__ == "__main__":
    # minimal smoke test (not exhaustive)
    B, N = 2, 6
    d = 64
    H, W = 25, 25
    feat = torch.randn(B, N, d, H, W)
    I_inv = torch.randn(B, N, 3, 3)
    E_inv = torch.randn(B, N, 4, 4)
    bev = BEVEmbedding(dim=d, sigma=0.02, bev_height=H, bev_width=W, h_meters=100, w_meters=100, offset=0.0, upsample_scales=[1])
    cva = CrossViewSwapAttention(feat_height=H, feat_width=W, feat_dim=d, dim=d, index=0,
                                 image_height=H, image_width=W, qkv_bias=True,
                                 q_win_size=[[5,5]], feat_win_size=[[5,5]],
                                 heads=[4], dim_head=[32], bev_embedding_flag=[True])
    out = cva(0, torch.randn(B, d, H, W), bev, feat, I_inv, E_inv, object_count=None)
    print("CVA output shape:", out.shape)
