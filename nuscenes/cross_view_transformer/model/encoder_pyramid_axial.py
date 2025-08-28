# encoder_pyramid_axial.py
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
    """ copied from ..data.common but want to keep models standalone """
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
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()  # 3 3
        for i, scale in enumerate(upsample_scales):
            h = bev_height // scale
            w = bev_width // scale
            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]
            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')  # 3 (h w)
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)  # 3 h w
            self.register_buffer('grid%d' % i, grid, persistent=False)  # 3 h w
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height // upsample_scales[0], bev_width // upsample_scales[0])
        )  # d h w

    def get_prior(self):
        return self.learned_features


class Attention(nn.Module):
    def __init__(
        self, dim, dim_head=32, dropout=0., window_size=25
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
        batch, _, height, width, device, h = *x.shape, x.device, self.heads
        # flatten
        x = rearrange(x, 'b d h w -> b (h w) d')
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h), (q, k, v))
        # scale
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

    def forward(self, q, k, v, skip=None, attention_scale_factor: Optional[torch.Tensor] = None):
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
        q = self.to_q(q)  # b (X Y) (n W1 W2) (heads dim_head)
        k = self.to_k(k)
        v = self.to_v(v)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)

        # optionally scale the attention logits (sharpening) using attention_scale_factor (scalar or tensor per-batch)
        if attention_scale_factor is not None:
            # attention_scale_factor expected shape: (batch,) or scalar -> we must expand to match (b*m, ...)
            m = self.heads
            scale_val = attention_scale_factor.view(-1, 1)  # (b,1)
            scale_expanded = repeat(scale_val, 'b 1 -> (b m) 1', m=m)  # (b*m,1)
            dot = dot * scale_expanded.unsqueeze(-1)

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        att = dot.softmax(dim=-1)
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, ' b (x y) (n w1 w2) d -> b n x y w1 w2 d', x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)
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
        # keep this parameter because config may pass it
        no_image_features: bool = False,
        # to-do no_image_features: bool = False,
        skip: bool = True,
        norm=nn.LayerNorm,
        # performance options
        max_global_mlp_mult: int = 3,
        global_self_attn_threshold: int = 12,  # if average objects > threshold, run global self attention
    ):
        super().__init__()

        # configuration
        self.max_global_mlp_mult = max_global_mlp_mult
        self.global_self_attn_threshold = global_self_attn_threshold

        # image plane (pixel coords)
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)

        # feature projections
        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False)
        )

        # allow config to disable projecting image features from feature tensor
        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False)
            )

        # bev / img / cam embedding flags
        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
            self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
            self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)
        else:
            self.bev_embed = None
            self.img_embed = None
            self.cam_embed = None

        # window sizes & pos emb
        self.q_win_size = q_win_size[index]
        self.feat_win_size = feat_win_size[index]
        self.rel_pos_emb = rel_pos_emb

        # cross-window attention blocks
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)

        # additional global self attention (applied to BEV features when scene complex)
        ws_list = self.q_win_size if isinstance(self.q_win_size, (list, tuple)) else [self.q_win_size]
        window_size_for_global = max(ws_list + [25])
        self.global_self_attn = Attention(dim=dim, dim_head=dim_head[index], dropout=0., window_size=window_size_for_global)

        self.skip = skip

        # prenorms and MLPs — increase hidden dims for more capacity (use max multiplier)
        hidden_mult = 2 * self.max_global_mlp_mult
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(
            nn.Linear(dim, int(hidden_mult * dim)),
            nn.GELU(),
            nn.Linear(int(hidden_mult * dim), dim)
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(dim, int(hidden_mult * dim)),
            nn.GELU(),
            nn.Linear(int(hidden_mult * dim), dim)
        )
        self.postnorm = norm(dim)

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divisible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def _compute_attention_scale(self, object_count: Optional[torch.Tensor], b: int, n: int, device):
        """
        Compute a per-batch scalar that sharpens attention when scene is complex.
        Returns tensor of shape (b,) with >=1 scalars.
        """
        if object_count is None:
            return torch.ones(b, device=device, dtype=torch.float32)

        try:
            oc = object_count
            oc = oc.detach().to(device)
            if oc.dim() == 1:
                totals = oc  # (b,)
            elif oc.dim() == 2:
                totals = oc.sum(dim=-1)
            elif oc.dim() > 2:
                totals = oc.view(b, -1).sum(dim=-1)
            else:
                totals = oc.sum(dim=-1)
            s = (totals.float() / max(1.0, totals.mean().clamp(min=1.0))).clamp(min=0.0)
            scalars = 1.0 + (s.sqrt() * (self.max_global_mlp_mult - 1.0))
            scalars = scalars.clamp(min=1.0, max=float(self.max_global_mlp_mult))
            return scalars
        except Exception:
            return torch.ones(b, device=device, dtype=torch.float32)

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
        object_count: optional info used to scale attention
        Returns: (b, d, H, W)
        """

        if object_count is not None:
            try:
                print(">> object_count(crossviewswapattention):", object_count.shape, object_count)
            except Exception:
                pass

        b, n, _, _, _ = feature.shape
        _, _, H, W = x.shape
        pixel = self.image_plane  # 1 1 3 h w
        _, _, _, h, w = pixel.shape

        # camera center c from E_inv
        c = E_inv[..., -1:]  # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]  # (b n) 4 1 1
        c_embed = None
        if self.bev_embed is not None:
            c_embed = self.cam_embed(c_flat)  # (b n) d 1 1

        # compute image->ego coords
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat  # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)  # b n 4 (h w)
        d = E_inv @ cam  # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)  # (b n) 4 h w
        d_embed = self.img_embed(d_flat) if self.img_embed is not None else None  # (b n) d h w

        if self.bev_embed is not None and d_embed is not None and c_embed is not None:
            img_embed = d_embed - c_embed
            img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)
        else:
            img_embed = None

        # get world/bev grid for index
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

        # BEV embedding computation
        if self.bev_embed is not None:
            w_embed = self.bev_embed(world[None])  # 1 d H W
            bev_embed = w_embed - (c_embed if c_embed is not None else 0)
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)
        else:
            query_pos = None

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')

        # build key_flat with defensive handling if feature_proj or img_embed are missing
        if self.feature_proj is not None:
            if img_embed is not None:
                key_flat = img_embed + self.feature_proj(feature_flat)
            else:
                key_flat = self.feature_proj(feature_flat)
        else:
            if img_embed is not None:
                key_flat = img_embed
            else:
                # fallback: use raw projected features via feature_linear if nothing else
                key_fallback = self.feature_linear(feature_flat)
                key_flat = key_fallback

        val_flat = self.feature_linear(feature_flat)

        # dynamic attention scaling based on object_count
        attention_scale = self._compute_attention_scale(object_count, b=b, n=n, device=x.device)  # (b,)

        # Expand + refine the BEV embedding
        if query_pos is not None:
            query = query_pos + x[:, None]  # b n d H W
        else:
            query = x[:, None]

        # reshape key/val back to (b n d h w)
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)

        # pad divisible
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # ----- Add global summary to val to boost long-range interactions -----
        try:
            global_k = reduce(key, 'b n d h w -> b n d 1 1', 'mean')
            global_v = reduce(val, 'b n d h w -> b n d 1 1', 'mean')
            global_scale = attention_scale.mean().clamp(min=1.0).view(b, 1, 1, 1, 1)
            val = val + (global_v * (global_scale))
            key = key + (global_k * (global_scale))
        except Exception:
            pass

        # local-to-local cross-attention (windowed)
        query_windows = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_windows = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_windows = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        query_out = rearrange(
            self.cross_win_attend_1(
                query_windows,
                key_windows,
                val_windows,
                skip=rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None,
                attention_scale_factor=attention_scale
            ),
            'b x y w1 w2 d -> b (x w1) (y w2) d'
        )

        # residual + MLP
        query_out = query_out + self.mlp_1(self.prenorm_1(query_out))
        x_skip = query_out
        query_out = repeat(query_out, 'b x y d -> b n x y d', n=n)

        # local-to-global cross-attention (flip partitions)
        query_windows_2 = rearrange(query_out, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_grid = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key_grid = rearrange(key_grid, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_grid = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val_grid = rearrange(val_grid, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        query_out_2 = rearrange(
            self.cross_win_attend_2(
                query_windows_2,
                key_grid,
                val_grid,
                skip=rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None,
                attention_scale_factor=attention_scale
            ),
            'b x y w1 w2 d -> b (x w1) (y w2) d'
        )

        query_out_2 = query_out_2 + self.mlp_2(self.prenorm_2(query_out_2))
        query_out_2 = self.postnorm(query_out_2)
        query_out_2 = rearrange(query_out_2, 'b H W d -> b d H W')

        # Optionally run a global self-attention on BEV when average object count large
        try:
            avg_scalar = 0.0
            if object_count is not None:
                if object_count.dim() == 1:
                    avg_objects = object_count
                else:
                    avg_objects = object_count.view(b, -1).sum(dim=-1) / max(1, object_count.view(b, -1).shape[-1])
                avg_scalar = float(avg_objects.mean().item())
            if avg_scalar >= float(self.global_self_attn_threshold):
                global_refined = self.global_self_attn(query_out_2)
                query_out_2 = query_out_2 + global_refined
        except Exception:
            pass

        return query_out_2


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
        # self.self_attn = Attention(dim[-1], **self_attn)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()

        # object_count (optional)
        object_count = batch.get('object_count', None)
        if object_count is not None:
            try:
                print(">> object_count(pyramid axial encoder):", object_count.shape, object_count)
            except Exception:
                pass

        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = self.bev_embedding.get_prior()  # d H W
        x = repeat(x, '... -> b ...', b=b)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features) - 1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)
        # x = self.self_attn(x)
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
            re.compile(u'''^(?: [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)? |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+) |\\.[0-9_]+(?:[eE][-+][0-9]+)? |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]* |[-+]?\\.(?:inf|Inf|INF) |\\.(?:nan|NaN|NAN))$''', re.X), list(u'-+0123456789.'))
        param = yaml.load(stream, Loader=loader)
        if "yaml_parser" in param:
            param = eval(param["yaml_parser"])(param)
        return param

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    block = CrossWinAttention(dim=128, heads=4, dim_head=32, qkv_bias=True,)
    block.cuda()
    test_q = torch.rand(1, 6, 5, 5, 5, 5, 128)
    test_k = test_v = torch.rand(1, 6, 5, 5, 6, 12, 128)
    test_q = test_q.cuda()
    test_k = test_k.cuda()
    test_v = test_v.cuda()
    output = block(test_q, test_k, test_v)
    print(output.shape)
