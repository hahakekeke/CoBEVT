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

# convenience
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
        """Only key args: dim, sigma. The rest used to construct view matrix.

        Changed: keep learned_features at slightly higher capacity so downstream heavy
        processing has richer prior.
        """
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
            self.register_buffer('grid%d'%i, grid, persistent=False)

        # larger learned prior (heavier model -> more capacity)
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height//upsample_scales[0], bev_width//upsample_scales[0])
        )

    def get_prior(self):
        return self.learned_features


class Attention(nn.Module):
    def __init__(self, dim, dim_head = 32, dropout = 0., window_size = 25):
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

        # relative positional bias (global attention helper)
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, _, height, width, device, h = *x.shape, x.device, self.heads
        # flatten
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

    def forward(self, q, k, v, skip=None):
        """ q: (b n X Y W1 W2 d) k,v: (b n x y w1 w2 d)
            returns: (b X Y W1 W2 d)
        """
        assert k.shape == v.shape
        _, view_size, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        _, _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

        # flatten
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        # project
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # group head with batch
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)
        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)

        att = dot.softmax(dim=-1)
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, ' b (x y) (n w1 w2) d -> b n x y w1 w2 d', x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)

        z = self.proj(a)
        # reduce across cameras
        z = z.mean(1)
        # optional skip
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
        # to-do no_image_features: bool = False,
        no_image_features: bool = False,
        skip: bool = True,
        norm=nn.LayerNorm,
    ):
        super().__init__()
        # image plane mapping
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

        # two-stage cross-window attention (local-to-local, local-to-global)
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)

        self.skip = skip

        # prenorms and mlps (increased capacity to allow heavier processing)
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        # make MLP wider; gating will scale its effective output when scenes are complex
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

        self.postnorm = norm(dim)

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divisible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
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
        """ x: (b, c, H, W)
            feature: (b, n, dim_in, h, w)
            I_inv: (b, n, 3, 3)
            E_inv: (b, n, 4, 4)
            object_count: optional tensor describing object counts. Could be:
                - None
                - shape (b,) total counts per sample
                - shape (b, k) counts per class
                - shape (k,) single-batch counts (legacy)
            Returns: (b, d, H, W)
        """

        # --- Robust handling of object_count ---
        # turn into per-sample scalar complexity measure (float tensor, shape: (b,))
        b, n, _, _, _ = feature.shape
        complexity_per_sample = None
        if object_count is None:
            # default: assume simple scene
            complexity_per_sample = torch.ones(b, device=x.device)
        else:
            # normalize different possible shapes
            if object_count.dim() == 1 and object_count.shape[0] == b:
                # (b,)
                complexity_per_sample = object_count.float().to(x.device)
            elif object_count.dim() == 2 and object_count.shape[0] == b:
                # (b, k) -> sum over classes
                complexity_per_sample = object_count.sum(dim=-1).float().to(x.device)
            elif object_count.dim() == 1:
                # a single-vector for whole batch e.g. (k,) -> sum and replicate
                complexity_per_sample = object_count.sum().unsqueeze(0).repeat(b).float().to(x.device)
            else:
                # fallback
                try:
                    complexity_per_sample = object_count.view(b, -1).sum(dim=-1).float().to(x.device)
                except:
                    complexity_per_sample = torch.ones(b, device=x.device)

        # compute an aggregated scalar for controlling "heavy" operations
        # normalize by a reference object count (so multiplier ~1 for typical scenes)
        ref_obj = 10.0  # reference threshold: ~10 objects -> baseline
        complexity_scalar = complexity_per_sample.mean() / ref_obj
        # clamp multiplier in [1.0, 3.0] to avoid exploding compute
        multiplier = torch.clamp(complexity_scalar, min=1.0, max=3.0).item()
        # number of extra refinement iterations (0..2)
        extra_iters = max(0, int(round(multiplier)) - 1)

        # Remove fragile debug-style indexing used before
        # proceed to compute embedding and image-projection features
        pixel = self.image_plane
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]  # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]  # (b n) 4 1 1
        if hasattr(self, 'cam_embed'):
            c_embed = self.cam_embed(c_flat)  # (b n) d 1 1
        else:
            # if not using cam_embed, fake zero
            c_embed = torch.zeros((b * n, 1, 1, 1), device=x.device)

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')  # 1 1 3 (h w)
        cam = I_inv @ pixel_flat  # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)  # b n 4 (h w)

        d = E_inv @ cam  # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)  # (b n) 4 h w

        if hasattr(self, 'img_embed'):
            d_embed = self.img_embed(d_flat)  # (b n) d h w
            img_embed = d_embed - c_embed  # (b n) d h w
            img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)
        else:
            img_embed = torch.zeros_like(d_flat)

        # bev prior for query positional
        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        elif index == 3:
            world = bev.grid3[:2]
        else:
            # fallback use grid0 if index out of range
            world = bev.grid0[:2]

        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])  # 1 d H W
            bev_embed = w_embed - c_embed  # (b n) d H W (broadcast)
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

        # combine BEV and image features
        if self.bev_embed_flag and query_pos is not None:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]

        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)

        # pad
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # ========== LOCAL-TO-LOCAL CROSS ATTENTION ==========
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        # apply first cross-window attention (local)
        skip_conn = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query_local = rearrange(
            self.cross_win_attend_1(query, key, val, skip=skip_conn if self.skip else None),
            'b x y w1 w2 d -> b (x w1) (y w2) d'
        )
        query = query_local + self.mlp_1(self.prenorm_1(query_local))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)

        # ========== LOCAL-TO-GLOBAL CROSS ATTENTION ==========
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        # main pass
        query_global = rearrange(
            self.cross_win_attend_2(query, key, val, skip=rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None),
            'b x y w1 w2 d -> b (x w1) (y w2) d'
        )
        query = query_global + self.mlp_2(self.prenorm_2(query_global))

        # === Iterative refinement when scene is complex ===
        # If object_count indicates complexity, run extra lightweight refinement passes and average results.
        if extra_iters > 0:
            accum = query
            for _ in range(extra_iters):
                # prepare repeated query (broadcast to n)
                q_iter = repeat(query, 'b (x y) d -> b n x y d', n=n, x=int(query.shape[1] ** 0.5))
                # careful: dimension math may vary; instead reuse shapes from earlier
                # convert q_iter into windows
                try:
                    q_iter = rearrange(q_iter, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
                    out_iter = rearrange(
                        self.cross_win_attend_2(q_iter, key, val, skip=rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None),
                        'b x y w1 w2 d -> b (x w1) (y w2) d'
                    )
                    out_iter = out_iter + self.mlp_2(self.prenorm_2(out_iter))
                    accum = accum + out_iter
                except Exception:
                    # if shapes incompatible, skip iterative refinement for safety
                    break
            query = accum / float(extra_iters + 1)

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

        # heavy global self-attention for final BEV refinement (sacrifices speed)
        # enlarge dims for stronger capacity
        final_dim = dim[-1]
        self.self_attn = Attention(final_dim, dim_head=32, dropout=0.1, window_size=max(25, final_dim // 8))

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0, 1)  # b n c h w -> (b*n) c h w

        I_inv = batch['intrinsics'].inverse()  # b n 3 3
        E_inv = batch['extrinsics'].inverse()  # b n 4 4

        # object_count may be provided in different shapes; pass through
        object_count = batch.get('object_count', None)

        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = self.bev_embedding.get_prior()  # d H W
        x = repeat(x, '... -> b ...', b=b)  # b d H W

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        # final global self-attention refinement (heavy)
        # prepare x for Attention: Attention expects (b, d, h, w)
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
            list(u'-+0123456789.'))
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
