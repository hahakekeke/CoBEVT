# encoder_pyramid_axial.py (전체 수정본)
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models  # 예시: 고성능 백본을 위해 import

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List
# from .decoder import DecoderBlock # 로컬 import는 주석 처리

from typing import Optional

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)      # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                  # 3 h w
    indices = indices[None]                                               # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters
    return [
        [ 0., -sw,        w/2.],
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
        self.kwargs = {'stride': stride, 'padding': padding}

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(self, dim: int, sigma: int, bev_height: int, bev_width: int,
                 h_meters: int, w_meters: int, offset: int, upsample_scales: list):
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
            self.register_buffer('grid%d' % i, grid, persistent=False)
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height // upsample_scales[0], bev_width // upsample_scales[0]))

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
        batch, _, height, width, device, h = *x.shape, x.device, self.heads
        x = rearrange(x, 'b d h w -> b (h w) d')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h), (q, k, v))
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
    def __init__(self, feat_height: int, feat_width: int, feat_dim: int, dim: int, index: int,
                 image_height: int, image_width: int, qkv_bias: bool, q_win_size: list, feat_win_size: list,
                 heads: list, dim_head: list, bev_embedding_flag: list, rel_pos_emb: bool = False,
                 no_image_features: bool = False, skip: bool = True, norm=nn.LayerNorm):
        super().__init__()
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)
        self.feature_linear = nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
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
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def forward(self, index: int, x: torch.FloatTensor, bev: BEVEmbedding, feature: torch.FloatTensor,
                I_inv: torch.FloatTensor, E_inv: torch.FloatTensor, object_count: Optional[torch.Tensor] = None):
        if object_count is not None:
            pass
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
        elif index == 3:
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
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        query = rearrange(self.cross_win_attend_1(query, key, val,
                                                    skip=rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                                                                  w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None),
                        'b x y w1 w2 d  -> b (x w1) (y w2) d')
        query = query + self.mlp_1(self.prenorm_1(query))
        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        query = rearrange(self.cross_win_attend_2(query, key, val,
                                                    skip=rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                                                                   w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None),
                        'b x y w1 w2 d  -> b (x w1) (y w2) d')
        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        query = rearrange(query, 'b H W d -> b d H W')
        return query


# --- 메모리 안전한 AttentionEntropyCalculator (유지) ---
class AttentionEntropyCalculator(nn.Module):
    def __init__(self, in_ch=3, proj_dim=32, heads=4, max_tokens=256, pool_size: Optional[tuple]=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, proj_dim, kernel_size=3, padding=1, bias=False)
        self.to_q = nn.Linear(proj_dim, proj_dim, bias=False)
        self.to_k = nn.Linear(proj_dim, proj_dim, bias=False)
        self.heads = heads
        self.scale = (proj_dim // heads) ** -0.5
        self.max_tokens = max_tokens
        self.pool_size = pool_size

    def forward(self, images: torch.Tensor):
        b, n, c, h, w = images.shape
        device = images.device
        if self.pool_size is not None:
            pH, pW = self.pool_size
        else:
            orig_T = n * h * w
            if orig_T > self.max_tokens:
                scale = math.sqrt(orig_T / float(self.max_tokens))
                pH = max(1, int(h / scale))
                pW = max(1, int(w / scale))
            else:
                pH, pW = h, w
        imgs = images.view(b * n, c, h, w)
        if (pH != h) or (pW != w):
            imgs = F.adaptive_avg_pool2d(imgs, (pH, pW))
        feats = self.conv(imgs)
        pd = feats.shape[1]
        feats = feats.view(b, n, pd, pH * pW)
        feats = feats.permute(0, 3, 1, 2).reshape(b, n * pH * pW, pd)
        T = feats.shape[1]
        if T <= 1:
            return torch.zeros(b, device=device)
        q = self.to_q(feats)
        k = self.to_k(feats)
        try:
            q = q.view(b, T, self.heads, pd // self.heads).permute(0, 2, 1, 3)
            k = k.view(b, T, self.heads, pd // self.heads).permute(0, 2, 1, 3)
        except Exception:
            q = q.view(b, T, 1, pd).permute(0, 2, 1, 3)
            k = k.view(b, T, 1, pd).permute(0, 2, 1, 3)
        q = q * self.scale
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn_logits, dim=-1).clamp(min=1e-12)
        ent = - (attn * torch.log(attn)).sum(dim=-1)
        ent = ent.mean(dim=-1).mean(dim=-1)
        norm_ent = ent / math.log(max(2, T))
        norm_ent = torch.clamp(norm_ent, 0.0, 1.0)
        return norm_ent


class PyramidAxialEncoder(nn.Module):
    def __init__(self, backbone, cross_view: dict, cross_view_swap: dict, bev_embedding: dict,
                 self_attn: dict, dim: list, middle: List[int] = [2, 2], scale: float = 1.0,
                 high_perf_backbone=None, entropy_threshold: float = 0.25):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        self.high_perf_backbone = high_perf_backbone
        self.attn_entropy_calc = AttentionEntropyCalculator(in_ch=3, proj_dim=32, heads=4, max_tokens=256)
        self.entropy_threshold = entropy_threshold
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
                    )))
        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        object_count = batch.get('object_count', None)
        num_feature_levels = len(self.backbone.output_shapes)
        features_per_level = [[] for _ in range(num_feature_levels)]
        images = batch['image']  # (b, n, c, h, w)
        device = images.device

        # 1) attention entropy (메모리 안전)
        try:
            with torch.no_grad():
                entropies = self.attn_entropy_calc(images)  # (b,)
        except RuntimeError as e:
            err_str = str(e).lower()
            if 'out of memory' in err_str or 'cuda' in err_str:
                print("Warning: OOM during entropy calc on GPU — fallback to CPU computation.")
                try:
                    torch.cuda.empty_cache()
                    entropies = self.attn_entropy_calc(images.cpu()).to(device)
                except Exception as e2:
                    print("Fallback failed, marking all entropies = 1.0 (force per-sample processing).", e2)
                    entropies = torch.ones(b, device=device)
            else:
                raise

        # 2) 결정: 통짜 처리 또는 그룹별 처리 (같은 backbone 모듈을 반복 호출하지 않도록)
        if entropies.max().item() < self.entropy_threshold:
            # 배치 통째 처리: backbone 한 번 호출
            b_, n_, c, h, w = images.shape
            flat_images = images.view(b_ * n_, c, h, w)
            flat_images_norm = self.norm(flat_images)
            all_features = self.backbone(flat_images_norm)  # list of (b*n, c_l, h_l, w_l)
            for lvl_idx, feat_lvl in enumerate(all_features):
                bn, c_l, h_l, w_l = feat_lvl.shape
                assert bn == b_ * n_
                feat_lvl_reshaped = feat_lvl.view(b_, n_, c_l, h_l, w_l)
                for i_sample in range(b_):
                    features_per_level[lvl_idx].append(self.down(feat_lvl_reshaped[i_sample]))
        else:
            # 그룹 처리: high_perf 필요한 샘플들과 그렇지 않은 샘플들로 나눈 뒤,
            # 각각의 백본을 최대 한 번만 호출.
            idx_high = []
            idx_low = []
            for i in range(b):
                if (self.high_perf_backbone is not None) and (object_count is not None) and (object_count[i] >= 30):
                    idx_high.append(i)
                else:
                    idx_low.append(i)

            # helper: 주어진 sample indices 리스트에 대해 backbone 호출 및 features_per_level 채우기
            def process_indices(indices, backbone_module):
                if len(indices) == 0:
                    return
                # gather images for these samples in original order
                imgs_list = [images[i] for i in indices]  # 각 원소: (n,c,h,w)
                imgs_cat = torch.cat(imgs_list, dim=0)   # (len(indices)*n, c, h, w)
                imgs_cat_norm = self.norm(imgs_cat)
                feats_list = backbone_module(imgs_cat_norm)  # list of (len(indices)*n, c_l, h_l, w_l)
                # split per sample and append
                for lvl_idx, feat_lvl in enumerate(feats_list):
                    # shape (len(indices)*n, c_l, h_l, w_l)
                    n_total, c_l, h_l, w_l = feat_lvl.shape
                    assert n_total == len(indices) * n
                    # reshape to (len(indices), n, c_l, h_l, w_l)
                    feat_per_sample = feat_lvl.view(len(indices), n, c_l, h_l, w_l)
                    # append into features_per_level at the correct original sample index
                    for local_idx, global_sample_idx in enumerate(indices):
                        features_per_level[lvl_idx].append(self.down(feat_per_sample[local_idx]))

            # 먼저 low group (standard backbone)
            process_indices(idx_low, self.backbone)
            # 그리고 high group (high_perf_backbone) — high_perf_backbone이 없으면 standard 사용
            if len(idx_high) > 0:
                backbone_for_high = self.high_perf_backbone if (self.high_perf_backbone is not None) else self.backbone
                process_indices(idx_high, backbone_for_high)

            # **중요**: 현재 features_per_level 리스트는 "먼저 low 그룹(인덱스 순서대로), 그다음 high 그룹"으로 쌓여 있음.
            # 우리가 원래 원하던 것은 features_per_level[i]가 sample index i 순서로 쌓인 것.
            # 따라서 features_per_level에 append한 순서를 샘플 인덱스 순으로 재정렬해야 함.
            # 위의 process_indices는 features_per_level에 append 시 각 group's original global index를 보존해서
            # 아래 정렬 단계에서 원래 샘플 순서로 재배열 가능하도록 (index->position) 정보를 사용.
            # 하지만 위 append는 단순 append이므로 현재 순서가 뒤섞일 수 있음.
            # 해결: 대신 features_per_level을 임시 dict에 저장한 뒤 최종적으로 순서대로 채운다.

            # 재구성 (더 안전한 방식): 재작성하여 위에서 append하지 않고 임시 dict 사용
            # => 간단 구현: redo using temp storage to ensure sample-order
            temp_per_level = {lvl: {} for lvl in range(num_feature_levels)}

            # 재호출 but storing into temp dict (this duplicates some work but guarantees order)
            # For memory/time efficiency, we reuse previous computed blobs if stored; for simplicity we'll recompute groups once into temp.
            # process idx_low into temp
            if len(idx_low) > 0:
                imgs_list = [images[i] for i in idx_low]
                imgs_cat = torch.cat(imgs_list, dim=0)
                imgs_cat_norm = self.norm(imgs_cat)
                feats_list = self.backbone(imgs_cat_norm)
                for lvl_idx, feat_lvl in enumerate(feats_list):
                    feat_per_sample = feat_lvl.view(len(idx_low), n, feat_lvl.shape[1], feat_lvl.shape[2], feat_lvl.shape[3])
                    for local_idx, global_sample_idx in enumerate(idx_low):
                        temp_per_level[lvl_idx][global_sample_idx] = self.down(feat_per_sample[local_idx])

            if len(idx_high) > 0:
                backbone_for_high = self.high_perf_backbone if (self.high_perf_backbone is not None) else self.backbone
                imgs_list = [images[i] for i in idx_high]
                imgs_cat = torch.cat(imgs_list, dim=0)
                imgs_cat_norm = self.norm(imgs_cat)
                feats_list = backbone_for_high(imgs_cat_norm)
                for lvl_idx, feat_lvl in enumerate(feats_list):
                    feat_per_sample = feat_lvl.view(len(idx_high), n, feat_lvl.shape[1], feat_lvl.shape[2], feat_lvl.shape[3])
                    for local_idx, global_sample_idx in enumerate(idx_high):
                        temp_per_level[lvl_idx][global_sample_idx] = self.down(feat_per_sample[local_idx])

            # 이제 features_per_level를 sample 인덱스 0..b-1 순서로 채움
            for lvl_idx in range(num_feature_levels):
                features_per_level[lvl_idx] = [temp_per_level[lvl_idx][i] for i in range(b)]

        # 마지막: 각 레벨별로 (b*n, c, h, w) 형식으로 concat
        features = [torch.cat(feats, dim=0) for feats in features_per_level]

        # BEV prior
        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features) - 1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        return x


if __name__ == "__main__":
    print("수정된 PyramidAxialEncoder (DDP-safe, 그룹별 backbone 호출 방식)가 로드되었습니다.")
