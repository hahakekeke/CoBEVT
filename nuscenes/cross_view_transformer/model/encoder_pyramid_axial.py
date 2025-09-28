import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models # 예시: 고성능 백본을 위해 import

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

# from .decoder import DecoderBlock # 로컬 import는 주석 처리


ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)      # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                  # 3 h w
    indices = indices[None]                                               # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,      w/2.],
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

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
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
            self.register_buffer('grid%d'%i, grid, persistent=False)

            # 3 h w
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim,
                                bev_height//upsample_scales[0],
                                bev_width//upsample_scales[0]))  # d h w

    def get_prior(self):
        return self.learned_features


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 25
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

        # flatten

        x = rearrange(x, 'b d h w -> b (h w) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b m (h w) d -> b h w (m d)',
                        h = height, w = width)

        # combine heads out

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
        q = self.to_q(q)                                      # b (X Y) (n W1 W2) (heads dim_head)
        k = self.to_k(k)                                      # b (X Y) (n w1 w2) (heads dim_head)
        v = self.to_v(v)                                      # b (X Y) (n w1 w2) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)  # b (X Y) (n W1 W2) (n w1 w2)
        # dot = rearrange(dot, 'b l n Q K -> b l Q (n K)')  # b (X Y) (W1 W2) (n w1 w2)

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
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

        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip
        # self.proj = nn.Linear(2 * dim, dim)

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divible by window size."""
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
        object_count: Optional[torch.Tensor] = None, #object_count
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        
        # 참고: object_count는 PyramidAxialEncoder 레벨에서 처리되므로 여기서는 직접 사용하지 않습니다.
        # 필요하다면 디버깅 용도로 남겨둘 수 있습니다.
        if object_count is not None:
            # print(">> object_count(crossviewswapattention):", object_count.shape, object_count)
            pass
        
        b, n, _, _, _ = feature.shape
        _, _, H, W = x.shape

        pixel = self.image_plane                                                      # b n 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]                                                           # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                      # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                              # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                         # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                      # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                           # b n 4 (h w)
        d = E_inv @ cam                                                               # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)                 # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                                              # (b n) d h w

        img_embed = d_embed - c_embed                                                 # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)          # (b n) d h w

        # todo: some hard-code for now.
        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        elif index == 3:
            world = bev.grid3[:2]

        if self.bev_embed_flag:
            # 2 H W
            w_embed = self.bev_embed(world[None])                                     # 1 d H W
            bev_embed = w_embed - c_embed                                             # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)      # (b n) d H W
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)        # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')                     # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)                    # (b n) d h w
        else:
            key_flat = img_embed                                                      # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                                  # (b n) d h w

        # Expand + refine the BEV embedding
        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]  # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)                   # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)                   # b n d h w

        # pad divisible
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # local-to-local cross-attention
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition
        query = rearrange(self.cross_win_attend_1(query, key, val,
                                                skip=rearrange(x,
                                                               'b d (x w1) (y w2) -> b x y w1 w2 d',
                                                               w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None),
                          'b x y w1 w2 d  -> b (x w1) (y w2) d')   # reverse window to feature

        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)            # b n x y d

        # local-to-global cross-attention
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')  # reverse window to feature
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                      w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')  # reverse window to feature
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                      w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition
        query = rearrange(self.cross_win_attend_2(query,
                                                key,
                                                val,
                                                skip=rearrange(x_skip,
                                                               'b (x w1) (y w2) d -> b x y w1 w2 d',
                                                               w1=self.q_win_size[0],
                                                               w2=self.q_win_size[1])
                                                if self.skip else None),
                          'b x y w1 w2 d  -> b (x w1) (y w2) d')  # reverse grid to feature

        query = query + self.mlp_2(self.prenorm_2(query))

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
        high_perf_backbone=None,
        entropy_threshold: float = 4.5, # <<<<<<< 엔트로피 임계값 추가
        object_count_threshold: int = 20,

        # 새 옵션들:
        enable_color_correction: bool = True,               # 색보정 사용 여부 (엔트로피 계산에 적용)
        use_corrected_for_backbone: bool = False,           # 백본 입력에도 보정이미지를 쓸지 여부
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        self.high_perf_backbone = high_perf_backbone
        self.entropy_threshold = entropy_threshold # <<<<<<< 임계값 저장
        self.object_count_threshold = object_count_threshold

        # 색보정 옵션 저장
        self.enable_color_correction = enable_color_correction
        self.use_corrected_for_backbone = use_corrected_for_backbone

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
        # self.self_attn = Attention(dim[-1], **self_attn)


        def _color_correct_images(self, images: torch.Tensor) -> torch.Tensor:
            """
            간단한 저조도 보정:
              1) 이미지 밝기(휘도) 기반 감마 보정 (저조도면 밝게)
              2) 각 채널에 대해 히스토그램 평활화(정규화된 CDF 매핑) 적용
    
            images: (b, n, c, h, w), float in [0,1] 예상
            returns: corrected images in same shape, float in [0,1]
            """
            device = images.device
            imgs = images.clone()  # (b, n, c, h, w)
    
            b, n, c, h, w = imgs.shape
            # 1) 휘도(평균 밝기) 계산 (각 카메라별, 샘플별)
            lum = 0.2989 * imgs[:, :, 0] + 0.5870 * imgs[:, :, 1] + 0.1140 * imgs[:, :, 2]  # (b, n, h, w)
            mean_lum_per_cam = lum.view(b, n, -1).mean(-1)  # (b, n)
            # 카메라들 평균 -> 샘플당 평균휘도
            mean_lum_per_sample = mean_lum_per_cam.mean(dim=1)  # (b,)
    
            # 감마 값 결정: 매우 어두우면 더 강하게 밝게 (감마 < 1 -> 밝아짐)
            # 하이퍼파라미터: 조정 가능
            gamma = torch.ones(b, device=device)
            gamma = torch.where(mean_lum_per_sample < 0.15, 0.6, gamma)
            gamma = torch.where((mean_lum_per_sample >= 0.15) & (mean_lum_per_sample < 0.35), 0.8, gamma)
            gamma = gamma.view(b, 1, 1, 1, 1)  # (b,1,1,1,1) - expand for n and channels when broadcasting
    
            # apply gamma per-sample to all cameras
            imgs = imgs ** gamma  # 브로드캐스트: (b,n,c,h,w)
    
            # 2) 채널별 히스토그램 평활화 (작업: 각 (b,n,ch)별로 수행)
            # 성능상 비용이 있으나 간단한 균등화는 시도해볼 만함.
            imgs_255 = torch.clamp((imgs * 255.0).round().to(torch.int64), 0, 255)
    
            out = torch.empty_like(imgs, dtype=torch.float32, device=device)
            for bi in range(b):
                for ni in range(n):
                    for ch in range(3):
                        vals = imgs_255[bi, ni, ch].flatten()               # (h*w,)
                        hist = torch.bincount(vals, minlength=256).float()  # (256,)
                        cdf = torch.cumsum(hist, dim=0)
                        # 정규화된 CDF
                        cdf_min = cdf[0]
                        denom = (cdf[-1] - cdf_min).clamp(min=1.0)
                        cdf_norm = (cdf - cdf_min) / denom                  # in [0,1]
                        mapped = cdf_norm[vals].view(h, w)                 # 매핑된 값 (0..1)
                        out[bi, ni, ch] = mapped
    
            # out already in [0,1], float32
            return out
    
        # (원래 _calculate_attention_map_entropy는 남기되, forward에서 호출 시 보정된 이미지를 전달)

    

    def _calculate_attention_map_entropy(self, images: torch.Tensor) -> torch.Tensor:
        """
        이미지 배치의 평균 엔트로피를 계산합니다.
        엔트로피는 이미지의 복잡도를 나타내는 척도로 사용됩니다.
        
        images: (b, n, c, h, w) 형태의 텐서
        returns: 배치 전체의 평균 엔트로피 (스칼라 텐서)
        """
        b, n, c, h, w = images.shape
        
        # (b, n, c, h, w) -> (b*n, c, h, w)
        images_reshaped = images.reshape(b * n, c, h, w)
        
        # 그레이스케일로 변환: (b*n, h, w)
        # Y = 0.299R + 0.587G + 0.114B
        grayscale_images = 0.2989 * images_reshaped[:, 0, :, :] + \
                           0.5870 * images_reshaped[:, 1, :, :] + \
                           0.1140 * images_reshaped[:, 2, :, :]
        
        # 0-255 범위의 정수 값으로 변환
        grayscale_images = (grayscale_images * 255).long()

        entropies = []
        # 각 이미지에 대해 엔트로피 계산
        for i in range(b * n):
            # 픽셀 값의 히스토그램 계산
            hist = torch.bincount(grayscale_images[i].flatten(), minlength=256).float()
            
            # 확률 분포 계산
            prob = hist / (h * w)
            
            # log(0)을 피하기 위해 작은 값(epsilon) 추가
            # 엔트로피 계산: H = -sum(p * log2(p))
            entropy = -torch.sum(prob * torch.log2(prob + 1e-9))
            entropies.append(entropy)
            
        # 텐서로 변환 후 전체 배치의 평균 엔트로피 계산
        return torch.stack(entropies).mean()

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        I_inv = batch['intrinsics'].inverse()      # b n 3 3
        E_inv = batch['extrinsics'].inverse()      # b n 4 4
        
        object_count = batch.get('object_count', None)

        # --- 색보정 적용(옵션) ---
        if self.enable_color_correction:
            # corrected_for_entropy: 엔트로피 계산용(항상 사용)
            corrected_for_entropy = self._color_correct_images(batch['image'])
        else:
            corrected_for_entropy = batch['image']

        # 1) 배치 전체의 평균 엔트로피 계산 (보정된 영상 사용)
        avg_entropy = self._calculate_attention_map_entropy(corrected_for_entropy)

        # <<<< 1. 평균 객체 수 계산 로직 추가 >>>>
        # object_count가 제공되었는지 확인
        object_count_available = object_count is not None
        avg_object_count = 0.0 # 기본값 초기화
        if object_count_available:
            # 텐서의 평균을 계산 (.float()으로 타입 변환 후 .mean())
            avg_object_count = object_count.float().mean()

        # <<<< 2. 디버깅 코드 수정 >>>>
        print(f"[디버깅 정보] Avg Entropy: {avg_entropy:.2f} (임계값: {self.entropy_threshold}), "
              f"Avg Objects: {avg_object_count:.2f} (임계값: {self.object_count_threshold})")


        # --- 백본 입력: 보정영상을 쓸지 여부 선택 ---
        if self.use_corrected_for_backbone and self.enable_color_correction:
            images_for_backbone = rearrange(corrected_for_entropy, 'b n c h w -> (b n) c h w')
        else:
            images_for_backbone = rearrange(batch['image'], 'b n c h w -> (b n) c h w')
        
        
        # 2. 엔트로피 값에 따라 분기 처리
        if (avg_entropy >= self.entropy_threshold) and \
           (object_count_available and avg_object_count >= self.object_count_threshold):
            # [CASE 1] 엔트로피가 높을 때: 개별적으로 백본 처리 (기존 방식)
            # print(f"High entropy ({avg_entropy:.2f}), processing batch items individually.")
            num_feature_levels = len(self.backbone.output_shapes)
            features_per_level = [[] for _ in range(num_feature_levels)]

            for i in range(b):
                sample_images = batch['image'][i] # (n, c, h, w)

                # object_count에 따라 사용할 백본 선택
                if self.high_perf_backbone is not None and object_count is not None and object_count[i] >= 30:
                    backbone_to_use = self.high_perf_backbone
                else:
                    backbone_to_use = self.backbone

                sample_features = backbone_to_use(self.norm(sample_images))
                
                for level_idx, feat in enumerate(sample_features):
                    features_per_level[level_idx].append(self.down(feat))

            # 각 레벨의 피처들을 하나로 합침
            features = [torch.cat(feats, dim=0) for feats in features_per_level]

        else:
            # [CASE 2] 엔트로피가 낮을 때: 배치 전체를 한번에 백본 처리
            # print(f"Low entropy ({avg_entropy:.2f}), processing batch as a whole.")
            
            # (b, n, c, h, w) -> (b * n, c, h, w)
            images_flat = rearrange(batch['image'], 'b n c h w -> (b n) c h w')
            
            # 배치 전체 처리 시에는 기본 백본을 사용
            backbone_to_use = self.backbone
            
            # 정규화 및 백본 통과
            features_list = backbone_to_use(self.norm(images_flat))
            
            # 각 피처 레벨에 다운샘플링 적용
            features = [self.down(feat) for feat in features_list]

        # <<<<<<<<<<<<<<<< END: 신규 로직 추가 >>>>>>>>>>>>>>>>

        # 이후 로직은 두 경우 모두 동일하게 적용됨
        x = self.bev_embedding.get_prior()        # d H W
        x = repeat(x, '... -> b ...', b=b)        # b d H W

        for i, (cross_view, feature, layer) in \
                enumerate(zip(self.cross_views, features, self.layers)):
            
            # (b*n, c, h, w) -> (b, n, c, h, w)
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        # x = self.self_attn(x)
        return x


if __name__ == "__main__":
    # 이 부분은 외부 파일(config, backbone 구현)에 의존하므로
    # 직접 실행하기보다는 클래스 구조와 로직을 확인하는 용도로 사용하세요.
    
    # 예시: torchvision에서 백본 로드 (실제 코드에서는 별도의 백본 래퍼 클래스가 필요할 수 있음)
    # class MyBackboneWrapper(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         # ... 실제 백본 모델 로드 및 중간 피처 추출 로직 ...
    #         self.output_shapes = [(1, 64, 128, 128), (1, 128, 64, 64), ...] # 예시
    #     def forward(self, x):
    #         # ... 피처 추출 로직 ...
    #         return [feat1, feat2, ...]
    
    # backbone_std = MyBackboneWrapper()
    # backbone_high_perf = MyBackboneWrapper() # 고성능 백본 래퍼
    
    # # 모델 초기화 
    # encoder = PyramidAxialEncoder(
    #     backbone=backbone_std,
    #     high_perf_backbone=backbone_high_perf,
    #     entropy_threshold=4.5, # 임계값 설정
    #     # ... 기타 설정값들 ...
    # )

    print("수정된 PyramidAxialEncoder 클래스가 로드되었습니다.")
    print("forward 메서드에 엔트로피 기반의 동적 백본 처리 로직이 추가되었습니다.")
