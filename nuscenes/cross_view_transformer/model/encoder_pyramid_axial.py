import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

# from .decoder import DecoderBlock # 상대 경로 임포트는 스크립트 실행 시 오류를 유발할 수 있어 주석 처리합니다.

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

        #디버깅
        if object_count is not None:
            # print(">> object_count(crossviewswapattention):", object_count.shape, object_count) #각 인덱스가 특정 종류(차, 트럭, 보행자)의 객체 수임
            # value_1 = object_count[0].item()
            # value_2 = object_count[1].item()
            # value_3 = object_count[2].item()
            # value_4 = object_count[3].item()
            # value_5 = object_count[4].item()
            # value_6 = object_count[5].item()
            # value_7 = object_count[6].item()
            # value_8 = object_count[7].item()
            # print(f"Batch 0 object count: {value_1}")
            # print(f"Batch 1 object count: {value_2}")
            # print(f"Batch 2 object count: {value_3}")
            # print(f"Batch 3 object count: {value_4}")
            # print(f"Batch 4 object count: {value_5}")
            # print(f"Batch 5 object count: {value_6}")
            # print(f"Batch 6 object count: {value_7}")
            # print(f"Batch 7 object count: {value_8}")
            pass # 디버깅 프린트문은 주석 처리
        else:
            # print(">> object_count(crossviewswapattention) is None")
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
                        'b x y w1 w2 d  -> b (x w1) (y w2) d')     # reverse window to feature

        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)             # b n x y d

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
        # self.self_attn = Attention(dim[-1], **self_attn)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)        # b n c h w
        I_inv = batch['intrinsics'].inverse()         # b n 3 3
        E_inv = batch['extrinsics'].inverse()         # b n 4 4

        # ✅ 여기서 object_count 가져오기
        object_count = batch.get('object_count', None)

        #디버깅
        if object_count is not None:
            # print(">> object_count(pyramid axial encoder):", object_count.shape, object_count) #각 인덱스가 특정 종류(차, 트럭, 보행자)의 객체 수임
            pass # 디버깅 프린트문은 주석 처리
        else:
            # print(">> object_count(pyramid axial encoder) is None")
            pass
        
        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()            # d H W
        x = repeat(x, '... -> b ...', b=b)            # b d H W

        for i, (cross_view, feature, layer) in \
                enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        # x = self.self_attn(x)

        return x

# =========================================================================================
# ===                  [앙상블 모델] 새로운 BEVEnsembleModel 클래스 추가                  ===
# =========================================================================================
class BEVEnsembleModel(nn.Module):
    """
    Output-level Ensemble (Logit Averaging)을 수행하는 모델.
    서로 다른 구조 또는 다른 가중치로 학습된 여러 BEV 모델의 출력을 평균냅니다.
    """
    def __init__(self, models: List[nn.Module]):
        """
        Args:
            models (List[nn.Module]): 앙상블에 사용할 사전 학습된 BEV 모델의 리스트.
        """
        super().__init__()
        
        if not models:
            raise ValueError("모델 리스트는 비어 있을 수 없습니다.")
        
        self.models = nn.ModuleList(models)
        print(f"앙상블 모델이 {len(self.models)}개의 개별 모델로 초기화되었습니다.")

    def forward(self, batch: dict) -> torch.Tensor:
        """
        각 모델에서 로짓을 예측하고 평균을 계산합니다.
        추론 시에만 사용되어야 하므로, 내부적으로 `eval` 모드와 `no_grad` 컨텍스트에서 실행됩니다.

        Args:
            batch (dict): 단일 모델과 동일한 형식의 입력 데이터 배치.

        Returns:
            torch.Tensor: 모든 모델의 출력 로짓을 평균낸 최종 BEV 맵.
        """
        outputs = []
        
        for model in self.models:
            # 앙상블은 추론 시에 사용되므로, 각 모델을 eval 모드로 설정합니다.
            model.eval()
            
            # 그래디언트 계산을 비활성화하여 메모리 사용량과 계산 속도를 최적화합니다.
            with torch.no_grad():
                output = model(batch)
                outputs.append(output)
        
        # 출력들을 새로운 차원으로 쌓습니다. (num_models, batch_size, channels, height, width)
        stacked_outputs = torch.stack(outputs, dim=0)
        
        # 모델 차원(dim=0)을 따라 평균을 계산하여 최종 앙상블 출력을 얻습니다.
        ensembled_output = torch.mean(stacked_outputs, dim=0)
        
        return ensembled_output


if __name__ == "__main__":
    import os
    import re
    import yaml

    # --- 기존 코드의 YAML 로더 및 테스트 코드 ---
    def load_yaml(file):
        # ... (기존 YAML 로더 코드는 변경 없음)
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

    # --- 앙상블 모델 사용 예시 ---
    print("\n" + "="*50)
    print("      BEV 앙상블 모델 사용 예시")
    print("="*50)

    # 이 예시를 실행하려면 PyramidAxialEncoder의 의존성(backbone 등)이 필요합니다.
    # 개념적인 이해를 돕기 위해, 실제 모델 객체를 생성하는 부분은 가상으로 표현합니다.
    # 실제 사용 시에는 아래 주석 처리된 부분을 실제 모델 생성 및 가중치 로드 코드로 대체해야 합니다.

    # --- Step 1: 개별 모델들을 생성하고 학습된 가중치를 로드합니다. ---
    # 실제 시나리오: 서로 다른 설정으로 학습된 3개의 모델을 로드한다고 가정합니다.
    # 예: model1 = create_model('config1.yaml'); model1.load_state_dict(torch.load('model1.pth'))
    #     model2 = create_model('config2.yaml'); model2.load_state_dict(torch.load('model2.pth'))
    #     model3 = create_model('config3.yaml'); model3.load_state_dict(torch.load('model3.pth'))
    
    # 여기서는 개념 시연을 위해, 동일한 구조의 더미(dummy) 모델을 생성합니다.
    # PyramidAxialEncoder를 직접 생성하기에는 backbone 등 추가적인 구성요소가 필요하므로
    # 간단한 DummyModel로 대체하여 앙상블 로직만 보여줍니다.
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 128, 1)
            # PyramidAxialEncoder가 output_shapes 속성을 요구하므로 추가합니다.
            # (batch, channel, height, width) 형태
            self.output_shapes = [(1, 128, 28, 60)]

        def forward(self, x):
            return [self.conv(x)]
            
    class DummyModel(PyramidAxialEncoder):
        def __init__(self):
            # PyramidAxialEncoder를 초기화하기 위한 더미 파라미터들
            # 실제 사용 시에는 YAML 파일에서 이 값들을 로드해야 합니다.
            dummy_params = {
                'backbone': DummyBackbone(),
                'cross_view': {'image_height': 224, 'image_width': 400, 'qkv_bias': True},
                'cross_view_swap': {
                    'q_win_size': [[25, 25]], 'feat_win_size': [[28, 60]], 
                    'heads': [4], 'dim_head': [32], 'bev_embedding_flag': [True]
                },
                'bev_embedding': {
                    'sigma': 1.0, 'bev_height': 200, 'bev_width': 200,
                    'h_meters': 100, 'w_meters': 100, 'offset': 0.0,
                    'upsample_scales': [1, 2, 4, 8]
                },
                'self_attn': {},
                'dim': [128],
                'middle': [1] # backbone의 output_shapes 길이와 일치해야 함
            }
            super().__init__(**dummy_params)

    # 3개의 (동일한) 모델 인스턴스 생성
    model1 = DummyModel()
    model2 = DummyModel()
    model3 = DummyModel()
    
    # 실제로는 각 모델에 다른 가중치를 로드해야 합니다.
    # model1.load_state_dict(...)
    # model2.load_state_dict(...)
    # model3.load_state_dict(...)

    model_list = [model1, model2, model3]
    
    # --- Step 2: 앙상블 모델을 생성합니다. ---
    ensemble_model = BEVEnsembleModel(models=model_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_model.to(device)

    # --- Step 3: 더미 입력 데이터 생성 ---
    batch_size = 2
    num_cameras = 6
    dummy_batch = {
        'image': torch.rand(batch_size, num_cameras, 3, 224, 400).to(device),
        'intrinsics': torch.rand(batch_size, num_cameras, 3, 3).to(device),
        'extrinsics': torch.rand(batch_size, num_cameras, 4, 4).to(device),
        'object_count': torch.randint(0, 10, (batch_size,)).to(device)
    }

    # --- Step 4: 앙상블 모델로 추론 실행 ---
    print("\n앙상블 모델 추론을 시작합니다...")
    ensembled_output = ensemble_model(dummy_batch)
    print("앙상블 모델 추론 완료!")

    # --- Step 5: 출력 확인 ---
    print(f"입력 이미지 배치 형태: {dummy_batch['image'].shape}")
    print(f"최종 앙상블 출력 BEV맵 형태: {ensembled_output.shape}")

    # 개별 모델의 출력 형태와 최종 앙상블 출력 형태가 동일해야 합니다.
    # (batch_size, channels, height, width)
    # DummyModel의 마지막 차원은 128, BEV 임베딩의 초기 H, W는 200/8=25
    # 따라서 예상 출력: (2, 128, 25, 25)
    # (참고: downsample_layers가 없으므로 마지막 레이어의 출력 형태가 유지됨)
    print(f"예상 출력 형태: ({batch_size}, 128, 25, 25)")
    print("="*50)
