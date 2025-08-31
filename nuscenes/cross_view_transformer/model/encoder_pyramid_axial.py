import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

# --- 가상 DecoderBlock 클래스 정의 (원본 코드의 .decoder 임포트를 대체) ---
# 실제 프로젝트에서는 기존의 DecoderBlock을 사용하시면 됩니다.
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
# --- 가상 DecoderBlock 정의 종료 ---


ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)      # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                  # 3 h w
    indices = indices[None]                                                 # 1 3 h w

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
        q = self.to_q(q)                                  # b (X Y) (n W1 W2) (heads dim_head)
        k = self.to_k(k)                                  # b (X Y) (n w1 w2) (heads dim_head)
        v = self.to_v(v)                                  # b (X Y) (n w1 w2) (heads dim_head)

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
        I_inv = batch['intrinsics'].inverse()       # b n 3 3
        E_inv = batch['extrinsics'].inverse()       # b n 4 4

        object_count = batch.get('object_count', None)

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()          # d H W
        x = repeat(x, '... -> b ...', b=b)          # b d H W

        for i, (cross_view, feature, layer) in \
                enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            
            # object_count는 필요 시 CrossViewSwapAttention 내부에서 활용 가능
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        # x = self.self_attn(x)

        return x

# --- 변경 사항 시작 ---

class FusionLayer(nn.Module):
    """
    여러 BEV 인코더의 feature map을 융합(fusion)하는 클래스.
    Concatenate 후 1x1 Conv로 채널을 조정하고 정보를 통합합니다.
    """
    def __init__(self, num_encoders: int, in_channels: int, out_channels: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_encoders * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feature_maps: 각 인코더에서 나온 BEV feature map의 리스트. 
                          각 텐서의 shape는 (b, c, h, w)
        Returns:
            융합된 feature map.
        """
        # 채널(dim=1)을 기준으로 feature map들을 합칩니다.
        concatenated_features = torch.cat(feature_maps, dim=1)
        fused_features = self.fusion_conv(concatenated_features)
        
        return fused_features


class EnsembleBEVModel(nn.Module):
    """
    Feature-level Ensemble을 수행하는 메인 모델.
    서로 다른 구조를 가질 수 있는 여러 개의 BEV 인코더를 포함합니다.
    """
    def __init__(self, encoder_configs: List[dict], num_classes: int):
        super().__init__()
        
        # 설정에 따라 여러 개의 BEV 인코더(PyramidAxialEncoder)를 생성
        self.encoders = nn.ModuleList([
            PyramidAxialEncoder(**config) for config in encoder_configs
        ])
        
        # 모든 인코더의 출력 채널 크기가 동일하다고 가정
        # PyramidAxialEncoder의 마지막 레이어 출력 채널
        final_encoder_dim = encoder_configs[0]['dim'][-1]
        num_encoders = len(encoder_configs)
        
        # Fusion Layer 초기화
        # 융합된 feature의 채널 수를 단일 인코더의 출력 채널 수와 동일하게 설정
        self.fusion_layer = FusionLayer(
            num_encoders=num_encoders,
            in_channels=final_encoder_dim,
            out_channels=final_encoder_dim
        )
        
        # 융합된 feature를 받아 최종 출력을 만드는 Decoder
        # 예시: 2번의 업샘플링 후 최종 클래스 수로 채널 변경
        self.decoder = nn.Sequential(
            DecoderBlock(final_encoder_dim, final_encoder_dim // 2),
            DecoderBlock(final_encoder_dim // 2, final_encoder_dim // 4),
            nn.Conv2d(final_encoder_dim // 4, num_classes, kernel_size=1)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        # 1. 각 인코더로부터 BEV feature map 추출
        feature_maps = []
        for encoder in self.encoders:
            # 각 인코더는 독립적으로 이미지 feature를 추출하고 BEV로 변환합니다.
            # 이로 인해 추론 시간은 길어지지만, 각 모델의 장점을 취할 수 있습니다.
            bev_features = encoder(batch)
            feature_maps.append(bev_features)
        
        # 2. 추출된 feature map들을 FusionLayer를 통해 융합
        fused_features = self.fusion_layer(feature_maps)
        
        # 3. 융합된 feature map을 Decoder에 통과시켜 최종 결과 생성
        output = self.decoder(fused_features)
        
        return output

# --- 변경 사항 종료 ---

if __name__ == "__main__":
    import os
    import re
    import yaml
    
    # --- 테스트를 위한 가상 백본 클래스 ---
    class DummyBackbone(nn.Module):
        def __init__(self, output_shapes):
            super().__init__()
            self.output_shapes = output_shapes
            # 각 layer는 해당 shape의 더미 텐서를 반환하도록 정의
            self.layers = nn.ModuleList([
                nn.Conv2d(3, shape[1], kernel_size=1) for shape in output_shapes
            ])
        
        def forward(self, x):
            # 실제 백본과 유사하게 여러 스케일의 feature를 리스트로 반환
            return [torch.randn(shape) for shape in self.output_shapes]

    # YAML 로더 (원본 코드 유지)
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

    # --- Ensemble 모델 테스트 ---
    print("Testing EnsembleBEVModel...")
    
    B, N, C, H, W = 2, 6, 3, 224, 480 # 배치, 카메라 수, 채널, 높이, 너비
    
    # 가상 백본 생성
    # 서로 다른 구조의 백본을 가정하기 위해 output_shapes를 다르게 설정할 수 있습니다.
    # 여기서는 동일한 백본을 사용하는 두 인코더로 앙상블을 테스트합니다.
    dummy_backbone = DummyBackbone(output_shapes=[
        (B*N, 64, H//4, W//4), 
        (B*N, 128, H//8, W//8)
    ])

    # 앙상블할 두 개의 BEV 인코더 설정
    # 최상의 성능을 위해서는 backbone, middle, dim 등을 다르게 설정하는 것이 좋습니다.
    encoder_config_1 = {
        'backbone': dummy_backbone,
        'bev_embedding': {'dim': 64, 'sigma': 1.0, 'bev_height': 100, 'bev_width': 100, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0, 'upsample_scales':[8, 16, 32, 64]},
        'cross_view': {'image_height': H, 'image_width': W, 'qkv_bias': True},
        'cross_view_swap': {'q_win_size': [[12, 12],[6,6]], 'feat_win_size': [[56, 120],[28,60]], 'heads': [4, 4], 'dim_head': [32, 32], 'bev_embedding_flag': [True, False, False, False]},
        'self_attn': {'dim_head': 32, 'dropout': 0.0, 'window_size': 12},
        'dim': [64, 128],
        'middle': [2, 2],
    }

    encoder_config_2 = {
        'backbone': dummy_backbone, # 다른 백본 사용 가능
        'bev_embedding': {'dim': 64, 'sigma': 1.0, 'bev_height': 100, 'bev_width': 100, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0, 'upsample_scales':[8, 16, 32, 64]},
        'cross_view': {'image_height': H, 'image_width': W, 'qkv_bias': True},
        'cross_view_swap': {'q_win_size': [[12, 12],[6,6]], 'feat_win_size': [[56, 120],[28,60]], 'heads': [8, 8], 'dim_head': [16, 16], 'bev_embedding_flag': [True, False, False, False]}, # Attention 헤드 수 변경
        'self_attn': {'dim_head': 32, 'dropout': 0.0, 'window_size': 12},
        'dim': [64, 128],
        'middle': [3, 3], # middle layer 수 변경
    }

    # Ensemble 모델 초기화
    ensemble_model = EnsembleBEVModel(
        encoder_configs=[encoder_config_1, encoder_config_2],
        num_classes=5  # 예: 5개 클래스 (도로, 차선 등)
    )

    # 더미 입력 데이터 생성
    batch = {
        'image': torch.rand(B, N, C, H, W),
        'intrinsics': torch.rand(B, N, 3, 3),
        'extrinsics': torch.rand(B, N, 4, 4),
        'object_count': torch.randint(0, 10, (B,))
    }

    # 모델 forward 테스트
    output = ensemble_model(batch)
    
    # 최종 출력 shape 확인
    # Decoder의 업샘플링에 따라 BEV 맵 크기가 결정됩니다.
    # 초기 BEV 크기: 100x100, 인코더 최종 출력(stride 16): (100/16) -> 6.25, 반올림하면 6x6
    # Decoder 2번 업샘플링: 6x6 -> 12x12 -> 24x24
    print("Ensemble model output shape:", output.shape) # 예상: [B, num_classes, 24, 24]
