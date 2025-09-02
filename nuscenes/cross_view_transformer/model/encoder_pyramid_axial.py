import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Tuple
# from .decoder import  DecoderBlock # 주석 처리 (현재 파일에 없으므로)

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
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,        w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,           1.]
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

        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim,
                                bev_height//upsample_scales[0],
                                bev_width//upsample_scales[0]))  # d h w

    def get_prior(self):
        return self.learned_features

# [추가] SwiGLU FFN. 기존 MLP보다 표현력이 좋다고 알려져 있습니다.
class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

# [추가] Object Count 정보를 BEV Feature에 주입하기 위한 어댑터 모듈
class ObjectCountAdapter(nn.Module):
    def __init__(self, feature_dim: int, object_count_dim: int, mlp_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(object_count_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, feature_dim * 2) # gamma와 beta를 위해 2배
        )
        
    def forward(self, x: torch.Tensor, object_count: torch.Tensor) -> torch.Tensor:
        # object_count: [B, object_count_dim]
        # x: [B, C, H, W]
        
        # StyleGAN의 AdaIN과 유사한 방식으로 변조(Modulation)
        style = self.mlp(object_count.float()) # [B, C*2]
        gamma, beta = style.chunk(2, dim=1) # [B, C], [B, C]
        
        # x의 shape에 맞게 broadcasting을 위해 reshape
        gamma = gamma.view(-1, self.feature_dim, 1, 1) # [B, C, 1, 1]
        beta = beta.view(-1, self.feature_dim, 1, 1)   # [B, C, 1, 1]
        
        return x * (1 + gamma) + beta


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64, # [수정] dim_head 값을 32에서 64로 늘려 표현력 강화
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
        batch, _, height, width, h = *x.shape, self.heads

        x = rearrange(x, 'b d h w -> b (h w) d')
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        
        # 윈도우 크기가 입력의 H, W와 다를 수 있으므로 rel_pos_indices를 잘라내서 사용
        if height * width != self.rel_pos_indices.shape[0]:
            # This is a temporary fix for variable window sizes. A more robust solution might be needed.
            local_indices = self.rel_pos_indices[:height*width, :height*width]
            bias = self.rel_pos_bias(local_indices)
        else:
            bias = self.rel_pos_bias(self.rel_pos_indices)

        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        return rearrange(out, 'b (h w) d -> b d h w', h=height, w=width)


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
        object_count_dim: int, # [추가] object_count의 차원
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

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        # [수정] MLP를 SwiGLU로 변경
        self.mlp_1 = SwiGLU(dim, dim * 2, dim) 
        self.mlp_2 = SwiGLU(dim, dim * 2, dim)
        self.postnorm = norm(dim)
        
        # [추가] ObjectCountAdapter 추가
        if object_count_dim > 0:
            self.obj_adapter = ObjectCountAdapter(dim, object_count_dim)
        else:
            self.obj_adapter = None


    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divible by window size."""
        _, _, _, h, w = x.shape
        padh = (win_h - h % win_h) % win_h
        padw = (win_w - w % win_w) % win_w
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
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
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
        
        # [수정] query 패딩 추가. key/val과 window 수가 맞아야 함
        query = self.pad_divisble(query, self.q_win_size[0], self.q_win_size[1])
        x_padded = self.pad_divisble(x[:, None], self.q_win_size[0], self.q_win_size[1]).squeeze(1)


        # local-to-local cross-attention
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])

        skip_conn = rearrange(x_padded, 'b d (x w1) (y w2) -> b x y w1 w2 d',
                              w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        
        query = self.cross_win_attend_1(query, key, val, skip=skip_conn)
        query = rearrange(query, 'b x y w1 w2 d -> b (x w1) (y w2) d')

        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)

        # local-to-global cross-attention
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        skip_conn_2 = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d',
                                w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None

        query = self.cross_win_attend_2(query, key, val, skip=skip_conn_2)
        query = rearrange(query, 'b x y w1 w2 d -> b (x w1) (y w2) d')

        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        query = rearrange(query, 'b H W d -> b d H W')
        
        # [수정] 원본 크기로 복원 (패딩 제거)
        query = query[:, :, :H, :W]

        # [추가] object_count가 있다면 어댑터를 통해 특징을 조절
        if self.obj_adapter is not None and object_count is not None:
            query = self.obj_adapter(query, object_count)

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
            object_count_dim: int = 8, # [추가] object_count의 차원. nuScenes은 보통 8개 클래스
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
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(1, *feat_shape)).shape

            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, 
                                         object_count_dim=object_count_dim, # [추가]
                                         **cross_view, **cross_view_swap)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)

            if i < len(middle) - 1:
                # [수정] 다운샘플링 블록을 조금 더 표준적인 방식으로 변경 (Conv-stride)
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dim[i+1]),
                    nn.ReLU(inplace=True)
                ))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        
        # [수정] Self-Attention을 활성화하고, 마지막 BEV 특징맵에 적용
        self.self_attn = Attention(dim[-1], **self_attn)

    def forward(self, batch: dict) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)      # b n c h w
        I_inv = batch['intrinsics'].inverse()     # b n 3 3
        E_inv = batch['extrinsics'].inverse()     # b n 4 4

        # ✅ 여기서 object_count 가져오기
        object_count = batch.get('object_count', None)
        
        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()        # d H W
        x = repeat(x, '... -> b ...', b=b)        # b d H W
        
        # [추가] 디코더로 전달할 중간 BEV 특징들을 저장할 리스트
        bev_features_list = []

        for i, (cross_view, feature, layer) in \
                enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)

            # [추가] 다운샘플링 전에 현재 스케일의 특징을 저장
            bev_features_list.append(x)

            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        # [수정] 마지막에 BEV 공간 내 Self-Attention 수행
        x = x + self.self_attn(x)
        
        # [수정] 마지막 레이어의 결과도 리스트에 추가
        bev_features_list.append(x)
        
        # [수정] 디코더를 위해 마지막 BEV 특징맵과 중간 스케일의 특징맵 리스트를 함께 반환
        return x, bev_features_list
