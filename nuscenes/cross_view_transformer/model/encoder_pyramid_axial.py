import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List
# from .decoder import  DecoderBlock # 원본 코드에 있었으나 정의가 없어 주석 처리

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
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

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
        batch, _, height, width, h = *x.shape, self.heads
        x = rearrange(x, 'b d h w -> b (h w) d')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (H W) d -> b H W (h d)', H=height, W=width)
        out = self.to_out(out)
        return rearrange(out, 'b H W d -> b d H W')


class CrossWinAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.proj = nn.Linear(heads * dim_head, dim)

    def forward(self, q, k, v, skip=None):
        _, _, q_height, q_width, _, _, _ = q.shape
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h=self.heads), (q, k, v))
        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)
        att = dot.softmax(dim=-1)
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b h) ... d -> b ... (h d)', h=self.heads)
        a = rearrange(a, 'b (x y) (n w1 w2) d -> b n x y w1 w2 d', x=q_height, y=q_width, n=v.shape[1] // (q_height * q_width))
        z = self.proj(a).mean(1)
        if skip is not None:
            z = z + skip
        return z

# --- START: MoE를 위한 전문가 모듈 정의 ---
class AttentionExpert(nn.Module):
    """'단순 전문가': 표준적인 어텐션 기반 처리를 수행합니다."""
    def __init__(self, dim, heads, dim_head, qkv_bias, q_win_size, feat_win_size, skip, norm):
        super().__init__()
        self.q_win_size = q_win_size
        self.feat_win_size = feat_win_size
        self.skip = skip
        self.attend = CrossWinAttention(dim, heads, dim_head, qkv_bias, norm=norm)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, query, key, val, x_skip, n):
        query_repeated = repeat(query, 'b x y d -> b n x y d', n=n)
        query_win = rearrange(query_repeated, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_grid = rearrange(key, 'b n d (x w1) (y w2) -> b n y x w2 w1 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_grid = rearrange(val, 'b n d (x w1) (y w2) -> b n y x w2 w1 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        x_skip_rearranged = None
        if self.skip and x_skip is not None:
            x_skip_rearranged = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])

        attended = self.attend(query_win, key_grid, val_grid, skip=x_skip_rearranged)
        attended = rearrange(attended, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        
        return attended + self.mlp(self.prenorm(attended))

class ConvolutionExpert(nn.Module):
    """'복합 전문가': 합성곱 기반으로 국소적 공간 정보를 처리합니다."""
    def __init__(self, dim, norm):
        super().__init__()
        self.conv_path = nn.Sequential(
            ResNetBottleNeck(dim),
            ResNetBottleNeck(dim)
        )
    
    def forward(self, query):
        # 입력 shape: (b, H, W, d), conv는 (b, d, H, W)를 기대
        query_reshaped = rearrange(query, 'b H W d -> b d H W')
        refined = self.conv_path(query_reshaped)
        # 출력 shape을 다시 (b, H, W, d)로 복원
        return rearrange(refined, 'b d H W -> b H W d')
# --- END: MoE를 위한 전문가 모듈 정의 ---

class CrossViewSwapAttention(nn.Module):
    def __init__(
        self,
        feat_height: int, feat_width: int, feat_dim: int, dim: int, index: int,
        image_height: int, image_width: int, qkv_bias: bool, q_win_size: list,
        feat_win_size: list, heads: list, dim_head: list, bev_embedding_flag: list,
        rel_pos_emb: bool = False, no_image_features: bool = False, skip: bool = True,
        norm=nn.LayerNorm
    ):
        super().__init__()
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        self.feature_proj = None
        if not no_image_features:
            self.feature_proj = nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.q_win_size = q_win_size[index]
        self.feat_win_size = feat_win_size[index]
        self.skip = skip

        # 공유하는 첫 번째 어텐션 블록 (Stem)
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias, norm=norm)
        self.prenorm_1 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

        # --- START: MoE 전문가 모듈 초기화 ---
        self.expert_simple = AttentionExpert(dim, heads[index], dim_head[index], qkv_bias, self.q_win_size, self.feat_win_size, skip, norm)
        self.expert_complex = ConvolutionExpert(dim, norm)
        # --- END: MoE 전문가 모듈 초기화 ---

        self.postnorm = norm(dim)

    def pad_divisble(self, x, win_h, win_w):
        _, _, _, h, w = x.shape
        padh = (win_h - h % win_h) % win_h
        padw = (win_w - w % win_w) % win_w
        return F.pad(x, (0, padw, 0, padh))

    def forward(self, index: int, x: torch.FloatTensor, bev: BEVEmbedding, feature: torch.FloatTensor, I_inv: torch.FloatTensor, E_inv: torch.FloatTensor, object_count: Optional[torch.Tensor] = None):
        b, n, _, _, _ = feature.shape
        pixel = self.image_plane
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        c_embed = self.cam_embed(c_flat)

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        cam = I_inv @ pixel_flat
        cam = F.pad(cam, (0, 0, 0, 1), value=1)
        d = E_inv @ cam
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)
        d_embed = self.img_embed(d_flat)

        img_embed = d_embed - c_embed
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        world = getattr(bev, f'grid{index}')[:2]
        
        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        key_flat = img_embed
        if self.feature_proj is not None:
            key_flat = key_flat + self.feature_proj(feature_flat)
        val_flat = self.feature_linear(feature_flat)

        query = x[:, None]
        if self.bev_embed_flag:
            query = query_pos + query
            
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)

        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # 첫 번째 공통 블록 (Stem)
        query_win = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key_win = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val_win = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        skip_rearranged = None
        if self.skip:
             skip_rearranged = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        
        query = self.cross_win_attend_1(query_win, key_win, val_win, skip=skip_rearranged)
        query = rearrange(query, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_1(self.prenorm_1(query))
        
        x_skip = query
        
        # --- START: MoE 게이팅 로직 ---
        # 두 전문가를 모두 실행하여 DDP 에러를 방지
        output_simple = self.expert_simple(query, key, val, x_skip, n)
        output_complex = self.expert_complex(query)

        if object_count is not None:
            # 마스크 생성. 텐서 연산을 위해 차원 확장 (b,) -> (b, 1, 1, 1)
            mask = (object_count >= 30).view(-1, 1, 1, 1)
            # torch.where를 사용해 조건에 따라 두 전문가의 결과물을 조합
            query = torch.where(mask, output_complex, output_simple)
        else:
            # object_count가 없으면 단순 전문가의 결과만 사용
            query = output_simple
        # --- END: MoE 게이팅 로직 ---

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
        self.down = (lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)) if scale < 1.0 else (lambda x: x)

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views, layers, downsample_layers = [], [], []
        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            cross_views.append(CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap))
            layers.append(nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)]))
            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Conv2d(dim[i], dim[i] // 2, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(dim[i+1], dim[i+1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim[i+1]), nn.ReLU(inplace=True),
                    nn.Conv2d(dim[i+1], dim[i+1], 1, padding=0, bias=False),
                    nn.BatchNorm2d(dim[i+1])
                ))

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

        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)
        return x

if __name__ == "__main__":
    # main 블록은 테스트 코드이므로 수정하지 않았습니다.
    # 이 코드를 실행하기 위해서는 실제 백본과 설정 파일이 필요합니다.
    print("Mixture-of-Experts (MoE) based adaptive model code has been generated.")
