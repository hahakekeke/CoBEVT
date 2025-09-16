import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

# ===================================================================
# [신규] 엔트로피 계산을 위한 헬퍼 함수
# ===================================================================
def calculate_attention_entropy(attention_map: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
    """어텐션 맵의 정보 엔트로피를 계산합니다."""
    log_p = torch.log2(attention_map + epsilon)
    entropy_per_token = -torch.sum(attention_map * log_p, dim=-1)
    return entropy_per_token.mean()

# ===================================================================
# 기존 클래스 및 함수들
# ===================================================================
ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)
    indices = indices[None]
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
        [ 0.,  0.,         1.]
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
            self.register_buffer('grid%d'%i, grid, persistent=False)
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim,
                                bev_height//upsample_scales[0],
                                bev_width//upsample_scales[0]))
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

# ===================================================================
# [수정] CrossWinAttention: 어텐션 맵을 반환하도록 수정
# ===================================================================
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
        # [수정] 어텐션 맵(att)을 함께 반환
        return z, att

# ===================================================================
# [수정] CrossViewSwapAttention: CrossWinAttention의 반환값 변경에 맞춰 호출부 수정
# ===================================================================
class CrossViewSwapAttention(nn.Module):
    def __init__(
        self,
        feat_height: int, feat_width: int, feat_dim: int, dim: int, index: int,
        image_height: int, image_width: int, qkv_bias: bool, q_win_size: list,
        feat_win_size: list, heads: list, dim_head: list, bev_embedding_flag: list,
        rel_pos_emb: bool = False, no_image_features: bool = False, skip: bool = True,
        norm=nn.LayerNorm,
    ):
        super().__init__()
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)
        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        if no_image_features: self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag: self.bev_embed = nn.Conv2d(2, dim, 1)
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

    def forward(
        self, index: int, x: torch.FloatTensor, bev: BEVEmbedding, feature: torch.FloatTensor,
        I_inv: torch.FloatTensor, E_inv: torch.FloatTensor, object_count: Optional[torch.Tensor] = None,
    ):
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
        if index == 0: world = bev.grid0[:2]
        elif index == 1: world = bev.grid1[:2]
        elif index == 2: world = bev.grid2[:2]
        elif index == 3: world = bev.grid3[:2]
        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])
            bev_embed = w_embed - c_embed
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        if self.feature_proj is not None: key_flat = img_embed + self.feature_proj(feature_flat)
        else: key_flat = img_embed
        val_flat = self.feature_linear(feature_flat)
        if self.bev_embed_flag: query = query_pos + x[:, None]
        else: query = x[:, None]
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])
        
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        # [수정] 반환값이 2개이므로 첫 번째 값만 사용하고, rearrange는 그 후에 적용
        skip_rearranged = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query_out, _ = self.cross_win_attend_1(query, key, val, skip=skip_rearranged)
        query = rearrange(query_out, 'b x y w1 w2 d -> b (x w1) (y w2) d')

        query = query + self.mlp_1(self.prenorm_1(query))
        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        
        # [수정] 반환값이 2개이므로 첫 번째 값만 사용하고, rearrange는 그 후에 적용
        skip_rearranged = rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None
        query_out, _ = self.cross_win_attend_2(query, key, val, skip=skip_rearranged)
        query = rearrange(query_out, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        
        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        query = rearrange(query, 'b H W d -> b d H W')
        return query

# ===================================================================
# [수정] PyramidAxialEncoder: 메인 클래스 로직 대폭 수정
# ===================================================================
class PyramidAxialEncoder(nn.Module):
    def __init__(
        self,
        backbone,
        cross_view: dict,
        cross_view_swap: dict,
        bev_embedding: dict,
        dim: list,
        middle: List[int] = [2, 2],
        scale: float = 1.0,
        high_perf_backbone=None,
        entropy_threshold: float = 4.5, # [신규] 엔트로피 임계값 파라미터
    ):
        super().__init__()
        
        self.norm = Normalize()
        self.backbone = backbone
        self.high_perf_backbone = high_perf_backbone
        self.ENTROPY_THRESHOLD = entropy_threshold # [신규] 임계값 저장

        # [신규] -------------------------------------------------------------
        # 1. 사전 피처 추출을 위한 아주 가벼운 CNN 모델 정의
        # --------------------------------------------------------------------
        pre_attn_dim = 32
        self.shallow_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, pre_attn_dim, kernel_size=3, stride=2, padding=1),
        )

        # [신규] -------------------------------------------------------------
        # 2. 사전 어텐션 계산 전용 CrossWinAttention 모듈 정의
        # --------------------------------------------------------------------
        self.pre_attention_module = CrossWinAttention(
            dim=pre_attn_dim, heads=4, dim_head=8, qkv_bias=False
        )
        self.pre_bev_embed = nn.Conv2d(2, pre_attn_dim, 1)
        self.pre_cam_embed = nn.Conv2d(4, pre_attn_dim, 1, bias=False)
        self.pre_img_embed = nn.Conv2d(4, pre_attn_dim, 1, bias=False)

        # [기존 코드] ---------------------------------------------------------
        # 메인 Cross-view Attention 모듈들 (기존과 동일)
        # --------------------------------------------------------------------
        if scale < 1.0: self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else: self.down = lambda x: x
        
        assert len(self.backbone.output_shapes) == len(middle)
        cross_views, layers, downsample_layers = [], [], []
        
        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)
            layers.append(nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)]))
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
        b, n, c, h, w = batch['image'].shape
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        object_count = batch.get('object_count', None)

        # [신규] =================================================================
        # 1. 사전 어텐션 계산 단계 (Proxy Entropy Calculation)
        # =========================================================================
        with torch.no_grad():
            all_images = rearrange(batch['image'], 'b n c h w -> (b n) c h w')
            norm_images = self.norm(all_images)
            shallow_features = self.shallow_feature_extractor(norm_images)
            
            # 사전 어텐션을 위한 Query(BEV), Key(Image) 생성
            bev_grid = self.bev_embedding.grid0[:2][None]
            q_bev_pos = self.pre_bev_embed(bev_grid)

            # 이미지 좌표계 생성 (CrossViewSwapAttention의 로직 재활용)
            _, _, sh, sw = shallow_features.shape
            pixel = generate_grid(sh, sw)
            pixel[:, 0] *= w
            pixel[:, 1] *= h
            
            pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
            cam = I_inv @ pixel_flat
            cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
            d = E_inv @ cam
            d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=sh, w=sw)
            d_embed = self.pre_img_embed(d_flat)
            
            c_embed = self.pre_cam_embed(E_inv[..., -1:].flatten(0, 1)[..., None])
            k_img_pos = d_embed - c_embed
            k_img_pos = k_img_pos / (k_img_pos.norm(dim=1, keepdim=True) + 1e-7)

            key_shallow = k_img_pos + shallow_features
            
            # 윈도우 파티셔닝 없이 전체를 하나의 윈도우로 간주하여 간단히 계산
            q_bev = rearrange(q_bev_pos, '1 d h w -> 1 1 1 1 h w d')
            k_img = rearrange(key_shallow, '(b n) d h w -> b n 1 1 h w d', b=b, n=n)

            _, pre_attn_map = self.pre_attention_module(q_bev, k_img, k_img)
            avg_entropy = calculate_attention_entropy(pre_attn_map)
            print(f"Pre-Attention Entropy: {avg_entropy.item():.4f}")

        # [신규] =================================================================
        # 2. 엔트로피 기반 조건부 분기
        # =========================================================================
        features_per_level = [[] for _ in range(len(self.backbone.output_shapes))]
        if avg_entropy >= self.ENTROPY_THRESHOLD:
            print(f"High entropy detected. Processing samples individually.")
            for i in range(b):
                sample_images = batch['image'][i]
                if self.high_perf_backbone is not None and object_count is not None and object_count[i] >= 30:
                    backbone_to_use = self.high_perf_backbone
                    print(f"  - Batch index {i} uses high-performance backbone (object count: {object_count[i]})")
                else:
                    backbone_to_use = self.backbone
                sample_features = backbone_to_use(self.norm(sample_images))
                for level_idx, feat in enumerate(sample_features):
                    features_per_level[level_idx].append(self.down(feat))
            features = [torch.cat(feats, dim=0) for feats in features_per_level]
        else:
            print(f"Low entropy detected. Processing batch at once.")
            all_images_flat = rearrange(batch['image'], 'b n c h w -> (b n) c h w')
            batched_features = self.backbone(self.norm(all_images_flat))
            features = [self.down(feat) for feat in batched_features]

        # =========================================================================
        # 3. 메인 Cross-View Attention 및 후속 처리 (공통)
        # =========================================================================
        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)

        for i, (cross_view, feature, layer) in \
                enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
            if i < len(self.layers)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)
        return x

if __name__ == "__main__":
    print("수정된 PyramidAxialEncoder 클래스가 로드되었습니다.")
    print("forward 메서드에 사전 엔트로피 계산 및 조건부 처리 로직이 추가되었습니다.")
