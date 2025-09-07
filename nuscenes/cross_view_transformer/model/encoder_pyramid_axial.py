import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from typing import List, Optional

from decoder import DecoderBlock   # 상대 경로 대신 절대 import


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++ ConvNeXt Block 구현 ++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block """

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) 
        return input + x


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++ 유틸 함수들 ++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), 0)   # (2, h, w)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)             # (3, h, w)
    indices = indices[None]                                           # (1, 3, h, w)
    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
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
        self.kwargs = {'stride': stride, 'padding': padding}

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++ BEV Embedding ++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class BEVEmbedding(nn.Module):
    def __init__(self, dim, sigma, bev_height, bev_width,
                 h_meters, w_meters, offset, upsample_scales: list):
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
            sigma * torch.randn(dim,
                                bev_height // upsample_scales[0],
                                bev_width // upsample_scales[0])
        )

    def get_prior(self):
        return self.learned_features


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++ Cross View Swap Attention ++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class CrossViewSwapAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, bev_feat, img_feat, object_count=None):
        """
        bev_feat: (B, C, H_bev, W_bev)
        img_feat: (B, C, H_img, W_img)
        """
        B, C, H_bev, W_bev = bev_feat.shape
        _, _, H_img, W_img = img_feat.shape

        q = self.q_proj(rearrange(bev_feat, "b c h w -> b (h w) c"))
        k = self.k_proj(rearrange(img_feat, "b c h w -> b (h w) c"))
        v = self.v_proj(rearrange(img_feat, "b c h w -> b (h w) c"))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        attn = einsum("b h n d, b h m d -> b h n m", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h n m, b h m d -> b h n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)

        out = rearrange(out, "b (h w) c -> b c h w", h=H_bev, w=W_bev)
        return out


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++ 최종 모델 ++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class BEVModel(nn.Module):
    def __init__(self, dim=128, bev_h=50, bev_w=50, img_h=200, img_w=200,
                 h_meters=100.0, w_meters=100.0, offset=0.0):
        super().__init__()
        self.bev_embed = BEVEmbedding(
            dim=dim, sigma=0.01,
            bev_height=bev_h, bev_width=bev_w,
            h_meters=h_meters, w_meters=w_meters, offset=offset,
            upsample_scales=[1]
        )
        self.cross_attn = CrossViewSwapAttention(dim)
        self.decoder = DecoderBlock(dim, dim)

        feat_height, feat_width = img_h, img_w
        image_plane = generate_grid(feat_height, feat_width)  # 수정됨 (차원 문제 해결)
        self.register_buffer("image_plane", image_plane, persistent=False)

    def forward(self, img_feat, object_count=None):
        bev_prior = self.bev_embed.get_prior()
        bev_feat = bev_prior.unsqueeze(0).repeat(img_feat.size(0), 1, 1, 1)

        bev_feat = self.cross_attn(bev_feat, img_feat, object_count)
        bev_feat = self.decoder(bev_feat)

        return bev_feat
