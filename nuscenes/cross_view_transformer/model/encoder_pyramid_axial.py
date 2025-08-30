import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional
from .decoder import DecoderBlock

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

# --------------------
# utils
# --------------------
def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)
    indices = indices[None]
    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters
    return [
        [0., -sw, w/2.],
        [-sh, 0., h*offset+h/2.],
        [0., 0., 1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)
    def forward(self, x):
        return (x - self.mean) / self.std


# --------------------
# BEV Embedding
# --------------------
class BEVEmbedding(nn.Module):
    def __init__(self, dim, sigma, bev_height, bev_width, h_meters, w_meters, offset, upsample_scales):
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
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, bev_height//upsample_scales[0], bev_width//upsample_scales[0]))
    def get_prior(self):
        return self.learned_features


# --------------------
# Cross Window Attention (수정됨)
# --------------------
class CrossWinAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm):
        super().__init__()
        self.base_scale = dim_head ** -0.5  # [MOD]
        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb
        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.proj = nn.Linear(heads * dim_head, dim)

    def forward(self, q, k, v, skip=None, temperature: Optional[torch.Tensor] = None):  # [MOD]
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

        # [MOD] adaptive scaling with τ
        if temperature is None:
            tau = 1.0
        else:
            if temperature.dim() == 1:
                tau = temperature[:, None, None, None]
            else:
                tau = temperature
        bmh, *_ = q.shape
        m = self.heads
        b = bmh // m
        tau = tau.to(q.dtype).to(q.device)
        tau = tau.repeat_interleave(m, dim=0)
        q = q * (self.base_scale / tau)

        dot = torch.einsum('b l Q d, b l K d -> b l Q K', q, k)
        att = dot.softmax(dim=-1)
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, 'b (x y) (n w1 w2) d -> b n x y w1 w2 d', x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)
        z = self.proj(a)
        z = z.mean(1)
        if skip is not None:
            z = z + skip
        return z


# --------------------
# Cross View Swap Attention (수정됨)
# --------------------
class CrossViewSwapAttention(nn.Module):
    def __init__(self, feat_height, feat_width, feat_dim, dim, index, image_height, image_width, qkv_bias, q_win_size, feat_win_size, heads, dim_head, bev_embedding_flag, rel_pos_emb=False, no_image_features=False, skip=True, norm=nn.LayerNorm):
        super().__init__()
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)
        self.feature_linear = nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        self.feature_proj = None if no_image_features else nn.Sequential(nn.BatchNorm2d(feat_dim), nn.ReLU(), nn.Conv2d(feat_dim, dim, 1, bias=False))
        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.q_win_size = q_win_size[index]
        self.feat_win_size = feat_win_size[index]
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)
        # [MOD] LN + gate
        self.img_ln = nn.LayerNorm(dim)
        self.cam_ln = nn.LayerNorm(dim)
        self.key_ln = nn.LayerNorm(dim)
        self.val_ln = nn.LayerNorm(dim)
        self.embed_gate = nn.Parameter(torch.tensor(1.0))

    # [MOD] pad/unpad
    def pad_divisble(self, x, win_h, win_w):
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h - 1)//win_h)*win_h, ((w + win_w - 1)//win_w)*win_w
        padh, padw = h_pad-h, w_pad-w
        x = F.pad(x, (0, padw, 0, padh), value=0)
        return x, (h, w)
    def unpad(self, x, orig_hw):
        oh, ow = orig_hw
        return x[..., :oh, :ow]

    def forward(self, index, x, bev, feature, I_inv, E_inv, object_count: Optional[torch.Tensor] = None):
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
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True)+1e-7)

        # world grid
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
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True)+1e-7)
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        if self.feature_proj is not None:
            key_flat = img_embed * self.embed_gate + self.feature_proj(feature_flat)  # [MOD]
        else:
            key_flat = img_embed * self.embed_gate
        val_flat = self.feature_linear(feature_flat)
        key_flat = rearrange(key_flat, '(b n) d h w -> b n d h w', b=b, n=n)
        val_flat = rearrange(val_flat, '(b n) d h w -> b n d h w', b=b, n=n)
        key_flat = self.key_ln(key_flat)
        val_flat = self.val_ln(val_flat)

        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]
        key, orig_hw_feat = self.pad_divisble(key_flat, self.feat_win_size[0], self.feat_win_size[1])
        val, _ = self.pad_divisble(val_flat, self.feat_win_size[0], self.feat_win_size[1])

        # [MOD] τ from object_count
        if object_count is not None:
            total_obj = object_count.sum(dim=-1) if object_count.dim()>1 else object_count
            a, b0, c0 = 2.0, 5.0, 0.6
            tau = a/(b0+total_obj.float().clamp(min=0))+c0
            tau = tau.clamp(0.4, 1.5)
        else:
            tau = None

        # local-to-local
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        out1 = self.cross_win_attend_1(query, key, val, skip=rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None, temperature=tau)
        query = rearrange(out1, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)

        # local-to-global
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1])
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d', w1=self.feat_win_size[0], w2=self.feat_win_size[1])
        out2 = self.cross_win_attend_2(query, key, val, skip=rearrange(x_skip, 'b (x w1) (y w2) d -> b x y w1 w2 d', w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None, temperature=tau)
        query = rearrange(out2, 'b x y w1 w2 d -> b (x w1) (y w2) d')
        query = query + self.mlp_2(self.prenorm_2(query))
        query = self.postnorm(query)
        query = rearrange(query, 'b H W d -> b d H W')
        query = self.unpad(query, (H, W))  # [MOD] unpad
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

        image = batch['image'].flatten(0, 1)            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        # ✅ 여기서 object_count 가져오기
        object_count = batch.get('object_count', None)

        #디버깅
        if object_count is not None:
            print(">> object_count(pyramid axial encoder):", object_count.shape, object_count) #각 인덱스가 특정 종류(차, 트럭, 보행자)의 객체 수임
        else:
            print(">> object_count(pyramid axial encoder) is None")
       
        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

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


if __name__ == "__main__":
    import os
    import re
    import yaml
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    block = CrossWinAttention(dim=128,
                              heads=4,
                              dim_head=32,
                              qkv_bias=True,)
    block.cuda()
    test_q = torch.rand(1, 6, 5, 5, 5, 5, 128)
    test_k = test_v = torch.rand(1, 6, 5, 5, 6, 12, 128)
    test_q = test_q.cuda()
    test_k = test_k.cuda()
    test_v = test_v.cuda()

    # test pad divisible
    # output = block.pad_divisble(x=test_data, win_h=6, win_w=12)
    output = block(test_q, test_k, test_v)
    print(output.shape)

    # block = CrossViewSwapAttention(
    #     feat_height=28,
    #     feat_width=60,
    #     feat_dim=128,
    #     dim=128,
    #     index=0,
    #     image_height=25,
    #     image_width=25,
    #     qkv_bias=True,
    #     q_win_size=[5, 5],
    #     feat_win_size=[6, 12],
    #     heads=[4,],
    #     dim_head=[32,],
    #     qkv_bias=True,)

    image = torch.rand(1, 6, 128, 28, 60)            # b n c h w
    I_inv = torch.rand(1, 6, 3, 3)           # b n 3 3
    E_inv = torch.rand(1, 6, 4, 4)           # b n 4 4

    feature = torch.rand(1, 6, 128, 25, 25)

    x = torch.rand(1, 128, 25, 25)                     # b d H W

    # output = block(0, x, self.bev_embedding, feature, I_inv, E_inv)
    block.cuda()

    ##### EncoderSwap
    params = load_yaml('config/model/cvt_pyramid_swap.yaml')

    print(params)

    batch = {}
    batch['image'] = image
    batch['intrinsics'] = I_inv
    batch['extrinsics'] = E_inv

    out = encoder(batch)

    print(out.shape)
