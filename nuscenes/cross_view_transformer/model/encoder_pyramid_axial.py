import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List, Optional
from .decoder import DecoderBlock

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

# -------------------------
# Utility Functions
# -------------------------
def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)  # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)  # 3 h w
    indices = indices[None]  # 1 3 h w
    return indices

def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters
    return [[0., -sw, w/2.],
            [-sh, 0., h*offset+h/2.],
            [0., 0., 1.]]

# -------------------------
# Normalization
# -------------------------
class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)
    def forward(self, x):
        return (x - self.mean) / self.std

# -------------------------
# BEV Embedding
# -------------------------
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
            self.register_buffer('grid%d'%i, grid, persistent=False)
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, bev_height//upsample_scales[0], bev_width//upsample_scales[0])
        )
    def get_prior(self):
        return self.learned_features

# -------------------------
# Attention
# -------------------------
class Attention(nn.Module):
    def __init__(self, dim, dim_head = 32, dropout = 0., window_size = 25):
        super().__init__()
        assert (dim % dim_head) == 0
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        self.rel_pos_bias = nn.Embedding((2*window_size-1)**2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2*window_size-1,1])).sum(dim=-1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)
    def forward(self, x):
        b, _, h, w, device, heads = *x.shape, x.device, self.heads
        x = rearrange(x, 'b d h w -> b (h w) d')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b m (h w) d -> b h w (m d)', h=h, w=w)
        out = self.to_out(out)
        return rearrange(out, 'b h w d -> b d h w')

# -------------------------
# CrossViewSwapAttention
# -------------------------
class CrossViewSwapAttention(nn.Module):
    def __init__(self, feat_height, feat_width, feat_dim, dim, index, image_height, image_width,
                 qkv_bias, q_win_size, feat_win_size, heads, dim_head, bev_embedding_flag, rel_pos_emb=False,
                 no_image_features=False, skip=True, norm=nn.LayerNorm):
        super().__init__()
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)
        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim), nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False)
        )
        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim), nn.ReLU(),
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
        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.postnorm = norm(dim)

    def pad_divisble(self, x, win_h, win_w):
        _, _, _, h, w = x.shape
        h_pad = ((h + win_h) // win_h) * win_h
        w_pad = ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def forward(self, index, x, bev, feature, I_inv, E_inv, object_count=None):
        b, n, _, _, _ = feature.shape
        # -----------------------------
        # Dynamic attention scaling based on object_count
        # -----------------------------
        if object_count is not None:
            obj_scale = object_count.float().mean(dim=1, keepdim=True) / (object_count.float().mean() + 1e-6)
            scale_factor = obj_scale.clamp(0.5, 2.0)  # Scene complexity-aware
        else:
            scale_factor = 1.0
        # Feature fusion
        feature_flat = rearrange(feature, 'b n d h w -> b (n d) h w')
        feature_flat = feature_flat * scale_factor.view(b,1,1,1)
        val_flat = self.feature_linear(feature_flat)
        # BEV embedding addition
        x = x + val_flat.mean(1, keepdim=False)  # soft voting base-level
        # -----------------------------
        # Decoder can be applied outside this module
        # -----------------------------
        return x

# -------------------------
# PyramidAxialEncoder + Decoder integrated
# -------------------------
class PyramidAxialBEV(nn.Module):
    def __init__(self, backbone, cross_view: dict, cross_view_swap: dict,
                 bev_embedding: dict, self_attn: dict, dim: list, middle: List[int] = [2,2],
                 scale: float = 1.0, decoder_params: dict = None):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone
        self.scale = scale
        self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False) if scale<1.0 else (lambda x: x)
        assert len(self.backbone.output_shapes) == len(middle)
        self.cross_views = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape
            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            self.cross_views.append(cva)
            self.layers.append(nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)]))
        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        # Decoder ensemble block
        self.decoder = DecoderBlock(**decoder_params)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape
        image = batch['image'].flatten(0,1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()
        object_count = batch.get('object_count', None)
        features = [self.down(y) for y in self.backbone(self.norm(image))]
        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)
        # Encoder
        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv, object_count)
            x = layer(x)
        # Decoder + Dynamic Ensemble
        x = self.decoder(x, object_count)  # Dynamic ensemble inside decoder
        return x

# -------------------------
# Example usage
# -------------------------
if __name__=="__main__":
    # Dummy input
    image = torch.rand(1,6,128,28,60)
    I_inv = torch.rand(1,6,3,3)
    E_inv = torch.rand(1,6,4,4)
    object_count = torch.randint(0,20,(1,6))  # batch x n
    batch = {'image': image, 'intrinsics': I_inv, 'extrinsics': E_inv, 'object_count': object_count}

    # Backbone dummy
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_shapes = [(1,128,25,25),(1,256,13,13)]
        def forward(self,x):
            return [torch.rand(x.shape[0],128,25,25), torch.rand(x.shape[0],256,13,13)]
    backbone = DummyBackbone()
    decoder_params = {'in_channels':128, 'out_channels':64}
    model = PyramidAxialBEV(backbone, cross_view={}, cross_view_swap={}, bev_embedding={'dim':128,'sigma':0.01,'bev_height':25,'bev_width':25,'h_meters':100,'w_meters':100,'offset':0,'upsample_scales':[1]}, self_attn={}, dim=[128,256], decoder_params=decoder_params)
    out = model(batch)
    print("Output shape:", out.shape)
