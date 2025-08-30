import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List, Optional

# --- New Feature: Loss Functions (for use in training script) ---
# 요청하신 Focal Loss와 Dice Loss를 여기에 정의하여 학습 스크립트에서 바로 사용할 수 있도록 합니다.
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

# --- New Feature: Squeeze-and-Excitation (SE) Module ---
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BottleneckWithSE(Bottleneck):
    """ Bottleneck with SE module """
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckWithSE, self).__init__(inplanes, planes, stride, downsample, groups,
                                               base_width, dilation, norm_layer)
        # Add SE layer at the end of the bottleneck
        self.se = SELayer(self.expansion * planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.se(out) # Apply SE module

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

ResNetBottleNeck = lambda c: BottleneckWithSE(c, c // 4)

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

# --- New Feature: Padding Recovery Utility ---
def unpad_to_size(x, h, w):
    """Crops the tensor x to the original size (h, w)."""
    _, _, H, W = x.shape
    if H > h:
        x = x[:, :, :h, :]
    if W > w:
        x = x[:, :, :, :w]
    return x

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
        # This method provides the learnable BEV prior, enabling warm-start strategies.
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
    def __init__(self, dim, heads, dim_head, qkv_bias, temperature=1.0, rel_pos_emb=False, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5
        
        # --- New Feature: Attention Temperature (τ) ---
        # The temperature parameter 'τ' for scaling attention scores.
        # Can be adjusted during training (e.g., annealing schedule).
        self.temperature = temperature
        
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

        # --- Apply Attention Temperature ---
        dot = dot / self.temperature

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
        attention_temperature: float = 1.0, # Pass temperature here
        rel_pos_emb: bool = False,
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

        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias, temperature=attention_temperature)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias, temperature=attention_temperature)
        self.skip = skip
        
        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def pad_divisible(self, x, win_h, win_w):
        """Pad the x to be divisible by window size."""
        b, n, c, h, w = x.shape
        
        # Calculate padding needed
        padh = (win_h - h % win_h) % win_h
        padw = (win_w - w % win_w) % win_w
        
        # Apply padding to the bottom and right
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
        _, _, H, W = x.shape # Original BEV height and width

        pixel = self.image_plane                                              # 1 1 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]                                                   # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]              # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                      # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                 # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                              # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                    # b n 4 (h w)
        d = E_inv @ cam                                                       # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)          # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                                      # (b n) d h w

        img_embed = d_embed - c_embed                                         # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)   # (b n) d h w

        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        elif index == 3:
            world = bev.grid3[:2]

        if self.bev_embed_flag:
            w_embed = self.bev_embed(world[None])                             # 1 d H W
            bev_embed = w_embed - c_embed                                     # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7) # (b n) d H W
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n) # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')             # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)            # (b n) d h w
        else:
            key_flat = img_embed                                              # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                          # (b n) d h w

        # Expand + refine the BEV embedding
        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]  # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)           # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)           # b n d h w

        # --- Padding/Unpadding Logic ---
        # Pad key and val to be divisible by window size for attention
        key = self.pad_divisible(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisible(val, self.feat_win_size[0], self.feat_win_size[1])

        # local-to-local cross-attention
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition
        
        query_out_local = rearrange(self.cross_win_attend_1(query, key, val,
                                  skip=rearrange(x,
                                                 'b d (x w1) (y w2) -> b x y w1 w2 d',
                                                 w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None),
                               'b x y w1 w2 d  -> b d (x w1) (y w2)')  # reverse window to feature
        
        # Unpad to original size
        query_out_local = unpad_to_size(query_out_local, H, W)
        query_out_local = rearrange(query_out_local, 'b d H W -> b H W d')
        
        query_out_local = query_out_local + self.mlp_1(self.prenorm_1(query_out_local))

        x_skip = query_out_local
        query_global = repeat(query_out_local, 'b x y d -> b n x y d', n=n)         # b n x y d

        # local-to-global cross-attention
        query_global = rearrange(query_global, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')    # reverse window to feature
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')    # reverse window to feature
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                       w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition
        
        query_out_global = rearrange(self.cross_win_attend_2(query_global,
                                          key,
                                          val,
                                          skip=rearrange(x_skip,
                                                         'b (x w1) (y w2) d -> b x y w1 w2 d',
                                                         w1=self.q_win_size[0],
                                                         w2=self.q_win_size[1])
                                          if self.skip else None),
                               'b x y w1 w2 d  -> b (x w1) (y w2) d')  # reverse grid to feature
        
        query_out_global = query_out_global + self.mlp_2(self.prenorm_2(query_out_global))
        query_out_global = self.postnorm(query_out_global)
        query_out_global = rearrange(query_out_global, 'b H W d -> b d H W')

        return query_out_global

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
            # --- New Feature: Augmentation Params ---
            use_camera_dropout: bool = True,
            camera_dropout_p: float = 0.1,
            use_calibration_noise: bool = True,
            calibration_noise_std: float = 0.05,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)
        
        # Store augmentation parameters
        self.use_camera_dropout = use_camera_dropout
        self.camera_dropout_p = camera_dropout_p
        self.use_calibration_noise = use_calibration_noise
        self.calibration_noise_std = calibration_noise_std

        cross_views = list()
        layers = list()
        downsample_layers = list()

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone.output_shapes, middle)):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim, dim[i], i, **cross_view, **cross_view_swap)
            cross_views.append(cva)

            # --- Use Bottleneck with SE module ---
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

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)      # b n c h w
        I_inv = batch['intrinsics'].inverse()     # b n 3 3
        E_inv = batch['extrinsics'].inverse()     # b n 4 4

        # --- New Feature: Augmentations (applied during training only) ---
        if self.training:
            # Apply Calibration Noise
            if self.use_calibration_noise:
                # Add noise to extrinsic translation and rotation
                ext_noise = torch.randn(b, n, 4, 4, device=E_inv.device) * self.calibration_noise_std
                # Ensure the last row remains [0, 0, 0, 1]
                ext_noise[..., 3, :] = 0
                E_inv = E_inv + ext_noise

                # Add noise to intrinsic focal length and principal point
                int_noise = torch.randn(b, n, 3, 3, device=I_inv.device) * self.calibration_noise_std
                # Ensure the last row remains [0, 0, 1] and other entries are 0
                int_noise[..., :, 2] = 0
                int_noise[..., 2, :] = 0
                I_inv = I_inv + int_noise

            # Apply Camera Dropout
            if self.use_camera_dropout:
                # Create a mask for cameras to keep. (b, n, 1, 1, 1)
                camera_mask = (torch.rand(b, n, 1, 1, 1, device=image.device) > self.camera_dropout_p).float()
        
        features_unaug = [self.down(y) for y in self.backbone(self.norm(image))]
        
        # Apply camera dropout mask to features
        if self.training and self.use_camera_dropout:
            features = []
            for feat in features_unaug:
                # Reshape feature and apply mask
                feat_reshaped = rearrange(feat, '(b n) c h w -> b n c h w', b=b, n=n)
                feat_masked = feat_reshaped * camera_mask
                features.append(rearrange(feat_masked, 'b n c h w -> (b n) c h w'))
        else:
            features = features_unaug
            
        x = self.bev_embedding.get_prior()        # d H W
        x = repeat(x, '... -> b ...', b=b)        # b d H W

        for i, (cross_view, feature, layer) in \
                enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)

        return x


if __name__ == "__main__":
    import os
    import re
    import yaml
    def load_yaml(file):
        with open(file, 'r') as stream:
            loader = yaml.SafeLoader
            param = yaml.load(stream, Loader=loader)
        return param

    # Example Usage (requires a backbone and config file)
    print("Code structure is updated with requested features.")
    print("To run a full test, please provide a backbone model and a configuration file.")

    # Example of how you might initialize the model with new params
    # config = load_yaml('path/to/your/config.yaml')
    # model_params = config['model']['params']
    
    # # Add new augmentation params to config or directly here
    # model_params['use_camera_dropout'] = True
    # model_params['camera_dropout_p'] = 0.2
    # model_params['use_calibration_noise'] = True
    # model_params['calibration_noise_std'] = 0.05
    
    # # Add attention temperature to config
    # # config['model']['cross_view_swap']['attention_temperature'] = 2.0
    
    # # backbone = YourBackbone(...)
    # # encoder = PyramidAxialEncoder(backbone, **model_params)
    
    # print("Model initialized with new features (SE, Augmentations, Attention Temp).")
