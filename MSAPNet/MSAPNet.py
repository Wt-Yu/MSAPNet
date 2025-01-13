import torch
import torch.nn as nn
from torch import Tensor

import random
import time
import math
from functools import partial
from typing import Optional, Callable
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, lecun_normal_
from .VSSBlock import VSSBlock
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, C, H, W)

        x = x.permute(0, 3, 1, 2).contiguous()  # DWconv
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class MambaBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        input = input.permute(0, 2, 3, 1)
        x = self.self_attention(self.ln_1(input))
        x = input + self.drop_path(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

        return out


class mamba_block(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, depth=2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.axis1_1 = nn.Conv2d(in_features, hidden_features, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.axis1_2 = nn.Conv2d(in_features, hidden_features, kernel_size=(11, 1), stride=1, padding=(5, 0))
        self.axis1_3 = nn.Conv2d(in_features, hidden_features, kernel_size=(21, 1), stride=1, padding=(10, 0))

        self.axis2_1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.axis2_2 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 11), stride=1, padding=(0, 5))
        self.axis2_3 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 21), stride=1, padding=(0, 10))

        self.norm = nn.BatchNorm2d(in_features)

        self.fc1 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=1)
        self.blocks = VSSBlock(hidden_dim=in_features)
        # self.blocks2= MambaLayer(out_features)
        self.blocks2 = MambaBlock(hidden_dim=out_features)

        self.fc3 = nn.Conv2d(out_features * 2, out_features, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.act = nn.GELU()

    def forward(self, x):
        x_n = self.norm(x)
        x_mamba_h, x_mamba_w = self.blocks(x_n)

        x_1 = self.fc1(self.axis1_1(x_n) + self.axis1_2(x_n) + self.axis1_3(x_n))

        x_h = x_mamba_h * self.sigmoid(x_1)

        x_2 = self.fc2(self.axis2_1(x_n) + self.axis2_2(x_n) + self.axis2_3(x_n))
        x_w = x_mamba_w * self.sigmoid(x_2)

        x_mamba = self.act(self.fc3(torch.cat((x_h, x_w), dim=1)))

        out = self.blocks2(x_mamba)
        return out


class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation funtion
    normalization includes BN and IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0,
                 groups=1, dilation=1, bias=False, norm=nn.BatchNorm2d, act=nn.GELU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x):

        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm=nn.BatchNorm2d, act=nn.GELU, preact=False):
        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, 3, stride=1, padding=1, norm=norm, act=act, preact=preact)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, 1, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += self.shortcut(residual)

        return out


class upConv_Ps(nn.Module):
    def __init__(self, in_ch, out_ch, num_block=1, scale_factor=None, block=BasicBlock):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        block_list = []
        block_list.append(block(2 * out_ch, out_ch))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out


# stem操作
class Stem(nn.Module):

    def __init__(self, in_chans=1, out_chans=32):
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.GELU()
        )

        self.proj2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.GELU()
        )

    def forward(self, x):
        x_skip = self.proj1(x)

        x = self.proj2(x_skip) + x_skip

        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x)


class ConvMLP(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, out_features=None, pool_size=3, depth=2):
        super().__init__()
        self.pool = Pooling(pool_size)

        out_features = out_features or in_features

        self.layers = nn.ModuleList(
            [
                BasicBlock(in_ch=in_features, out_ch=out_features)
                for i in range(depth)
            ]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(x)

        for layer in self.layers:
            x = layer(x)
        return x


class mamba_encoder_block(nn.Module):
    def __init__(self, in_dim, out_dim, pool_size=3, depth=2):
        super().__init__()
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1)
        )

        self.sigmoid = nn.Sigmoid()

        self.matchblock = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.layers1 = ConvMLP(in_features=out_dim, pool_size=pool_size)

        self.layers2 = mamba_block(out_dim)

    def forward(self, x):
        x = self.max_pool(x)

        template = self.matchblock(x)
        x_1 = self.layers1(x)

        x_1 = x_1 + x_1 * template

        feature_out = self.layers2(x_1)

        return feature_out





class conv_encoder_block(nn.Module):
    def __init__(self, in_dim, out_dim, pool_size=3, depth=2):
        super().__init__()

        self.max_pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1)
        )

        self.layers1 = ConvMLP(in_features=out_dim, pool_size=pool_size)
        self.layers2 = nn.ModuleList(
            [
                BasicBlock(in_ch=out_dim, out_ch=out_dim)
                for i in range(depth)
            ]
        )

        self.matchblock = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.max_pool(x)

        template = self.matchblock(x)
        x_1 = self.layers1(x)

        feature_out = x_1 + x_1 * template

        for layer in self.layers2:
            feature_out = layer(x_1)

        return feature_out


class MambaBlock_encoder(nn.Module):
    def __init__(self, input_channels=3, dims=[32, 64, 128, 256, 512], pool_size=3, depths=[2, 2, 2, 2],
                 num_classes=128):
        super().__init__()

        self.stem = Stem(in_chans=input_channels, out_chans=dims[0])
        self.encoder0 = conv_encoder_block(in_dim=dims[0], out_dim=dims[1], pool_size=pool_size, depth=depths[0])
        self.encoder1 = conv_encoder_block(in_dim=dims[1], out_dim=dims[2], pool_size=pool_size, depth=depths[1])
        self.encoder2 = mamba_encoder_block(in_dim=dims[2], out_dim=dims[3], pool_size=pool_size, depth=depths[2])
        self.encoder3 = mamba_encoder_block(in_dim=dims[3], out_dim=dims[4], pool_size=pool_size, depth=depths[3])
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x_stem = self.stem(x)
        x1 = self.encoder0(x_stem)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        return x1, x2, x3, x4, x_stem


class MambaBlock_decoder(nn.Module):
    def __init__(self, dims=[32, 64, 128, 256, 512], num_classes=10, deep_supervision=True):
        super().__init__()

        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.up_layer0 = upConv_Ps(in_ch=dims[4], out_ch=dims[3], scale_factor=2)
        self.up_layer1 = upConv_Ps(in_ch=dims[3], out_ch=dims[2], scale_factor=2)
        self.up_layer2 = upConv_Ps(in_ch=dims[2], out_ch=dims[1], scale_factor=2)
        self.up_layer3 = upConv_Ps(in_ch=dims[1], out_ch=dims[0], scale_factor=2)

        self.final_conv = nn.Conv2d(dims[0], num_classes, kernel_size=1)

        self.out_1 = nn.Conv2d(dims[3], num_classes, kernel_size=1)
        self.out_2 = nn.Conv2d(dims[2], num_classes, kernel_size=1)
        self.out_3 = nn.Conv2d(dims[1], num_classes, kernel_size=1)
        self.out_4 = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def forward(self, x1, x2, x3, x4, x_skip):
        seg_out = []

        skip1 = self.up_layer0(x4, x3)
        out1 = self.out_1(skip1)

        skip2 = self.up_layer1(skip1, x2)
        out2 = self.out_2(skip2)

        skip3 = self.up_layer2(skip2, x1)
        out3 = self.out_3(skip3)

        skip4 = self.up_layer3(skip3, x_skip)
        out = self.final_conv(skip4)

        seg_out.append(out1)
        seg_out.append(out2)
        seg_out.append(out3)
        seg_out.append(out)

        seg_outputs = seg_out[::-1]

        if self.deep_supervision:
            return seg_outputs
        else:
            return seg_outputs[0]


class MSAPNet(nn.Module):
    def __init__(self, input_channels=3, dims=[32, 64, 128, 256, 512], num_classes=10, depths=[2, 2, 2, 2],
                 deep_supervision=False):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.encoder = MambaBlock_encoder(input_channels=input_channels, dims=dims, depths=depths)

        self.decoder = MambaBlock_decoder(dims=dims, num_classes=num_classes, deep_supervision=deep_supervision)

    def forward(self, x):
        x1, x2, x3, x4, x_skip = self.encoder(x)

        return self.decoder(x1, x2, x3, x4, x_skip)


def get_MSAPNet_from_plans(plans_manager: PlansManager,
                               dataset_json: dict,
                               configuration_manager: ConfigurationManager,
                               num_input_channels: int,
                               deep_supervision: bool = False):


    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'MSAPNet'
    network_class = MSAPNet
    kwargs = {
        'MSAPNet': {
        }
    }

    model = network_class(
        input_channels=num_input_channels,
        **kwargs[segmentation_network_class_name]
    )

    return model