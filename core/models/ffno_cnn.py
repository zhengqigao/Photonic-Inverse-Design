"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-03 01:17:52
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-05 03:25:43
"""

from typing import List, Optional, Tuple

import numpy as np
from sympy import Identity
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.activation import Swish
from timm.models.layers import DropPath, to_2tuple
from torch import nn
from torch.functional import Tensor
from torch.types import Device
from torch.utils.checkpoint import checkpoint
from .constant import *
from .layers import FullFieldRegressionHead, PerfectlyMatchedLayerPad2d, PermittivityEncoder
from .layers.activation import SIREN
from .layers.ffno_conv2d import FFNOConv2d
from .pde_base import PDE_NN_BASE
from torch.types import _size
from .layers.layer_norm import MyLayerNorm

__all__ = ["FFNO2d"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "GELU",
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "siren":
            self.act_func = SIREN()
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x


class BSConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size = 3,
        stride: _size = 1,
        dilation: _size = 1,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [(dilation[i] * (kernel_size[i] - 1) + 1) // 2 for i in range(len(kernel_size))]
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=bias,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ResStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size = 3,
        stride: _size = 1,
        dilation: _size = 1,
        norm: str = "ln",
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [(dilation[i] * (kernel_size[i] - 1) + 1) // 2 for i in range(len(kernel_size))]

        # self.conv1 = nn.Conv2d(
        #     in_channels,
        #     out_channels // 2,
        #     kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     groups=groups,
        #     bias=bias,
        # )
        self.conv1 = BSConv2d(
            in_channels,
            out_channels // 2,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        if norm == "bn":
            self.norm1 = nn.BatchNorm2d(out_channels // 2)
        elif norm == "ln":
            self.norm1 = MyLayerNorm(out_channels // 2, data_format="channels_first")
        else:
            raise ValueError(f"Norm type {norm} not supported")
        self.act1 = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv2d(
        #     out_channels // 2,
        #     out_channels,
        #     kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     groups=groups,
        #     bias=bias,
        # )
        self.conv2 = BSConv2d(
            out_channels // 2,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        if self.norm == "bn":
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif self.norm == "ln":
            self.norm2 = MyLayerNorm(out_channels, data_format="channels_first")
        else:
            raise ValueError(f"Norm type {norm} not supported")
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class FFNO2dBlock(nn.Module):
    expansion = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "GELU",
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        with_cp=False,
        ffn: bool = True,
        ffn_dwconv: bool = True,
        aug_path: bool = True,
        norm: str = "ln",
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        # self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.f_conv = FFNOConv2d(in_channels, out_channels, n_modes, device=device)
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
            self.pre_norm = nn.BatchNorm2d(in_channels)
        elif norm == "ln":
            self.norm = MyLayerNorm(out_channels, data_format="channels_first")
            self.pre_norm = MyLayerNorm(in_channels, data_format="channels_first")
        self.with_cp = with_cp
        # self.norm.weight.data.zero_()
        if ffn:
            if ffn_dwconv:
                self.ff = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels * self.expansion, 1),
                    nn.Conv2d(
                        out_channels * self.expansion,
                        out_channels * self.expansion,
                        3,
                        groups=out_channels * self.expansion,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels * self.expansion) if norm == "bn" else MyLayerNorm(out_channels * self.expansion, data_format="channels_first"),
                    nn.GELU(),
                    nn.Conv2d(out_channels * self.expansion, out_channels, 1),
                )
            else:
                self.ff = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels * self.expansion, 1),
                    nn.BatchNorm2d(out_channels * self.expansion) if norm == "bn" else MyLayerNorm(out_channels * self.expansion, data_format="channels_first"),
                    nn.GELU(),
                    nn.Conv2d(out_channels * self.expansion, out_channels, 1),
                )
        else:
            self.ff = None
        if aug_path:
            self.aug_path = nn.Sequential(BSConv2d(in_channels, out_channels, 3), nn.GELU())
        else:
            self.aug_path = None
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "siren":
            self.act_func = SIREN()
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        def _inner_forward(x):
            y = x
            # x = self.norm(self.ff(self.f_conv(self.pre_norm(x))))
            if self.ff is not None:
                x = self.norm(self.ff(self.pre_norm(self.f_conv(x))))
                # x = self.norm(self.ff(self.act_func(self.pre_norm(self.f_conv(x)))))
                # x = self.norm(self.ff(self.act_func(self.pre_norm(self.f_conv(x)))))
                x = self.drop_path(x) + y
            else:
                x = self.act_func(self.drop_path(self.norm(self.f_conv(x))) + y)
            if self.aug_path is not None:
                x = x + self.aug_path(y)
            return x

        # def _inner_forward(x):
        #     x = self.drop_path(self.pre_norm(self.f_conv(x))) + self.aug_path(x)
        #     y = x
        #     x = self.norm(self.ff(x))
        #     x = self.drop_path2(x) + y
        #     return x

        if x.requires_grad and self.with_cp:
            return checkpoint(_inner_forward, x)
        else:
            return _inner_forward(x)


class FFNO2d(PDE_NN_BASE):
    """
    Frequency-domain scattered electric field envelop predictor
    Assumption:
    (1) TE10 mode, i.e., Ey(r, omega) = Ez(r, omega) = 0
    (2) Fixed wavelength. wavelength currently not being modeled
    (3) Only predict Ex_scatter(r, omega)

    Args:
        PDE_NN_BASE ([type]): [description]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dim: int = 16,
        kernel_list: List[int] = [72, 72, 72, 72],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [512],
        mode_list: List[Tuple[int]] = [(128, 129), (128, 129), (128, 129), (128, 129)],
        act_func: Optional[str] = "GELU",
        domain_size: Tuple[float] = [20, 100],  # computation domain in unit of um
        grid_step: float = 1.550
        / 20,  # grid step size in unit of um, typically 1/20 or 1/30 of the wavelength
        pml_width: float = 0,
        pml_permittivity: complex = 0 + 0j,
        buffer_width: float = 0.5,
        buffer_permittivity: complex = -1e-10 + 0j,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        eps_min: float = 2.085136,
        eps_max: float = 12.3,
        aux_head: bool = False,
        aux_head_idx: int = 1,
        pos_encoding: str = "none",
        with_cp=False,
        conv_stem: bool = False,
        aug_path: bool = True,
        ffn: bool = True,
        ffn_dwconv: bool = True,
        encoder_cfg: dict = {},
        backbone_cfg: dict = {},
        decoder_cfg: dict = {},
        in_frames: int = 0,
        **kwargs,
    ):
        super().__init__(
            encoder_cfg=encoder_cfg,
            backbone_cfg=backbone_cfg,
            decoder_cfg=decoder_cfg,
            **kwargs,
        )

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_frames = in_frames
        assert (
            out_channels % 2 == 0
        ), f"The output channels must be even number larger than 2, but got {out_channels}"
        self.dim = dim
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.hidden_list = hidden_list
        self.mode_list = mode_list
        self.act_func = act_func
        self.domain_size = domain_size
        self.grid_step = grid_step
        self.domain_size_pixel = [round(i / grid_step) for i in domain_size]
        self.buffer_width = buffer_width
        self.buffer_permittivity = buffer_permittivity
        self.pml_width = pml_width
        self.pml_permittivity = pml_permittivity
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.aux_head = aux_head
        self.aux_head_idx = aux_head_idx
        self.pos_encoding = pos_encoding
        self.with_cp = with_cp
        self.conv_stem = conv_stem
        self.aug_path = aug_path
        self.ffn = ffn
        self.ffn_dwconv = ffn_dwconv
        if pos_encoding == "none":
            pass
        elif pos_encoding == "linear":
            self.in_channels += 2
        elif pos_encoding == "exp":
            self.in_channels += 4
        elif pos_encoding == "exp_noeps":
            self.in_channels += 2
        elif pos_encoding == "exp3":
            self.in_channels += 6
        elif pos_encoding == "exp4":
            self.in_channels += 8
        elif pos_encoding in {"exp_full", "exp_full_r"}:
            self.in_channels += 7
        elif pos_encoding == "raw":
            self.in_channels += 3
        else:
            raise ValueError(f"pos_encoding only supports linear and exp, but got {pos_encoding}")

        self.device = device

        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()
        self.reset_parameters()

        self.permittivity_encoder = None
        self.set_trainable_permittivity(False)
        self.set_linear_probing_mode(False)

    def build_layers(self):
        if self.conv_stem:
            self.stem = ResStem(
                self.in_channels,
                self.dim,
                kernel_size=3,
                stride=1,
            )
        else:
            self.stem = nn.Conv2d(self.in_channels, self.dim, 1)
        kernel_list = [self.dim] + self.kernel_list
        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))

        features = [
            FFNO2dBlock(
                inc,
                outc,
                n_modes,
                kernel_size,
                padding,
                act_func=self.act_func,
                drop_path_rate=drop,
                device=self.device,
                with_cp=self.with_cp,
                aug_path=self.aug_path,
                ffn=self.ffn,
                ffn_dwconv=self.ffn_dwconv,
            )
            for inc, outc, n_modes, kernel_size, padding, drop in zip(
                kernel_list[:-1],
                kernel_list[1:],
                self.mode_list,
                self.kernel_size_list,
                self.padding_list,
                drop_path_rates,
            )
        ]
        self.features = nn.Sequential(*features)
        hidden_list = [self.kernel_list[-1]] + self.hidden_list
        head = [
            nn.Sequential(
                ConvBlock(inc, outc, kernel_size=1, padding=0, act_func=self.act_func, device=self.device),
                nn.Dropout2d(self.dropout_rate),
            )
            for inc, outc in zip(hidden_list[:-1], hidden_list[1:])
        ]
        # 2 channels as real and imag part of the TE field
        head += [
            ConvBlock(
                hidden_list[-1],
                self.out_channels,
                kernel_size=1,
                padding=0,
                act_func=None,
                device=self.device,
            )
        ]

        self.head = nn.Sequential(*head)

        if self.aux_head:
            hidden_list = [self.kernel_list[self.aux_head_idx]] + self.hidden_list
            head = [
                nn.Sequential(
                    ConvBlock(
                        inc, outc, kernel_size=1, padding=0, act_func=self.act_func, device=self.device
                    ),
                    nn.Dropout2d(self.dropout_rate),
                )
                for inc, outc in zip(hidden_list[:-1], hidden_list[1:])
            ]
            # 2 channels as real and imag part of the TE field
            head += [
                ConvBlock(
                    hidden_list[-1],
                    self.out_channels // 2,
                    kernel_size=1,
                    padding=0,
                    act_func=None,
                    device=self.device,
                )
            ]

            self.aux_head = nn.Sequential(*head)
        else:
            self.aux_head = None

    def set_trainable_permittivity(self, mode: bool = True) -> None:
        self.trainable_permittivity = mode

    def init_trainable_permittivity(
        self,
        regions: Tensor,
        valid_range: Tuple[int],
    ):
        self.permittivity_encoder = PermittivityEncoder(
            size=self.domain_size_pixel,
            regions=regions,
            valid_range=valid_range,
            device=self.device,
        )

    def requires_network_params_grad(self, mode: float = True) -> None:
        params = (
            self.stem.parameters()
            + self.features.parameters()
            + self.head.parameters()
            + self.full_field_head.parameters()
        )
        for p in params:
            p.requires_grad_(mode)

    def observe_waveprior(self, x: Tensor, wavelength: Tensor, grid_step: Tensor):
        epsilon = (
            x[:, 0:1] * (self.eps_max - self.eps_min) + self.eps_min
        )  # this is de-normalized permittivity

        # convert complex permittivity/mode to real numbers
        x = torch.view_as_real(x).permute(0, 1, 4, 2, 3).flatten(1, 2)  # [bs, inc*2, h, w] real

        # positional encoding
        grid = self.get_grid(
            x.shape,
            x.device,
            mode=self.pos_encoding,
            epsilon=epsilon,
            wavelength=wavelength,
            grid_step=grid_step,
        )
        return grid

    def observe_stem_output(self, x: Tensor, wavelength: Tensor, grid_step: Tensor):
        epsilon = (
            x[:, 0:1] * (self.eps_max - self.eps_min) + self.eps_min
        )  # this is de-normalized permittivity

        # convert complex permittivity/mode to real numbers
        x = torch.view_as_real(x).permute(0, 1, 4, 2, 3).flatten(1, 2)  # [bs, inc*2, h, w] real

        # positional encoding
        grid = self.get_grid(
            x.shape,
            x.device,
            mode=self.pos_encoding,
            epsilon=epsilon,
            wavelength=wavelength,
            grid_step=grid_step,
        )  # [bs, 2 or 4 or 8, h, w] real

        if grid is not None:
            x = torch.cat((x, grid), dim=1)  # [bs, inc*2+4, h, w] real
        return self.stem(x)

    def forward(
        self,
        x,
        src_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # x [bs, inc, h, w] complex
        # wavelength [bs, 1] real
        # grid_step [bs, 2] real
        # epsilon = (
        #     x[:, 0:1] * (self.eps_max - self.eps_min) + self.eps_min
        # )  # this is de-normalized permittivity
        epsilon = x[:, 0:1]
        input_fields = x[:, 1 : 1 + self.in_frames]
        src = x[:, 1 + self.in_frames :, ...]

        normalization_factor = []
        outs = []

        abs_values = torch.abs(input_fields)
        scaling_factor = (
            abs_values.amax(dim=(1, 2, 3), keepdim=True) + 1e-6
        )  # bs, 1, 1, 1

        x = torch.cat(
            [epsilon, input_fields / scaling_factor, src / scaling_factor], dim=1
        )

        # convert complex permittivity/mode to real numbers
        # if "noeps" in self.pos_encoding:  # no epsilon
        #     x = torch.view_as_real(x[:, 1:]).permute(0, 1, 4, 2, 3).flatten(1, 2)  # [bs, inc*2, h, w] real
        # else:
        #     x = torch.view_as_real(x).permute(0, 1, 4, 2, 3).flatten(1, 2)  # [bs, inc*2, h, w] real

        # positional encoding
        # grid = self.get_grid(
        #     x.shape,
        #     x.device,
        #     mode=self.pos_encoding,
        #     epsilon=epsilon,
        #     wavelength=wavelength,
        #     grid_step=grid_step,
        # )  # [bs, 2 or 4 or 8, h, w] real

        # if grid is not None:
        #     x = torch.cat((x, grid), dim=1)  # [bs, inc*2+4, h, w] real

        if self.linear_probing_mode:
            with torch.no_grad():
                # DNN-based electric field envelop prediction
                x = self.stem(x)
                x = self.features(x)
        else:
            x = self.stem(x)
            x = self.features(x)
        x = self.head(x)  # [bs, outc, h, w] real

        src_mask = src_mask >= 0.5
        x = torch.where(src_mask, src[:, -self.out_channels:, ...]/scaling_factor, x)
        x = x*padding_mask

        normalization_factor.append(scaling_factor)
        outs.append(x)

        for i in range(len(normalization_factor)):
            normalization_factor[i] = normalization_factor[i].repeat(
                1, self.out_channels, 1, 1
            )
        normalization_factor = torch.cat(normalization_factor, dim=1)
        outs = torch.cat(outs, dim=1)
        # convert to complex frequency-domain electric field envelops
        # x = torch.view_as_complex(
        #     x.view(x.size(0), -1, 2, x.size(-2), x.size(-1)).permute(0, 1, 3, 4, 2).contiguous()
        # )  # [bs, outc/2, h, w] complex
        return outs, normalization_factor
