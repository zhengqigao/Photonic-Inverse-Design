"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-03 01:17:52
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-03 20:01:20
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.activation import Swish
from timm.models.layers import DropPath, to_2tuple
from torch import nn
from torch.functional import Tensor
from torch.types import Device

from .constant import *
from .layers import FullFieldRegressionHead, PerfectlyMatchedLayerPad2d, PermittivityEncoder
from .layers.activation import SIREN
from .layers.ffno_conv2d import FFNOConv2d
from .pde_base import PDE_NN_BASE
from torch.types import _size

__all__ = ["MSFFNO2d"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        norm=None,
        act_func: Optional[str] = "GELU",
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = norm
        if norm is not None:
            self.norm = nn.BatchNorm2d(out_channels)

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
        if self.norm is not None:
            x = self.norm(x)
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
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
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
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        torch.cuda.empty_cache()
        return x


class MSFFNO2dBlock(nn.Module):
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
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.f_conv = FFNOConv2d(in_channels, out_channels, n_modes, device=device)
        self.norm = nn.BatchNorm2d(out_channels)
        # self.norm.weight.data.zero_()
        self.ff = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.Conv2d(
                out_channels * self.expansion,
                out_channels * self.expansion,
                3,
                groups=out_channels * self.expansion,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.GELU(),
            nn.Conv2d(out_channels * self.expansion, out_channels, 1),
        )
        self.aug_path = BSConv2d(in_channels, out_channels, 3)
        # if act_func is None:
        #     self.act_func = None
        # elif act_func.lower() == "siren":
        #     self.act_func = SIREN()
        # elif act_func.lower() == "swish":
        #     self.act_func = Swish()
        # else:
        #     self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        y = x
        x = self.norm(self.ff(self.f_conv(x)))
        x = self.drop_path(x) + y + self.aug_path(y)
        return x


class MSHead(nn.Module):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(
        self,
        num_inputs: int,
        in_channels: List[int],
        hidden_dim: int,
        out_channels: int,
        interpolate_mode="bilinear",
        device=torch.device("cuda:0"),
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.interpolate_mode = interpolate_mode
        self.device = device
        self.build_layers()

    def build_layers(self):
        self.convs = nn.ModuleList()
        for i in range(self.num_inputs):
            self.convs.append(
                ConvBlock(
                    in_channels=self.in_channels[i],
                    out_channels=self.hidden_dim,
                    kernel_size=1,
                    stride=1,
                    norm="BN",
                    act_func="ReLU",
                    device=self.device,
                )
            )

        self.fusion_conv = ConvBlock(
            in_channels=self.hidden_dim * self.num_inputs,
            out_channels=self.hidden_dim,
            kernel_size=1,
            norm="BN",
            act_func="ReLU",
        )
        self.head = ConvBlock(
            in_channels=self.hidden_dim,
            out_channels=self.out_channels,
            kernel_size=1,
            norm=None,
            act_func=None,
        )

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1, 1/2, 1/4, 1/8
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                torch.nn.functional.interpolate(
                    conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                )
            )

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.head(out)

        return out


class MSFFNO2d(PDE_NN_BASE):
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
        stage_list: List[int] = [2, 4, 2],
        kernel_list: List[int] = [16, 32, 64],
        mode_list: List[Tuple[int]] = [(41, 193), (21, 87), (11, 44)],
        hidden_dim: int = 256,
        act_func: Optional[str] = "GELU",
        domain_size: Tuple[float] = [80, 384],  # computation domain in unit of um
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
        pos_encoding: str = "exp",
    ):
        super().__init__()

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
        assert (
            out_channels % 2 == 0
        ), f"The output channels must be even number larger than 2, but got {out_channels}"
        self.dim = dim
        self.stage_list = stage_list
        self.kernel_list = kernel_list
        self.hidden_dim = hidden_dim
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
        if pos_encoding == "linear":
            self.in_channels += 2
        elif pos_encoding == "exp":
            self.in_channels += 4
        elif pos_encoding == "exp3":
            self.in_channels += 6
        elif pos_encoding == "exp4":
            self.in_channels += 8
        else:
            raise ValueError(f"pos_encoding only supports linear and exp, but got {pos_encoding}")

        self.device = device

        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()
        self.reset_parameters()

        self.permittivity_encoder = None
        self.set_trainable_permittivity(False)

    def build_layers(self):
        self.stem = ResStem(
            self.in_channels,
            self.dim,
            kernel_size=3,
            stride=1,
        )
        kernel_list = [self.dim] + self.kernel_list
        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))

        features = []
        for i, n_blk in enumerate(self.stage_list):
            stage = [
                BSConv2d(
                    kernel_list[i],
                    kernel_list[i + 1],
                    stride=1 if i == 0 else 2,
                ),
                nn.GELU(),
            ]
            stage += [
                MSFFNO2dBlock(
                    kernel_list[i + 1],
                    kernel_list[i + 1],
                    self.mode_list[i],
                    kernel_size=3,
                    padding=1,
                    act_func=self.act_func,
                    drop_path_rate=drop_path_rates[i],
                    device=self.device,
                )
                for _ in range(n_blk)
            ]
            features.append(nn.Sequential(*stage))

        self.features = nn.Sequential(*features)

        self.head = MSHead(
            num_inputs=len(self.stage_list),
            in_channels=self.kernel_list,
            hidden_dim=self.hidden_dim,
            out_channels=self.out_channels,
            device=self.device,
        )

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

    def forward(self, x: Tensor, wavelength: Tensor, grid_step: Tensor) -> Tensor:
        # x [bs, inc, h, w] complex
        # wavelength [bs, 1] real
        # grid_step [bs, 2] real
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

        x = torch.cat((x, grid), dim=1)  # [bs, inc*2+4, h, w] real

        # DNN-based electric field envelop prediction
        x = self.stem(x)
        outs = []
        for stage in self.features:
            x = stage(x)
            outs.append(x)
        x = self.head(outs)  # [bs, outc, h, w] real
        # convert to complex frequency-domain electric field envelops
        x = torch.view_as_complex(
            x.view(x.size(0), -1, 2, x.size(-2), x.size(-1)).permute(0, 1, 3, 4, 2).contiguous()
        )  # [bs, outc/2, h, w] complex

        return x
