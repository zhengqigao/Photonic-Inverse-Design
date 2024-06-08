'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-28 19:07:26
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-02 19:39:23
'''
"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-25 22:47:23
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-12-25 22:55:39
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.activation import Swish
from timm.models.layers import DropPath
from torch import nn
from torch.functional import Tensor
from torch.types import Device

from .constant import *
from .layers import (FullFieldRegressionHead, PerfectlyMatchedLayerPad2d,
                     PermittivityEncoder)
from .layers.activation import SIREN
from .layers.fno_conv2d import FNOConv2d
from .pde_base import PDE_NN_BASE

__all__ = ["MSFNO2d"]


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


class MSFNO2dBlock(nn.Module):
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
        # print(drop_path_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.f_conv = nn.ModuleList(
            [
                FNOConv2d(in_channels, out_channels, n_modes, device=device)
                if i == 0
                else nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=2 ** i + 1,
                        stride=2 ** i,
                        padding=2 ** (i - 1),
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_channels),
                    nn.Conv2d(in_channels, (2 ** i) * in_channels, kernel_size=1),
                    # nn.Conv2d(in_channels, (2 ** i) * in_channels, kernel_size=1, stride=2**i),
                    FNOConv2d(
                        (2 ** i) * in_channels,
                        (2 ** i) * in_channels,
                        (n_modes[0] // 2 ** i, n_modes[1] // 2 ** i),
                        device=device,
                    ),
                    nn.Conv2d(
                        (2 ** i) * in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        bias=True,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.Upsample(
                        scale_factor=2 ** i,
                        mode="nearest",
                    ),
                )
                for i in range(4)
            ]
        )
        self.norm = nn.BatchNorm2d(out_channels)
        # self.norm.weight.data.zero_()
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
        # x = self.f_conv(x)
        # x = self.conv(x) + self.norm(self.f_conv(x))
        y = 0
        for branch in self.f_conv:
            y = y + branch(x)
        y = self.drop_path(y)
        y = self.norm(self.conv(x) + y)
        # x = self.conv(x) + self.f_conv(x)

        if self.act_func is not None:
            y = self.act_func(y)
        return y


class MSFNO2d(PDE_NN_BASE):
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
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        mode_list: List[Tuple[int]] = [(20, 20), (20, 20), (20, 20), (20, 20)],
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
        if pos_encoding == "linear":
            self.in_channels += 2
        elif pos_encoding == "exp":
            self.in_channels += 4
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
        # self.pml_pad2d = PerfectlyMatchedLayerPad2d(
        #     self.pml_width, self.buffer_width, self.grid_step, self.pml_permittivity, self.buffer_permittivity
        # )
        self.stem = nn.Conv2d(
            self.in_channels,
            self.dim,
            1,
            padding=0,
        )  # input channel is 4: (a_r(x, y), a_i(x, y), x, y)
        kernel_list = [self.dim] + self.kernel_list
        drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))
        # print(len(kernel_list[:-1]),
        #         len(kernel_list[1:]),
        #         len(self.mode_list),
        #         len(self.kernel_size_list),
        #         len(self.padding_list),
        #         len(drop_path_rates))
        features = [
            MSFNO2dBlock(
                inc,
                outc,
                n_modes,
                kernel_size,
                padding,
                act_func=self.act_func,
                device=self.device,
                drop_path_rate=drop,
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

        # self.full_field_head = FullFieldRegressionHead(
        #     size=(
        #         self.domain_size_pixel[0] + self.pml_pad2d.padding_size[2] + self.pml_pad2d.padding_size[3],
        #         self.domain_size_pixel[1] + self.pml_pad2d.padding_size[0] + self.pml_pad2d.padding_size[1],
        #     ),
        #     grid_step=self.grid_step,
        # )

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
        )  # [bs, 2 or 4, h, w] real

        x = torch.cat((x, grid), dim=1)  # [bs, inc*2+4, h, w] real

        # DNN-based electric field envelop prediction
        x = self.stem(x)
        # x = self.full_field_head.forward_full(x, epsilon * (self.eps_max - self.eps_min) + self.eps_min, wavelength)  # [bs, outc/2, h, w] complex
        y = None
        x = self.features(x)
        # for i, feat in enumerate(self.features):
        #     x = feat(x)
        #     # print(self.aux_head, i==self.aux_head_idx, self.training)
        #     if self.aux_head is not None and i == self.aux_head_idx and self.training:
        #         y = self.pml_pad2d.trim(self.aux_head(x)) # [bs, outc/2, h, w] real as magnitude
        x = self.head(x)  # [bs, outc, h, w] real
        # convert to complex frequency-domain electric field envelops
        x = torch.view_as_complex(
            x.view(x.size(0), -1, 2, x.size(-2), x.size(-1)).permute(0, 1, 3, 4, 2).contiguous()
        )  # [bs, outc/2, h, w] complex
        # x = torch.view_as_complex(x.permute(0, 2, 3, 1).unsqueeze(1).contiguous())  # [bs, 1, h, w] complex
        # print(x.shape)
        # reconstruct full electric field
        # print(x.shape)
        # x = self.full_field_head.forward_full(x, epsilon * (self.eps_max - self.eps_min) + self.eps_min, wavelength)  # [bs, outc/2, h, w] complex
        # print(x.shape)
        # trim the buffer and pml
        # x = self.pml_pad2d.trim(x)  # [bs, outc/2, h, w]
        if y is not None:
            return x, y
        else:
            return x
