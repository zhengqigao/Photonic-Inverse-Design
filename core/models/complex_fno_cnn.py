"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-20 17:13:26
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-20 17:14:27
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.activation import CReLU
from torch import nn
from torch.functional import Tensor
from torch.types import Device

from .constant import *
from .layers import FullFieldRegressionHead, PerfectlyMatchedLayerPad2d, PermittivityEncoder

from .layers.fno_conv2d import ComplexFNOConv2d
from .pde_base import PDE_NN_BASE

__all__ = ["ComplexFNO2d"]


class ComplexConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "CReLU",
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "crelu":
            self.act_func = CReLU()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.complex(
            self.conv_real(x.real) - self.conv_imag(x.imag), self.conv_real(x.imag) + self.conv_imag(x.real)
        )
        if self.act_func is not None:
            x = self.act_func(x)
        return x


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.norm_real = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )
        self.norm_imag = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.complex(self.norm_real(x.real), self.norm_imag(x.imag))


class ComplexFNO2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "CReLU",
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.f_conv = ComplexFNOConv2d(in_channels, out_channels, n_modes, device=device)
        self.norm = ComplexBatchNorm2d(out_channels)
        self.norm.norm_real.weight.data.zero_()
        self.norm.norm_imag.weight.data.zero_()
        self.conv = ComplexConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            act_func=None,
            device=device,
        )
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "crelu":
            self.act_func = CReLU()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x) + self.norm(self.f_conv(x))

        if self.act_func is not None:
            x = self.act_func(x)
        return x


class ComplexFNO2d(PDE_NN_BASE):
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
        in_channels: int = 4,
        out_channels: int = 3,
        dim: int = 16,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        mode_list: List[Tuple[int]] = [(20, 20), (20, 20), (20, 20), (20, 20)],
        act_func: Optional[str] = "GELU",
        domain_size: Tuple[float] = [20, 100],  # computation domain in unit of um
        grid_step: float = 0.1,  # grid step size in unit of um, typically 1/20 or 1/30 of the wavelength
        pml_width: float = 0,
        pml_permittivity: complex = 0 + 0j,
        buffer_width: float = 0.5,
        buffer_permittivity: complex = -1e-10 + 0j,
        device: Device = torch.device("cuda:0"),
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
        self.in_channels = in_channels + 1
        self.out_channels = out_channels
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

        self.device = device

        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()
        self.reset_parameters()

        self.permittivity_encoder = None
        self.set_trainable_permittivity(False)

    def build_layers(self):
        self.pml_pad2d = PerfectlyMatchedLayerPad2d(
            self.pml_width, self.buffer_width, self.grid_step, self.pml_permittivity, self.buffer_permittivity
        )
        self.stem = ComplexConvBlock(
            self.in_channels,
            self.dim,
            1,
            act_func=None,
            device=device,
        )  # input channel is 4: (a_r(x, y), a_i(x, y), x, y)
        kernel_list = [self.dim] + self.kernel_list
        features = [
            ComplexFNO2dBlock(
                inc,
                outc,
                n_modes,
                kernel_size,
                padding,
                device=self.device,
            )
            for inc, outc, n_modes, kernel_size, padding in zip(
                kernel_list[:-1],
                kernel_list[1:],
                self.mode_list,
                self.kernel_size_list,
                self.padding_list,
            )
        ]
        self.features = nn.Sequential(*features)
        hidden_list = [self.kernel_list[-1]] + self.hidden_list
        head = [
            ComplexConvBlock(inc, outc, kernel_size=1, padding=0, act_func=self.act_func, device=self.device)
            for inc, outc in zip(hidden_list[:-1], hidden_list[1:])
        ]
        # 2 channels as real and imag part of the TE field
        head += [
            ComplexConvBlock(
                hidden_list[-1],
                self.out_channels,
                kernel_size=1,
                padding=0,
                act_func=None,
                device=self.device,
            )
        ]

        self.head = nn.Sequential(*head)

        self.full_field_head = FullFieldRegressionHead(
            size=(
                self.domain_size_pixel[0] + self.pml_pad2d.padding_size[2] + self.pml_pad2d.padding_size[3],
                self.domain_size_pixel[1] + self.pml_pad2d.padding_size[0] + self.pml_pad2d.padding_size[1],
            ),
            grid_step=self.grid_step,
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

    def forward(self, x: Optional[Tensor] = None, wavelength: Optional[float] = None) -> Tensor:
        # x [bs, inc, h, w] complex

        # If the permittivity field is not given as the PDE parameters
        # we will use learnable permittivity
        if self.trainable_permittivity and self.permittivity_encoder is not None:
            x = self.permittivity_encoder(wavelength)  # [bs, 1, h, w] complex

        # padding the permittivity with buffer and PML
        x = self.pml_pad2d(x)

        # positional encoding
        grid = self.get_grid(x.shape, x.device).fill_(0)  # [bs, 2, h, w] real
        grid = torch.view_as_complex(
            grid.view(grid.size(0), 1, 2, grid.size(-2), grid.size(-1)).permute(0, 1, 3, 4, 2).contiguous()
        )  # [bs, 1, h, w] complex
        x = torch.cat((x, grid), dim=1)  # [bs, inc+1, h, w] complex

        # DNN-based electric field envelop prediction
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)  # [bs, outc, h, w] complex

        x = self.full_field_head.forward_full(x, wavelength)  # [bs, outc, h, w] complex
        # print(x.shape)
        # trim the buffer and pml
        x = self.pml_pad2d.trim(x)  # [bs, outc, h, w]

        return x
