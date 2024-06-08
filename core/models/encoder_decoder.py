"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-20 16:54:18
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-20 16:56:45
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import nn
from torch.functional import Tensor
from torch.types import Device

from .constant import *
from .pde_base import PDE_NN_BASE
from .utils import conv_output_size
from .layers import PerfectlyMatchedLayerPad2d

__all__ = ["EncoderDecoder"]


class EncoderDecoder(PDE_NN_BASE):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        dim: int = 16,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        act_func: Optional[str] = "GELU",
        domain_size: Tuple[float] = [20, 100],  # computation domain in unit of um
        grid_step: float = 1.550
        / 20,  # grid step size in unit of um, typically 1/20 or 1/30 of the wavelength
        pml_width: float = 0,
        pml_permittivity: complex = 0 + 0j,
        buffer_width: float = 0.5,
        buffer_permittivity: complex = -1e-10 + 0j,
        dropout_rate: float = 0.0,
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
        self.act_func = act_func
        self.domain_size = domain_size
        self.grid_step = grid_step
        self.domain_size_pixel = [round(i / grid_step) for i in domain_size]
        self.buffer_width = buffer_width
        self.buffer_permittivity = buffer_permittivity
        self.pml_width = pml_width
        self.pml_permittivity = pml_permittivity
        self.dropout_rate = dropout_rate

        self.device = device

        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()
        self.reset_parameters()

        self.permittivity_encoder = None

    def build_layers(self):
        channels = [16] * 3
        kernels = [5] * 3
        self.pml_pad2d = PerfectlyMatchedLayerPad2d(
            self.pml_width, self.buffer_width, self.grid_step, self.pml_permittivity, self.buffer_permittivity
        )
        layers = []
        in_channels = self.in_channels
        if self.buffer_width == 0:
            out_size = self.domain_size_pixel
        else:
            out_size = (
                self.domain_size_pixel[0] + self.pml_pad2d.padding_size[2] + self.pml_pad2d.padding_size[3],
                self.domain_size_pixel[1] + self.pml_pad2d.padding_size[0] + self.pml_pad2d.padding_size[1],
            )

        for out_channels, kernel_size in zip(channels, kernels):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=2))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(p=self.dropout_rate))
            in_channels = out_channels
            out_size = (
                conv_output_size(out_size[0], kernel_size, stride=2, padding=2),
                conv_output_size(out_size[1], kernel_size, stride=2, padding=2),
            )

            # print(out_size)

        self.convnet = nn.Sequential(*layers)
        # print(out_size, out_size[0] * out_size[1] * out_channels)

        self.densenet = nn.Sequential(
            nn.Linear(out_size[0] * out_size[1] * out_channels, out_size[0] * out_size[1] * out_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(out_size[0] * out_size[1] * out_channels, out_size[0] * out_size[1] * out_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
        )

        transpose_layers = []
        transpose_channels = [*reversed(channels[1:]), self.out_channels]
        for i, (out_channels, kernel_size) in enumerate(zip(transpose_channels, reversed(kernels))):
            transpose_layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=2, output_padding=1
                )
            )
            if i < len(transpose_channels) - 1:
                transpose_layers.append(nn.LeakyReLU())
            if self.dropout_rate > 0:
                transpose_layers.append(nn.Dropout(p=self.dropout_rate))
            in_channels = out_channels

        self.invconvnet = nn.Sequential(*transpose_layers)

    def forward(self, epsilons, wavelength):
        batch_size, inc, W, H = epsilons.shape
        epsilons = self.pml_pad2d(epsilons)
        epsilons = torch.view_as_real(epsilons).permute(0, 1, 4, 2, 3).flatten(1, 2)

        out = self.convnet(epsilons)
        _, c, w2, h2 = out.shape

        out = out.view(batch_size, -1)
        out = self.densenet(out)

        out = out.view(batch_size, c, w2, h2)
        out = self.invconvnet(out)

        # print(out.shape)
        out = self.pml_pad2d.trim(out)
        # print(out.shape)
        out = torch.view_as_complex(out.view(batch_size, -1, 2, W, H).permute(0, 1, 3, 4, 2).contiguous())
        return out
