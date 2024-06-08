'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-20 16:57:43
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-20 16:59:12
'''

from functools import lru_cache
from typing import List, Optional, Tuple
import math
import numpy as np
from sympy import Identity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import dropout, nn
from torch.functional import Tensor
from torch.types import Device

from core.models.layers.mwt_conv2d import MWT2d, MWT_CZ2d

from .constant import *
from .layers.fno_conv2d import FNOConv2d, ComplexFNOConv2d
from .pde_base import PDE_NN_BASE
from .layers.activation import SIREN
from .simulator import Simulation2D, maxwell_residual_2d, maxwell_residual_2d_complex
from pyutils.activation import CReLU, Swish
from timm.models.layers import DropPath


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


class MWT2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "GELU",
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.f_conv = MWT_CZ2d(k=4, c=int(in_channels // 16), alpha=max(n_modes), L=0)
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
        b, inc, h, w = x.shape
        xf = (
            F.interpolate(x, size=(32, 32))
            .permute(0, 2, 3, 1)
            .reshape(b, 32, 32, self.f_conv.c, self.f_conv.k ** 2)
        )
        xf = self.f_conv(xf)
        xf = xf.flatten(3).permute(0, 3, 1, 2)
        xf = F.interpolate(xf, size=(h, w))
        x = self.norm(self.conv(x) + xf)
        # x = self.conv(x) + self.f_conv(x)

        if self.act_func is not None:
            x = self.act_func(x)
        return x

