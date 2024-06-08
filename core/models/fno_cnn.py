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
import neuralop
from .layers.local_fno import FNO
from .constant import *
from .layers import (FullFieldRegressionHead, PerfectlyMatchedLayerPad2d,
                     PermittivityEncoder)
from .layers.activation import SIREN
from .layers.fno_conv2d import FNOConv2d
from .layers.ffno_conv2d import FFNOConv2d, FFNOConv3d
import torch.nn.functional as F
from .pde_base import PDE_NN_BASE
from functools import lru_cache
from pyutils.torch_train import set_torch_deterministic
__all__ = ["FNO3d"]

class SpatialInterpolater(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        stride: int = 1,
        ln: bool = True,
        act_func: Optional[str] = "GELU",
        device: Device = torch.device("cuda:0"),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode="replicate",
            stride=stride,
            groups=groups,
        )
        if ln:
            self.ln = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        else:
            self.ln = None
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
        if self.ln is not None:
            x = self.ln(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x


class FNO3dBlock(nn.Module):
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
        self.f_conv = FFNOConv3d(in_channels, out_channels, n_modes, device=device)
        self.norm = LayerNorm(out_channels, dim = 3, eps=1e-6, data_format="channels_first")
        # self.norm.weight.data.zero_()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding) # should be 3d instead of 2d
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "siren":
            self.act_func = SIREN()
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.conv(x) + self.drop_path(self.f_conv(x)))
        if self.act_func is not None:
            x = self.act_func(x)
        return x

class FNO2dBlock(nn.Module):
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
        self.f_conv = FNOConv2d(in_channels, out_channels, n_modes, device=device)
        self.norm = LayerNorm(out_channels, dim = 2, eps=1e-6, data_format="channels_first")
        # self.norm.weight.data.zero_()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding) # should be 3d instead of 2d
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "siren":
            self.act_func = SIREN()
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.conv(x) + self.drop_path(self.f_conv(x)))
        if self.act_func is not None:
            x = self.act_func(x)
        return x

class LayerNorm(nn.Module):
    r"""LayerNorm implementation used in ConvNeXt
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape,
        dim = 2,
        eps=1e-6,
        data_format="channels_last",
        reshape_last_to_first=False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first
        self.dim = dim

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.dim == 3:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None] # add one extra dimension to match conv2d but not 2d
            elif self.dim == 2:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FNO3d(nn.Module):
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
        img_size: int = 256,
        in_channels: int = 1,
        out_channels: int = 2,
        in_frames: int = 8,
        dim: int = 16,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        mode_list: List[Tuple[int]] = [(20, 20)],
        act_func: Optional[str] = "GELU",
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        aux_head: bool = False,
        aux_head_idx: int = 1,
        pos_encoding: str = "none",
        with_cp: bool = False,
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
        self.img_size = img_size
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
      
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate

        self.aux_head = aux_head
        self.aux_head_idx = aux_head_idx
        self.pos_encoding = pos_encoding
        self.with_cp = with_cp
        self.field_norm_mode = "max"
        if pos_encoding == "none":
            pass
        elif pos_encoding == "linear":
            self.in_channels += 2
        elif pos_encoding == "exp":
            self.in_channels += 4
        elif pos_encoding == "exp3":
            self.in_channels += 6
        elif pos_encoding == "exp4":
            self.in_channels += 8
        elif pos_encoding in {"exp_full", "exp_full_r"}:
            self.in_channels += 7
        else:
            raise ValueError(f"pos_encoding only supports linear and exp, but got {pos_encoding}")

        self.device = device

        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()
    #     self.reset_parameters()

    # def reset_parameters(self, random_state: Optional[int] = None):
    #     for name, m in self.named_modules():
    #         if isinstance(m, self._conv) and hasattr(m, "reset_parameters"):
    #             if random_state is not None:
    #                 # deterministic seed, but different for different layer, and controllable by random_state
    #                 set_torch_deterministic(random_state + sum(map(ord, name)))
    #             m.reset_parameters()

    def build_layers(self):
        # self.pml_pad2d = PerfectlyMatchedLayerPad2d(
        #     self.pml_width, self.buffer_width, self.grid_step, self.pml_permittivity, self.buffer_permittivity
        # )
        # self.stem = nn.Conv2d(
        #     self.in_channels,
        #     self.dim,
        #     1,
        #     padding=0,
        # )  # input channel is 4: (a_r(x, y), a_i(x, y), x, y)
        # kernel_list = [self.dim] + self.kernel_list
        # drop_path_rates = np.linspace(0, self.drop_path_rate, len(kernel_list[:-1]))
        # print(len(kernel_list[:-1]),
        #         len(kernel_list[1:]),
        #         len(self.mode_list),
        #         len(self.kernel_size_list),
        #         len(self.padding_list),
        #         len(drop_path_rates))
        # self.eps_features = nn.Sequential(
        #         nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
        #         LayerNorm(16, dim=2, eps=1e-6, data_format="channels_first"),
        #         nn.GELU(),
        #         nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        #         LayerNorm(16, dim=2, eps=1e-6, data_format="channels_first"),
        #     ) ## used to encode device materials with wave prior, generate [bs, 16, h, w]
        # self.src_features = nn.ModuleList([
        #     neuralop.models.FNO1d(
        #     n_modes_height=self.out_channels//3, # [4, 80, 80]
        #     hidden_channels=16,
        #     lifting_channels=128,
        #     projection_channels=128,
        #     in_channels=1,
        #     out_channels=1,
        #     n_layers=2,
        # ),
        # nn.Conv2d(100, 16, kernel_size=3, padding=1),
        # nn.Sequential(
        #     ConvBlock(
        #         100, 128, kernel_size=3, padding=1, act_func=self.act_func, device=self.device
        #     ),
        #     ConvBlock(
        #         128, 16, kernel_size=3, padding=1, act_func=None, device=self.device
        #     ),
        # ),
        # ]) # generate [bs,outc,h,w] -> [bs*h*w, 1, outc] -> [bs, 16, h, w]
        # self.in_features = nn.Sequential(
        #     nn.Conv3d(1, 9, 3, padding=1),
        #     FNO3dBlock(9, 9, n_modes=[self.in_frames//2, *self.mode_list[0][0:2]], kernel_size=3, padding=1, device=self.device),
        #     FNO3dBlock(9, 9, n_modes=[self.in_frames//2, *self.mode_list[0][0:2]], kernel_size=3, padding=1, device=self.device),
        #     nn.Conv3d(9, 2, 3, padding=1),
        #     nn.Flatten(1, 2),
        # ) # generate [bs, 2, in_frames, h, w] -> [bs, 2*in_frames, h, w]
        # self.downsample = nn.Sequential(
        #         ConvBlock(
        #             in_channels=109, 
        #             out_channels=109, 
        #             kernel_size=3, 
        #             stride=2, 
        #             padding=1,
        #             groups=109,
        #             act_func="GELU",
        #             device=self.device,
        #         ),
        #         ConvBlock(
        #             in_channels=109, 
        #             out_channels=109, 
        #             kernel_size=1, 
        #             padding=0,
        #             ln=True,
        #             act_func=None,
        #             groups=109,
        #             device=self.device,
        #         ),
        #     )
        # self.upsample = nn.Sequential(
        #         ConvBlock(
        #             in_channels=100,
        #             out_channels=100,
        #             kernel_size=1,
        #             padding=0,
        #             act_func=None,
        #             device=self.device,
        #             groups=100,
        #         ),
        #         SpatialInterpolater(),
        #     )
        # self.upsample = nn.PixelShuffle(upscale_factor=2)

        self.head = FNO(
            n_modes=(168, 168),
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            lifting_channels=36,
            projection_channels=512,
            hidden_channels=36,
            n_layers=4,
            norm="my_layernorm",
        )

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

    @lru_cache(maxsize=16)
    def _get_linear_pos_enc(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.arange(0, size_x, device=device)
        gridy = torch.arange(0, size_y, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
        return mesh

    def forward(
        self,
        x,
        src_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ):
        ## x [bs, inc, h, w] real
        ## grid_step: [bs, 1] real
        ## wavelength: float
        ## ------there is no mesh any more because we are now dealing with fdtd, gaussian beam, no single frequency
        ## no longer use eps_feature any more
        # mesh = self._get_linear_pos_enc(x.shape, x.device)  # [1, 2, h, w] real
        # mesh = torch.view_as_real(
        #     torch.exp(mesh.mul(grid_step[..., None, None] / wavelength * 1j * 2 * np.pi)).mul(x[:, 0:1])
        # )  # [1,2,h,w]x[bs,1,1,1] complex * [bs, 1, h, w]real -> [bs, 2, h, w, 2] real
        # mesh = mesh.permute(0, 1, 4, 2, 3).flatten(1, 2)  # [bs, 4, h, w] real
        # eps = torch.cat((x[:, 1:2], mesh), dim=1)  # [bs, 5, h, w] real
        # eps = self.eps_features(eps) # [bs, 16, h, w] real

        eps = 1 / x[:, 0:1].square()
        input_fields = x[:, 1 : 1 + self.in_frames]
        src = x[:, 1 + self.in_frames :, ...]

        normalization_factor = []
        outs = []

        abs_values = torch.abs(input_fields)
        scaling_factor = (
            abs_values.amax(dim=(1, 2, 3), keepdim=True) + 1e-6
        )  # bs, 1, 1, 1

        # scaling_factor = torch.ones_like(scaling_factor) # [bs, 1, 1, 1]

        x = torch.cat(
            [eps, input_fields / scaling_factor, src / scaling_factor], dim=1
        )
        # src_feat = src.permute(0, 2, 3, 1)
        # src_size = src_feat.shape
        # src_feat = src_feat.flatten(0, 2).unsqueeze(1) # [bs*h*w, 1, outc]
        # src_feat = self.src_features[0](src_feat) # [bs*h*w, 1, outc] real
        # src_feat = src_feat.reshape(*src_size[0:3], src_feat.shape[-1]).permute(0, 3, 1, 2) # [bs, outc, h, w]
        # src_feat = self.src_features[1](src_feat) # [bs, 16, h, w]

        # src_feat = self.src_features[2](src)

        # x = self.in_features(x[:, None, 1:1+self.in_frames]) #(bs, 1, in_frames, H, W) -->[bs, 2*in_frames, h, w]
        # x = torch.cat((x, eps, src_feat), dim=1) # [bs, 2*in_frames+16+16, h, w]
        # x = torch.cat((x, src_feat), dim=1)
        # x = self.downsample(x)
        x = self.head(x) # [bs, out_channels, h, w]
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

        return outs, normalization_factor

        
