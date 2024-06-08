import math
from functools import lru_cache
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import einsum
from pyutils.activation import Swish
from pyutils.torch_train import set_torch_deterministic
from timm.models.layers import DropPath
from torch import nn
from torch.functional import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.types import Device

from core.utils import plot_compare, print_stat

from .constant import *
from .layers import (
    FourierConv3d,
    FullFieldRegressionHead,
    PerfectlyMatchedLayerPad2d,
    PermittivityEncoder,
)
from .layers.activation import SIREN
from .pde_base import PDE_NN_BASE

__all__ = ["CNNFDTD"]

import torch
import torch.nn as nn


def get_last_n_frames(packed_sequence, n=100):
    # Unpack the sequence
    padded_sequences, lengths = pad_packed_sequence(packed_sequence, batch_first=True)
    last_frames = []

    for i, length in enumerate(lengths):
        # Calculate the start index for slicing
        start = max(length - n, 0)
        # Extract up to the last n frames
        last_n = padded_sequences[i, start:length, :]
        last_frames.append(last_n)
    last_frames = torch.stack(last_frames, dim=0)
    return last_frames


class SinusoidActivation(nn.Module):
    def forward(self, input):
        return 1.1 * torch.sin(input)


class tanh2(nn.Module):
    def __init__(self):
        super(tanh2, self).__init__()

    def forward(self, input):
        return torch.tanh(input) * 2


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        stride: int = 1,
        norm: str = "ln",
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
        if norm == "ln":
            self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
            # self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "siren":
            self.act_func = SIREN()
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        elif act_func.lower() == "sin":
            self.act_func = SinusoidActivation()
        elif act_func.lower() == "tanh":
            self.act_func = nn.Tanh()
        elif act_func.lower() == "tanh2":
            self.act_func = tanh2()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x
    
class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Device = torch.device("cuda"),
        act_func: dict | None = dict(type="ReLU", inplace=True),
        verbose: bool = False,
        dropout: float = 0.0,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if norm == "ln":
            self.norm = LayerNorm(out_features, eps=1e-6, data_format="channels_first")
        elif norm == "bn":
            self.norm = nn.BatchNorm1d(out_features)
        else:
            self.norm = None
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "siren":
            self.act_func = SIREN()
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        elif act_func.lower() == "sin":
            self.act_func = SinusoidActivation()
        elif act_func.lower() == "tanh":
            self.act_func = nn.Tanh()
        elif act_func.lower() == "tanh2":
            self.act_func = tanh2()
        else:
            self.act_func = getattr(nn, act_func)()
        self.dropout_gate = True

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout is not None and self.dropout_gate:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
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
        eps=1e-6,  # TODO use small dataset to find if -6 is a good order of magnitude
        data_format="channels_last",
        reshape_last_to_first=False,
        interpolate=False,
        is_linear: bool = False,
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
        self.interpolate = interpolate

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.interpolate:
                weight = self.weight.unsqueeze(0).unsqueeze(0)
                weight = F.interpolate(
                    weight, size=x.shape[1], mode="linear", align_corners=True
                )
                weight = weight.squeeze()
                bias = self.bias.unsqueeze(0).unsqueeze(0)
                bias = F.interpolate(
                    bias, size=x.shape[1], mode="linear", align_corners=True
                )
                bias = bias.squeeze()
            else:
                weight = self.weight
                bias = self.bias
            if len(x.shape) == 4:
                x = weight[:, None, None] * x + bias[:, None, None]
            elif len(x.shape) == 5:
                x = weight[:, None, None, None] * x + bias[:, None, None, None]
            elif len(x.shape) == 2:
                x = weight[:] * x + bias[:]
            return x


class CNNFDTD(PDE_NN_BASE):
    """
    FDTD surrogate CNN, with the receptive field of 16x16 but not the whole solution space
    since FDTD is a very local behavior, we can use a small receptive field to capture the local behavior
    and shied the model from the global features
    """

    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 1,
        out_channels: int = 2,
        in_frames: int = 8,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        stride_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        act_func: Optional[str] = "Tanh",
        norm: str = "ln",
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        aux_head: bool = False,
        aux_stide: List[int] = [2, 2, 2],
        aux_padding: List[int] = [1, 1, 1],
        aux_kernel_size_list: List[int] = [3, 3, 3],
        field_norm_mode: str = "max",
        num_iters: int = 1,
        **kwargs,
    ):
        super().__init__()

        """
        The overall network. It contains several conv layer.
        1. the total of the conv network should be about 16*16 since the FDTD is a local behavior
        2. TODO need to think about which makes more sense, bn or ln

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, f = 1+8+50, x, y)
        output: the solution of next 50 frames (bs, f = 50, x ,y)
        """
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_frames = in_frames
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.hidden_list = hidden_list
        self.act_func = act_func
        self.norm = norm
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate

        self.aux_head = aux_head
        self.aux_stide = aux_stide
        self.aux_padding = aux_padding
        self.aux_kernel_size_list = aux_kernel_size_list

        self.field_norm_mode = field_norm_mode
        self.num_iters = num_iters

        self.device = device

        self.build_layers()
        self.reset_parameters()

    def build_layers(self):
        self.conv_stem = nn.ModuleList(
            [
                ConvBlock(
                    self.in_channels,
                    self.kernel_list[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    act_func=self.act_func,
                    device=self.device,
                    norm=self.norm,
                )
            ]
        )

        self.conv_stem.extend(
            ConvBlock(
                inc,
                outc,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                act_func=self.act_func,
                device=self.device,
                norm=self.norm,
            )
            for i, (inc, outc, kernel_size, stride, pad) in enumerate(
                zip(
                    self.kernel_list[:-1],
                    self.kernel_list[1:],
                    self.kernel_size_list,
                    self.stride_list,
                    self.padding_list,
                )
            )
        )
        # self.conv_stem = nn.Sequential(*self.conv_stem)

        self.predictor = nn.Sequential(
            ConvBlock(
                self.kernel_list[-1],
                128,
                kernel_size=3,
                padding=1,
                act_func=self.act_func,
                device=self.device,
                norm=self.norm,
            ),
            ConvBlock(
                128,
                self.out_channels,
                kernel_size=1,
                padding=0,
                act_func=None,
                device=self.device,
                norm="none",
            ),
        )

        if self.aux_head:
            hidden_list = [self.kernel_list[-1]] + self.hidden_list
            head = [
                nn.Sequential(
                    ConvBlock(
                        inc,
                        outc,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=pad,
                        act_func=self.act_func,
                        device=self.device,
                        norm=self.norm,
                    ),
                    nn.Dropout2d(self.dropout_rate),
                )
                for inc, outc, kernel_size, stride, pad in zip(
                    hidden_list[:-1],
                    hidden_list[1:],
                    self.aux_kernel_size_list,
                    self.aux_stide,
                    self.aux_padding,
                )
            ]
            head += [
                ConvBlock(
                    hidden_list[-1],
                    self.out_channels // 2,
                    kernel_size=1,  # element-wise convolution
                    padding=0,
                    act_func=self.act_func,
                    device=self.device,
                    norm=self.norm,
                ),
                nn.Flatten(),
                LinearBlock(
                    self.out_channels // 2 * self.img_size**2,
                    self.out_channels // 10 * self.img_size**2,
                    act_func=self.act_func,
                    device=self.device,
                    norm=self.norm,
                ),
                LinearBlock(
                    self.out_channels // 10 * self.img_size**2,
                    1,
                    act_func=None,
                    device=self.device,
                    norm=self.norm,
                ),
            ]

            self.aux_head = nn.Sequential(*head)
        else:
            self.aux_head = None

    def get_conv_output_dim(self, m: ConvBlock, img_size: int) -> int:
        if isinstance(m, ConvBlock):
            size_out = (
                img_size
                - ((m.conv.kernel_size[0] - 1) * m.conv.dilation[0] + 1)
                + 2 * m.conv.padding[0]
            ) / m.conv.stride[0] + 1
        else:
            raise ValueError("The input module is not a ConvBlock")
        return int(size_out)

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
        target,
        plot: bool = False,
        print_info: bool = False,
        grid_step: Optional[Tensor] = None,
        wavelength: float = 1.55,
        alpha: float = 0.0,
    ):
        ## x [bs, inc, h, w]
        ## inC = 1+8+50
        ## -------------------------------------------
        ## obtian the input fields and the source fields from input data
        eps = 1 / x[:, 0:1].square()
        input_fields = x[:, 1 : 1 + self.in_frames]
        srcs = x[:, 1 + self.in_frames :, ...].chunk(self.num_iters, dim=1)
        ## end obtaining data
        ## -------------------------------------------
        ## init some variables used in the forward pass
        normalization_factor = []
        outs = []
        iter_count = 0
        ## end init variables
        for src in srcs:
            ## -------------------------------------------
            ## begin data preprocessing
            if self.field_norm_mode == "max":
                abs_values = torch.abs(input_fields)
                scaling_factor = (
                    abs_values.amax(dim=(1, 2, 3), keepdim=True) + 1e-6
                )  # bs, 1, 1, 1
            elif self.field_norm_mode == "max99":
                input_fields_copy = input_fields.clone()
                input_fields_copy = torch.flatten(input_fields_copy, start_dim=1)
                p99 = torch.quantile(input_fields_copy, 0.99995, dim=-1, keepdim=True)
                scaling_factor = p99.unsqueeze(-1).unsqueeze(-1)
            elif self.field_norm_mode == "std":
                scaling_factor = 15 * input_fields.std(dim=(1, 2, 3), keepdim=True)
            else:
                raise NotImplementedError
            ## end data preprocessing
            ## -------------------------------------------
            ## preparing the input data
            x = torch.cat(
                [eps, input_fields / scaling_factor, src / scaling_factor], dim=1
            )
            ## end preparing data
            ## -------------------------------------------
            ## forward pass
            for i in range(len(self.conv_stem)):
                if x.shape[-3] == self.conv_stem[i].conv.out_channels:
                    x = self.conv_stem[i](x) + x
                else:
                    x = self.conv_stem[i](x)
            # x = self.conv_stem(x) # conv layer with receptive field of 32 * 32
            x = self.predictor(x)
            out = x + src / scaling_factor  # residual connection
            ## end forward pass
            ## -------------------------------------------
            ## update variables to output
            normalization_factor.append(scaling_factor)
            outs.append(out)
            ## end update variables
            ## -------------------------------------------
            ## update input fields for next iteration
            input_fields = out[:, -self.in_frames :, ...] * scaling_factor
            ## end update input fields
            ## -------------------------------------------
            ## print some information for me to debug
            # output_field = out[:, -self.out_channels:, ...]*scaling_factor
            # print(f"iter num: {iter_count}")
            # print("output:")
            # print_stat(output_field)
            # print("target:")
            # output_field_target = target[:, self.out_channels*iter_count: self.out_channels*(iter_count+1)]
            # print_stat(output_field_target)
            # print(". \n")
            # print(". \n")
            # print(". \n")
            ## end print information
            ## -------------------------------------------
            ## update the iteration count
            iter_count += 1
            ## end update iteration count
            ## -------------------------------------------
        ## end forward pass
        ## -------------------------------------------
        ## update variables to output
        for i in range(len(normalization_factor)):
            normalization_factor[i] = normalization_factor[i].repeat(
                1, self.out_channels, 1, 1
            )
        normalization_factor = torch.cat(normalization_factor, dim=1)
        outs = torch.cat(outs, dim=1)
        ## end update variables
        return outs, normalization_factor

