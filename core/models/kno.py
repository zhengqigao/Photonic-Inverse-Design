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
from .constant import *
from .layers import (
    FullFieldRegressionHead,
    PerfectlyMatchedLayerPad2d,
    PermittivityEncoder,
)
from .layers.activation import SIREN
from .layers.kno_conv2d import KNOConv2d, EigenKNOConv2d, FourierKNOConv2d, FourierKNOConv2dFixSz, dct1d, idct1d
from .pde_base import PDE_NN_BASE
from functools import lru_cache
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init
import math
import matplotlib.pyplot as plt
from core.utils import plot_compare, print_stat

__all__ = ["KNO2d", "KNO2d_Roll", "KNO2d_Roll_CNN_Encoder"]

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

class HIPPOConvBlock(nn.Module):
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
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        # Initialize bias as a Parameter with size out_channels
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        elif act_func.lower() == "sin":
            self.act_func = SinusoidActivation()
        elif act_func.lower() == "tanh":
            self.act_func = nn.Tanh()
        else:
            self.act_func = getattr(nn, act_func)()
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the HIPPO matrix here
        # Initialize weight
        hippo_matrix = self.calculate_hippo_matrix()
        self.weight.data.copy_(hippo_matrix)
        # Initialize bias
        nn.init.constant_(self.bias.data, 0)  # Example: Initialize bias to 0
    
    def calculate_hippo_matrix(self):
        # Placeholder for HIPPO matrix calculation
        hippo_matrix = torch.zeros(self.out_channels, self.in_channels)
        for n in range(self.out_channels):
            for k in range(self.in_channels):
                if n > k:
                    hippo_matrix[n][k] = -((2*n + 1)**0.5) * ((2*k + 1)**0.5)
                elif n == k:
                    hippo_matrix[n][k] = -(n+1)
                else:
                    pass
        hippo_matrix = hippo_matrix.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.kernel_size, self.kernel_size)
        return hippo_matrix

    def forward(self, x: Tensor) -> Tensor:
        x = x.type_as(self.bias)
        x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode='replicate')
        x = F.conv2d(x, self.weight, self.bias, stride=self.stride)
        if self.ln is not None:
            x = self.ln(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x

class SinusoidActivation(nn.Module):
    def forward(self, input):
        return 1.1*torch.sin(input)
    
class tanhn(nn.Module):
    def __init__(self,n):
        super(tanhn, self).__init__()
        self.n = n
        
    def forward(self, input):
        return torch.tanh(input) * self.n

class GELU6(nn.Module):
    def __init__(self):
        super(GELU6, self).__init__()
        self.gelu = nn.GELU()
        
    def forward(self, input):
        return torch.nn.functional.gelu(input).clamp(max=6)


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
        elif norm == "bn":
            self.norm = MyBatchNorm2d(out_channels)
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
        elif "tanh" in act_func.lower():
            h_idx = act_func.find("h")
            n = float(act_func[h_idx+1:])
            self.act_func = tanhn(n)
        elif act_func.lower() == "gelu6":
            self.act_func = GELU6()
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
        elif "tanh" in act_func.lower():
            h_idx = act_func.find("h")
            n = float(act_func[h_idx+1:])
            self.act_func = tanhn(n)
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

class WindowPatch(nn.Module): # modified from Jiaqi's code
    def __init__(self, in_frames: int = 8, patch_size: int = 2, stride: int = 1) -> None:
        super().__init__()
        self.in_frames = in_frames
        self.patch_size = patch_size
        self.stride = stride
        ## only the last patch_size in_fields will be packed into a patch.
        ## e.g., in_frames=8, patch_size=2, then only [E6, E7] will be packed as one patch, previous fields are ignored.

    def forward(self, x: Tensor) -> Tensor:
        """generate patch for input fields and source fields.

        Args:
            x (Tensor): [bs, 1+in_frames+f, h, w]: [eps, in_field, src_fields]
        Returns:
            field (Tensor): [bs, f, ps+1, h, w]: patched fields and eps. here is an example for patch size is 2.
            [E0, E1, eps]
            [J0, J1, eps]
            [J2, J3, eps]
            ....
            [Jf-2, Jf-1, eps]
            # note that Jf is not here, because we do not need to predict Jf.
        """
        eps, field = x[:, 0:1], x[:, 1:]
        bs, h, w = x.shape[0], x.shape[-2], x.shape[-1]
        field, source = (
            field[:, self.in_frames - self.patch_size : self.in_frames],
            field[:, self.in_frames :],
        )
        # source = source.permute(0, 2, 3, 1)[
        #     ..., :-1
        # ]  # [bs, h, w, f-1]: [J1, J2, ..., Jf-1]
        source = source.permute(0, 2, 3, 1) # [bs, h, w, f]: [J1, J2, ..., Jf-1, Jf]
        ## TODO: # 'replicate' is not quite right for padded source.
        source = torch.nn.functional.pad(
            source, (self.patch_size - 1, 0, 0, 0), mode="replicate"
        )  # [bs, h, w, f+ps]: [J0, J1, J2, J3, ..., Jf-1, Jf]
        source = torch.nn.functional.unfold(
            source.flatten(0, 2)[:, None, None, :],
            kernel_size=[1, self.patch_size],
            stride=self.stride,
            padding=0,
        )  # [bs*h*w, ps, (f+ps)/ps]
        source = source.reshape(bs, h, w, self.patch_size, -1).permute(
            0, 4, 3, 1, 2
        )  # [bs, (f+ps)/ps, ps, h, w]: [J0, J1], [J1, J2], ... [Jf-2, Jf-1]
        #print(field.shape, source.shape)
        field = torch.cat([field.unsqueeze(1), source], dim=1)  # [bs, (f+ps)/ps+1, ps, h, w]
        field = torch.cat(
            [field, eps.repeat(1, field.size(1), 1, 1).unsqueeze(2)], dim=2
        )  # [bs, (f+ps)/ps+1, ps, h, w], [bs, 1, h, w] -> [bs, (f+ps)/ps+1, ps+1, h, w]
        return field  # [bs, (f+ps)/ps+1, ps+1, h, w]
    
class DecoderWrapper(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x

class KNOBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        r: float = 6.0,
        kno_alg: str = "kno",  # kno or eigen_kno
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "Tanh",
        drop_path_rate: float = 0.0,
        transform: str = "dft",
        forward_transform: bool = True,
        inverse_transform: bool = True,
        device: Device = torch.device("cuda:0"),
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )
        self.r = r
        self.kno_alg = kno_alg
        if kno_alg == "kno":
            self.k_conv = KNOConv2d(in_channels, out_channels, n_modes, device=device)
        elif kno_alg == "eigen_kno":
            self.k_conv = EigenKNOConv2d(
                in_channels, out_channels, n_modes, device=device
            )
        elif kno_alg == "fourier_kno":
            self.k_conv = FourierKNOConv2d(
                in_channels, out_channels, n_modes, device=device, transform=transform,
                forward_transform=forward_transform, inverse_transform=inverse_transform
            )
        else:
            raise NotImplementedError
        if norm == "ln":
            self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        elif norm == "bn":
            self.norm = MyBatchNorm2d(out_channels)
        else:
            self.norm = None
        # self.norm.weight.data.zero_()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode="replicate",
        )
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "siren":
            self.act_func = SIREN()
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        elif act_func.lower() == "tanh":
            self.act_func = nn.Tanh()
        elif "tanh" in act_func.lower():
            h_idx = act_func.find("h")
            n = float(act_func[h_idx+1:])
            self.act_func = tanhn(n)
        elif act_func.lower() == "gelu6":
            self.act_func = GELU6()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x) -> Tensor:
        if self.norm is not None:
            x = self.norm(self.conv(x) + self.drop_path(self.k_conv(x, self.r)))
        else:
            x = self.conv(x) + self.drop_path(self.k_conv(x, self.r))
        if self.act_func is not None:
            x = self.act_func(x)
        return x

class KNOBlockContinuosTime(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        kno_alg: str = "kno",  # kno or eigen_kno
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "Tanh",
        drop_path_rate: float = 0.0,
        transform: str = "dft",
        forward_transform: bool = True,
        inverse_transform: bool = True,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )
        if kno_alg == "kno":
            self.k_conv = KNOConv2d(in_channels, out_channels, n_modes, device=device)
        elif kno_alg == "eigen_kno":
            self.k_conv = EigenKNOConv2d(
                in_channels, out_channels, n_modes, device=device
            )
        elif kno_alg == "fourier_kno":
            self.k_conv = FourierKNOConv2d(
                in_channels, out_channels, n_modes, device=device, transform=transform,
                forward_transform=forward_transform, inverse_transform=inverse_transform
            )
        else:
            raise NotImplementedError
        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
        # self.norm.weight.data.zero_()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode="replicate",
        )
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "siren":
            self.act_func = SIREN()
        elif act_func.lower() == "swish":
            self.act_func = Swish()
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, t: tuple) -> Tensor:
        x, r = t
        x = self.norm(self.conv(x) + self.drop_path(self.k_conv(x, r)))
        # x_w = x
        # x = self.drop_path(self.norm(self.k_conv(x, self.r)))
        # x = self.norm(self.conv(x) * self.drop_path(self.k_conv(x, self.r)))
        if self.act_func is not None:
            x = self.act_func(x)
        # x = x + self.conv(x_w)
        return (x, r)
    
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
        eps=1e-6, # TODO use small dataset to find if -6 is a good order of magnitude
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

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            weight = self.weight
            bias = self.bias
            if len(x.shape) == 4:
                x = weight[:, None, None] * x + bias[:, None, None]
            elif len(x.shape) == 5:
                x = weight[:, None, None, None] * x + bias[:, None, None, None]
            elif len(x.shape) == 2:
                x = weight[:] * x + bias[:]
            return x

class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(MyBatchNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.affine = affine

    def forward(self, x):
        u = x.mean(0, keepdim=True)
        s = (x - u).pow(2).mean(0, keepdim=True)
        s = torch.sqrt(s + self.eps)
        x = (x - u) / s
        weight = self.weight
        bias = self.bias
        if self.affine:
            x = weight[:, None, None] * x + bias[:, None, None]
        return x
class preNorm(nn.Module):
    def __init__(
        self,
        eps=1e-6,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.1]))
        self.eps = eps

    def forward(self, x):
        '''
        suppose the input std = 0.8 I hope it could output a distribution with std = 0.7. 0.8 --> 0.7
        I calculate std/Weight, and then use this std/Weight to normalize the input (x-u)
        the normalized R.V. will always have std = W lets say = 0.4 so in this iteration, the scaling factor = 2
        the network output will be different from x for example 0.34
        the loss function should be mse(0.34, target/scaling_factor = 0.35)
        Use 2 to put output back to the video
        ground truth std = 0.7 and we want our model to predict a distribution with std = 0.7 while now the prediciton std is 0.68, 0.8 --> 0.68

        the predicted result with std ~ 0.68, put it back to the input of the model, the GT output std should be 0.7*7/8 = 0.6125. 0.72 --> 0.6125
        the model will first normalized the input to dist (0.0, 0.4)
        the model will out put a value around 0.34
        in this iteration, the scaling factor = 0.68/0.4 = 1.7
        mse ask the output to mactch mse(0.34, 0.6125/1.7 = 0.3603)
        the ground T target should be 0.7*7/8 = 0.6125 while the model output is 0.34 * 1.7 = 0.578. 0.68 --> 0.578

        '''
        s, u = torch.std_mean(x, dim=(1,2,3), keepdim=True)
        # u = x.mean(dim=(1,2,3), keepdim=True)
        # var = (x-u).square().mean(dim=(1,2,3), keepdim=True) + self.eps
        # s = torch.sqrt(var)
        s = s + self.eps
        normalized_field = (x-u) / s
        weight = self.weight
        # x = weight[:, None, None] * x + bias[:, None, None]
        normalized_field = weight[:, None, None] * normalized_field
        weight_in = weight.clone().detach()
        return normalized_field, s/weight_in[:, None, None], u

class postNorm(nn.Module):
    def __init__(
        self,
        eps=1e-6,
    ):
        super().__init__()
        # self.weight = nn.Parameter(torch.tensor([0.1]))
        self.eps = eps

    def forward(self, x):
        '''
        the postNorm is used to normalized the output pattern using the std
        after this postNorm, the model output will always have the same std
        considering that the input is also normalized to a std
        the model is now mapping a distribution with std_in to a distribution with std_out
        std_in --> std_out

        the postNorm uses (x/std_out) * weight_out to normalize the output
        we will calculate the mse(x/std_out*weight_out, target/std_target*weight_out) to make the model to learn the pattern,
        beacuse now the two distribution have the same std ~ weight_out
        suppose that we can learn an accurate std_pred~std_target, then we can scale the output to the actual video by *std_pred/weight_out

        if std_pred is slightly different with std_target, the model should be more robust compared to former model
        the error that flows to the next iteration is mainly from the pattern error and the std_pred error
        '''
        s, u = torch.std_mean(x, dim=(1,2,3), keepdim=True)
        # u = x.mean(dim=(1,2,3), keepdim=True)
        # var = (x-u).square().mean(dim=(1,2,3), keepdim=True) + self.eps
        # s = torch.sqrt(var)
        s = s + self.eps
        normalized_field = (x-u) / s ## TODO not quite sure about if I could substrct the mean.
        # weight = self.weight
        # normalized_field = weight[:, None, None] * normalized_field
        # weight_out = weight.clone().detach()
        # return normalized_field, weight_out, u
        return normalized_field, u

class KNO2d(PDE_NN_BASE):
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
        act_func: Optional[str] = "Tanh",
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        aux_head: bool = False,
        aux_head_idx: int = 1,
        pos_encoding: str = "none",
        kno_alg: str = "kno",
        kno_r: float = 6,
        transform: str = "dft",
        with_cp: bool = False,
        encoder_cfg: dict = {},
        decoder_cfg: dict = {},
        backbone_cfg: dict = {},
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
        self.kno_alg = kno_alg
        self.kno_r = kno_r
        self.transform = transform

        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate

        self.aux_head = aux_head
        self.aux_head_idx = aux_head_idx
        self.pos_encoding = pos_encoding
        self.with_cp = with_cp
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
            raise ValueError(
                f"pos_encoding only supports linear and exp, but got {pos_encoding}"
            )

        self.device = device

        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()
        self.reset_parameters()

    def build_layers(self):
        # self.pml_pad2d = PerfectlyMatchedLayerPad2d(
        #     self.pml_width, self.buffer_width, self.grid_step, self.pml_permittivity, self.buffer_permittivity
        # )
        self.encoder = ConvBlock(
            self.in_channels,
            self.dim,
            3,
            padding=1,
            stride=1,
            norm="ln",
            act_func="GELU",
            device=self.device,
        )
        self.decoder = nn.Sequential(
            ConvBlock(
                self.dim,
                512,
                3,
                padding=1,
                stride=1,
                norm="ln",
                act_func="GELU",
                device=self.device,
            ),
            ConvBlock(
                512,
                self.out_channels,
                1,
                padding=0,
                stride=1,
                norm="ln",
                act_func=None,
                device=self.device,
            ),
        )
        drop_path_rates = np.linspace(
            0, self.drop_path_rate, len(self.kernel_size_list)
        )

        self.kernel = [
            KNOBlock(
                self.dim,
                self.dim,
                n_modes=self.mode_list[0],
                r=self.kno_r,
                kno_alg=self.kno_alg,
                kernel_size=kernel_size,
                padding=pad,
                act_func="GELU",
                drop_path_rate=drop,
                transform=self.transform,
                # forward_transform=True if ((self.T1 and i == 0) | (not self.T1)) else False,
                # inverse_transform=True if ((self.T1 and i == len(self.kernel_size_list) - 1) | (not self.T1)) else False,
                forward_transform=False,
                inverse_transform=False,
                device=self.device,
            )
            for i, (kernel_size, pad, drop) in enumerate(zip(
                self.kernel_size_list, self.padding_list, drop_path_rates)
            )
        ]
        self.kernel = nn.Sequential(*self.kernel)

        if self.aux_head:
            hidden_list = [self.kernel_list[self.aux_head_idx]] + self.hidden_list
            head = [
                nn.Sequential(
                    ConvBlock(
                        inc,
                        outc,
                        kernel_size=1,
                        padding=0,
                        act_func=self.act_func,
                        device=self.device,
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
    ) -> Tensor:

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

        x = self.encoder(x)
        x = self.kernel(x)
        x = self.decoder(x)

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

class KNO2d_Roll(PDE_NN_BASE):
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
        act_func: Optional[str] = "Tanh",
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        aux_head: bool = False,
        aux_head_idx: int = 1,
        pos_encoding: str = "none",
        kno_alg: str = "kno",
        kno_r: float = 6,
        num_iters: int = 4,
        T1: bool = False,
        transform: str = "dft",
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
        self.kno_alg = kno_alg
        self.kno_r = kno_r
        self.T1 = T1 # transform once
        self.transform = transform

        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate

        self.aux_head = aux_head
        self.aux_head_idx = aux_head_idx
        self.pos_encoding = pos_encoding

        self.num_iters = num_iters
        self.with_cp = with_cp
        

        self.device = device

        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()
        self.reset_parameters()

    def build_layers(self):
        # self.pml_pad2d = PerfectlyMatchedLayerPad2d(
        #     self.pml_width, self.buffer_width, self.grid_step, self.pml_permittivity, self.buffer_permittivity
        # )
        # each KNO block predict a part of the output sequence
        assert (
            self.out_channels % self.num_iters == 0
        ), f"out_channels must be divisible by num_iters, but got {self.out_channels} and {self.num_iters}"
        frames_per_iter = self.out_channels // self.num_iters
        in_channels = 1 + self.in_frames + frames_per_iter  # eps  # in_frames
        if self.pos_encoding == "none":
            pass
        elif self.pos_encoding == "linear":
            in_channels += 2
        elif self.pos_encoding == "exp":
            in_channels += 4
        else:
            raise ValueError(
                f"pos_encoding only supports linear and exp, but got {self.pos_encoding}"
            )
        
        self.encoder = nn.Sequential(
            ConvBlock(
                in_channels,
                self.dim,
                3,
                padding=1,
                ln=True,
                act_func="GELU",
                device=self.device,
            ),
            ConvBlock(
                self.dim,
                self.dim,
                3,
                padding=1,
                ln=True,
                act_func=None,
                device=self.device,
            ),
        )
        self.decoder = nn.Sequential(
            ConvBlock(
                self.dim,
                128,
                3,
                padding=1,
                ln=True,
                act_func="GELU",
                device=self.device,
            ),
            ConvBlock(
                128,
                frames_per_iter,
                1,
                padding=0,
                ln=False,
                act_func=None,
                device=self.device,
            ),
        )
        drop_path_rates = np.linspace(
            0, self.drop_path_rate, len(self.kernel_size_list)
        )

        self.kernel = [
            KNOBlock(
                self.dim,
                self.dim,
                n_modes=self.mode_list[0],
                r=self.kno_r,
                kno_alg=self.kno_alg,
                kernel_size=kernel_size,
                padding=pad,
                act_func="GELU",
                drop_path_rate=drop,
                transform=self.transform,
                forward_transform=True if ((self.T1 and i == 0) | (not self.T1)) else False,
                inverse_transform=True if ((self.T1 and i == len(self.kernel_size_list) - 1) | (not self.T1)) else False,
                device=self.device,
            )
            for i, (kernel_size, pad, drop) in enumerate(zip(
                self.kernel_size_list, self.padding_list, drop_path_rates)
            )
        ]
        self.kernel = nn.Sequential(*self.kernel)
        # self.kernel = nn.Sequential(*self.kernel)

        if self.aux_head:
            hidden_list = [self.kernel_list[self.aux_head_idx]] + self.hidden_list
            head = [
                nn.Sequential(
                    ConvBlock(
                        inc,
                        outc,
                        kernel_size=1,
                        padding=0,
                        act_func=self.act_func,
                        device=self.device,
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

    @lru_cache(maxsize=16)
    def _get_linear_pos_enc(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.arange(0, size_x, device=device)
        gridy = torch.arange(0, size_y, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
        return mesh

    def get_pos_encoding(self, x, grid_step, wavelength):
        if self.pos_encoding == "exp":  
            mesh = self._get_linear_pos_enc(x.shape, x.device)  # [1, 2, h, w] real
            mesh = torch.view_as_real(
                torch.exp(
                    mesh.mul(grid_step[..., None, None] / (wavelength / (2j * np.pi))).mul(
                        x[:, 0:1]
                    )
                )
            )  # [1,2,h,w]x[bs,1,1,1] complex * [bs, 1, h, w]real -> [bs, 2, h, w, 2] real
            mesh = mesh.permute(0, 1, 4, 2, 3).flatten(1, 2)  # [bs, 4, h, w] real
        elif self.pos_encoding == "none":
            mesh = None
        else:
            raise NotImplementedError
        return mesh

    def forward(self, x, grid_step: Tensor, wavelength: float = 1.55):
        ## x [bs, inc, h, w] real
        ## grid_step: [bs, 1] real
        ## wavelength: float

        mesh = self.get_pos_encoding(x, grid_step, wavelength)

        ## we partition sources into multiple chunks
        sources = x[:, -self.out_channels :].chunk(self.num_iters, dim=1)
        eps = 1 / x[:, 0:1].square()
        input_fields = x[:, 1 : 1 + self.in_frames]
        outs = []
        for src in sources:
            if mesh is None:
                x = torch.cat([eps, input_fields, src], dim=1)
            else:
                x = torch.cat([eps, mesh, input_fields, src], dim=1)
            
            x = self.encoder(x)
            x = self.kernel(x)
            out = self.decoder(x) + src
            outs.append(out)
            input_fields = out[:, -self.in_frames :]
        outs = torch.cat(outs, dim=1)
        return outs
class KNO2d_Roll_CNN_Encoder(PDE_NN_BASE):
    """
    This model is used to work as fast surrogate model for FDTD simulation
    it takes two inputs, the initial E field called E0 and the source
    E0 is fixed frames [batch_size, in_frames, h, w]
    Source is varying frames but grouped into fixed number of frames [batch_size, # of groups, group_frames, h, w]
    h = w = 256
    may need to downsample to speed up the training
    E0 will be encoded by a CNN to a latent feature with the size (batch_size, dim//2, h, w) and then unsqueeze to (batch_size, 1, dim//2, h, w)
    Source will be encoded by a CNN to a latent feature with the size (batch_size, # of groups, dim//2, h, w)
    concate the two latent features along dim=-2 to (batch_size, # of groups, dim, h, w)
    now use the DCT as the token mixing to mix the latent feature along dim=1 to (batch_size, # of groups, dim, h, w)
    and then feed this mixed tokens to the KNO block to do the time marching
    KNO block will output the same shape as the input
    then use the inverse DCT to decode the token mixing to (batch_size, # of groups, dim, h, w)
    and then use a CNN to decode the latent feature to (batch_size, # of groups, group_frames, h, w)
    then restore the output to (batch_size, out_frames, h, w)
    use lable to supervise the training

    Args:
        PDE_NN_BASE ([type]): [description]
    """

    def __init__(
        self,
        img_size: int = 256,
        out_channels: int = 2,
        in_frames: int = 8,
        dim: int = 16,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        mode_list: List[Tuple[int]] = [(20, 20)],
        act_func: Optional[str] = "Tanh",
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        aux_head: bool = False,
        aux_head_idx: int = 1,
        pos_encoding: str = "none",
        kno_alg: str = "kno",
        kno_r: float = 6,
        num_iters: int = 4,
        T1: bool = False,
        transform: str = "dft",
        with_cp: bool = False,
        downsample: bool = False,
        endecoder_dim: int = 6,
        downsample_dim: int = 1,
        num_hidden_encoder: int = 2,
        include_history: bool = False,
        mode: str = "from_gnd_truth",
        field_norm_mode: str = "none",
        patch_size: int = 8, 
        norm: str = "none",
    ):
        super().__init__()

        """
        the overall network:
        use 2 CNN with fixed in channel and out channel to encode the folded varying length of the input E field and source
        use dft to do the token mixing
        use KNO to do the time marching
        use idft to decode the token mixing
        use CNN with fixed in channel and out channel to decode the folded varying length of the latent feature to varying length of the output E field
        """
        self.img_size = img_size
        self.out_channels = out_channels
        self.in_frames = in_frames
        # assert (
        #     out_channels % 2 == 0
        # ), f"The output channels must be even number larger than 2, but got {out_channels}"
        self.dim = dim
        assert(
            self.dim % 2 == 0
        ), f"The dim must be even number larger than 2, but got {self.dim}"
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.hidden_list = hidden_list
        self.mode_list = mode_list
        self.act_func = act_func
        self.kno_alg = kno_alg
        self.kno_r = kno_r
        self.T1 = T1 # transform once
        self.transform = transform

        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate

        self.aux_head = aux_head
        self.aux_head_idx = aux_head_idx
        self.pos_encoding = pos_encoding

        self.num_iters = num_iters
        self.with_cp = with_cp
        self.downsample = downsample
        self.endecoder_dim = endecoder_dim
        self.downsample_dim = downsample_dim
        self.num_hidden_encoder = num_hidden_encoder
        self.include_history = include_history
        self.mode = mode
        self.field_norm_mode = field_norm_mode
        self.patch_size = patch_size
        self.norm = norm

        # assert (
        #     self.out_channels % self.num_iters == 0
        # ), f"out_channels must be divisible by num_iters, but got {self.out_channels} and {self.num_iters}"
        self.frames_per_iter = self.out_channels // self.num_iters

        self.device = device

        self.padding = 9  # pad the domain if input is non-periodic
        self.build_layers()
        self.reset_parameters()

    def build_layers(self):
        in_channels = 1+self.in_frames+self.out_channels
        in_channels_E0 = 1 + self.in_frames  # eps  # in_frames
        in_channels_source = 1 + self.out_channels  # eps  # frames_per_iter
        if self.pos_encoding == "none":
            pass
        elif self.pos_encoding == "linear":
            in_channels_E0 += 2
            in_channels_source += 2
        elif self.pos_encoding == "exp":
            in_channels += 4
            in_channels_E0 += 4
            in_channels_source += 4
        else:
            raise ValueError(
                f"pos_encoding only supports linear and exp, but got {self.pos_encoding}"
            )
        
        if self.kno_alg == "fourier_kno" and self.include_history:
            self.encoder = nn.Sequential(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=self.dim,
                    kernel_size=3,
                    padding=1,
                    ln=True,
                    act_func="GELU",
                    device=self.device,
                ),
                ConvBlock(
                    in_channels=self.dim,
                    out_channels=self.dim,
                    kernel_size=3,
                    padding=1,
                    ln=True,
                    act_func=None,
                    device=self.device,
                ),
            )
            # self.history_carrier = nn.Sequential(
            #     HIPPOConvBlock(
            #         in_channels=self.dim,
            #         out_channels=self.dim,
            #         kernel_size=3,
            #         padding=1,
            #         stride=1,
            #         ln=True,
            #         act_func="siren",
            #         device=self.device,
            #     ),
            #     HIPPOConvBlock(
            #         in_channels=self.dim,
            #         out_channels=self.dim,
            #         kernel_size=1,
            #         padding=0,
            #         stride=1,
            #         ln=True,
            #         act_func=None,
            #         device=self.device,
            #     ),
            # )
            self.decoder = nn.Sequential(
                ConvBlock(
                    self.dim,
                    128,
                    3,
                    padding=1,
                    ln=True,
                    act_func="GELU",
                    device=self.device,
                ),
                ConvBlock(
                    128,
                    # self.out_channels,
                    100,
                    1,
                    padding=0,
                    ln=False,
                    act_func=None,
                    device=self.device,
                ),
            )
        elif self.kno_alg == "fourier_kno" and not self.include_history:
            if self.field_norm_mode == "layernorm":
                self.prenorm = preNorm()
                self.postnorm = postNorm()
            self.encoder = nn.Sequential(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=self.dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    act_func="GELU",
                    device=self.device,
                    norm=self.norm,
                ),
                ConvBlock(
                    in_channels=self.dim,
                    out_channels=self.dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    act_func=None,
                    device=self.device,
                    norm=self.norm
                ),
            )
            self.decoder = nn.Sequential(
                ConvBlock(
                    self.dim,
                    128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    act_func="GELU",
                    device=self.device,
                    norm=self.norm,
                ),
                ConvBlock(
                    128,
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    act_func=None,
                    device=self.device,
                    norm="None",
                ),
            )

        drop_path_rates = np.linspace(
            0, self.drop_path_rate, len(self.kernel_size_list)
        )
        self.kernel = [
            KNOBlock(
                self.dim,
                self.dim,
                n_modes=self.mode_list[0],
                r=self.kno_r,
                kno_alg=self.kno_alg,
                kernel_size=kernel_size,
                padding=pad,
                act_func="GELU" if (i != len(self.kernel_size_list) - 1) else None,
                drop_path_rate=drop,
                transform=self.transform,
                # forward_transform=True if ((self.T1 and i == 0) | (not self.T1)) else False,
                # inverse_transform=True if ((self.T1 and i == len(self.kernel_size_list) - 1) | (not self.T1)) else False,
                forward_transform=True,
                inverse_transform=True,
                device=self.device,
                norm=self.norm,
            )
            for i, (kernel_size, pad, drop) in enumerate(zip(
                self.kernel_size_list, self.padding_list, drop_path_rates)
            )
        ]
        self.kernel = nn.Sequential(*self.kernel)

        if self.aux_head:
            hidden_list = [self.kernel_list[self.aux_head_idx]] + self.hidden_list
            head = [
                nn.Sequential(
                    ConvBlock(
                        inc,
                        outc,
                        kernel_size=1,
                        padding=0,
                        act_func=self.act_func,
                        device=self.device,
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

    @lru_cache(maxsize=16)
    def _get_linear_pos_enc(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.arange(0, size_x, device=device)
        gridy = torch.arange(0, size_y, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
        return mesh

    def get_pos_encoding(self, x, grid_step, wavelength):
        if self.pos_encoding == "exp":  
            mesh = self._get_linear_pos_enc(x.shape, x.device)  # [1, 2, h, w] real
            mesh = torch.view_as_real(
                torch.exp(
                    mesh.mul(grid_step[..., None, None] / (wavelength / (2j * np.pi))).mul(
                        x[:, 0:1]
                    )
                )
            )  # [1,2,h,w]x[bs,1,1,1] complex * [bs, 1, h, w]real -> [bs, 2, h, w, 2] real
            mesh = mesh.permute(0, 1, 4, 2, 3).flatten(1, 2)  # [bs, 4, h, w] real
        elif self.pos_encoding == "none":
            mesh = None
        else:
            raise NotImplementedError
        return mesh

    def forward(self,
                x, 
                # history_vector, 
                target, 
                plot: bool = False, 
                print_info: bool = False,
                wavelength: float = 1.55,
                grid_step: Optional[Tensor] = None,
                alpha: float = 0.0,
                total_max: Tensor = None,
                ):

        if self.pos_encoding == 'exp':
            mesh = self.get_pos_encoding(x, grid_step, wavelength)

        ## we partition sources into multiple chunks
        sources = x[:, 1+self.in_frames:, ...].chunk(self.num_iters, dim=1)
        if plot:
            eps_to_plot = x[0, 0:1, ...].clone() # used in plot_compare
        eps = 1 / x[:, 0:1].square()
        input_fields = x[:, 1 : 1 + self.in_frames]
        outs = []
        normalization_factor = []
        iter_count = 0
        for src in sources:
            ## -------------------------------------------
            ## begin data preprocessing
            if self.field_norm_mode == "layernorm":
                x = torch.cat([eps, input_fields, src], dim=1) 
                normed_in_field, scaling_factor, _ = self.prenorm(input_fields)
                normed_src = src / scaling_factor
                x = torch.cat([eps, normed_in_field, normed_src], dim=1)
                normalized_Efield = torch.cat([normed_in_field, normed_src], dim=1)
            elif self.field_norm_mode == "max":
                if self.mode == "auto_regressive":
                    abs_values = torch.abs(input_fields)
                    scaling_factor = abs_values.amax(dim=(1, 2, 3), keepdim=True) + 1e-6 # bs, 1, 1, 1 
                    if plot:
                        input_fields_to_plot_compare = input_fields[0, ...].clone() # used in plot_compare
                elif self.mode == "from_gnd_truth":
                    abs_values_src = torch.abs(src)
                    max_abs_src = abs_values_src.amax(dim=(1, 2, 3), keepdim=True) + 1e-6 # bs, 1, 1, 1
                    abs_values = torch.abs(input_fields)
                    max_abs = abs_values.amax(dim=(1, 2, 3), keepdim=True) + 1e-6 # bs, 1, 1, 1
                    scaling_factor = torch.max(max_abs, max_abs_src)
            elif self.field_norm_mode == "max99":
                input_fields_copy = input_fields.clone()
                input_fields_copy = torch.flatten(input_fields_copy, start_dim=1)
                p99 = torch.quantile(input_fields_copy, 0.99995, dim=-1, keepdim=True)
                scaling_factor = p99.unsqueeze(-1).unsqueeze(-1)
            elif self.field_norm_mode == "std":
                if self.mode == "auto_regressive":
                    scaling_factor = 15*input_fields.std(dim=(1, 2, 3), keepdim=True)
                elif self.mode == "from_gnd_truth":
                    std_src = src.std(dim=(1, 2, 3), keepdim=True) + 1e-6
                    std = input_fields.std(dim=(1, 2, 3), keepdim=True) + 1e-6
                    scaling_factor = torch.max(std, std_src)
            elif self.field_norm_mode == "total_max":
                scaling_factor = total_max
            else:
                raise NotImplementedError
            if print_info:
                print("this is scaling factor: \n", scaling_factor.squeeze())
                print("----before normalization----")
                origion_field = torch.cat([input_fields, src], dim=1)
                print_stat(origion_field)
                # for i in range(x.shape[0]):
                #     print_stat(x[i, ...])
                # normalized_field = torch.cat([normed_in_field, normed_src], dim=1)
                print("----after normalization----")
                print_stat(normalized_Efield)
                # for i in range(x.shape[0]):
                #     print_stat(x[i])
            ## end data preprocessing
            ## -------------------------------------------
            ## begin concate the tensor to prepare for the input of the model
            if self.field_norm_mode == "layernorm":
                pass
            else:
                if self.pos_encoding == 'none':
                    x = torch.cat([eps, input_fields/scaling_factor, src/scaling_factor], dim=1)
                else:
                    x = torch.cat([eps, mesh, input_fields/scaling_factor, src/scaling_factor], dim=1)
            ## end concate the tensor to prepare for the input of the model
            ## -------------------------------------------
            ## begin model forward
            if not self.include_history:
                if print_info:
                    print("----input----")
                    print_stat(x)
                x = self.encoder(x)
                if print_info:
                    print("----feature map after encoder----")
                    print_stat(x)
                x, _ = self.kernel((x, 1))
                if print_info:
                    print("----feature map after kernel----")
                    print_stat(x)
                x = self.decoder(x)
                if print_info:
                    print("----output after decoder----")
                    print_stat(x)
                out = x + src / scaling_factor
                if print_info:
                    print("----output after residual connection----")
                    print_stat(out)
            elif self.include_history:
                if print_info:
                    print("----input stats----")
                    print_stat(x)
                x = self.encoder(x)
                if print_info:
                    print("----feature map after encoder----")
                    print_stat(x)
                x, _ = self.kernel((x, 1)) # KNO and encoder, the Bbar
                if print_info:
                    print("----feature map after kernel----")
                    print_stat(x)
                # history_vector = self.history_carrier(history_vector) # h_t = Abar * h_{t-1} + Bbar * x_t
                history_vector = history_vector.type_as(x)
                history_vector, _ = self.kernel((history_vector, 1)) # history_vector is in linear space, so we need to use the kernel do the time step
                if print_info:
                    print("----history vector after carrier----")
                    print_stat(history_vector)
                x = history_vector + x
                if print_info:
                    print("----feature map after fusing history----")
                    history_vector = x.clone()
                out = self.decoder(x) + src # decoder is the matix C and + src is the residual connection which is the D matrix
                if print_info:
                    print("----output after decoder----")
            ## end model forward
            ## -------------------------------------------
            ## preparing the output data
            out_GT = target[:, self.out_channels*(iter_count): self.out_channels*(iter_count+1)]/scaling_factor
            out_ERR = out - out_GT
            # alpha = 0.5
            pred = out_GT + alpha*out_ERR # pred is a mixed intermediate result of the ground truth and the model solution
            # outs.append(pred)
            outs.append(out)
            normalization_factor.append(scaling_factor)
            ## end preparing the output data
            ## -------------------------------------------
            ## update the input fields for the next iteration
            if iter_count < self.num_iters:
                # update the input fields for the next iteration
                input_fields = out[:, -self.in_frames :]*normalization_factor[-1]
                if plot:
                    input_fields_to_plot = input_fields[0, -3, ...]
                    input_fields_to_plot = input_fields_to_plot.cpu().detach().numpy()
                    print("----prediction stats----")
                    print_stat(input_fields)
                    prediction_gnd_truth_to_plot = target[0, self.out_channels*(iter_count): self.out_channels*(iter_count+1)]
                    input_fields_from_gnd_truth = target[:, self.out_channels*(iter_count+1)-self.in_frames : self.out_channels*(iter_count+1)] # each iteration, instead of feeding the output of the last iteration, we feed the ground truth
                    input_fields_from_gnd_truth_to_plot = input_fields_from_gnd_truth[0, -3, ...]
                    input_fields_from_gnd_truth_to_plot = input_fields_from_gnd_truth_to_plot.cpu().detach().numpy()
                    print("----ground truth stats----")
                    print_stat(input_fields_from_gnd_truth)
                    input_fields_err = input_fields - input_fields_from_gnd_truth
                    input_fields_err_to_plot = input_fields_err[0, -3, ...]
                    input_fields_err_to_plot = input_fields_err_to_plot.cpu().detach().numpy()
                    err_percent = 100*(input_fields_err.abs()/input_fields_from_gnd_truth.abs()).mean().item()
                    # print(f"the error percentage is {err_percent:.2f}%")

                    # plot_compare(
                    #     epsilon=eps_to_plot,
                    #     input_fields=input_fields_to_plot_compare.squeeze(),
                    #     pred_fields=(out[0]*scaling_factor[0]).squeeze(),
                    #     target_fields=prediction_gnd_truth_to_plot.squeeze(),
                    #     # filepath=f'./plot/outC50_inC50_mixup_full_err/noisy_GT_long_input_fields_prediction_{iter_count}.png',
                    #     filepath=f'./plot/outC50_inC50_no_mixup_0p{int(alpha*10)}/noisy_GT_long_input_fields_prediction_{iter_count}.png',
                    #     # filepath=f'./plot/sweep/err_added_0p{int(alpha*10)}/outC50_masked_loss_{iter_count}.png',
                    #     pol="Ez",
                    #     norm=False,
                    # )
                    noise = input_fields_err * alpha
                    input_fields = input_fields_from_gnd_truth + noise # each iteration, instead of feeding the output of the last iteration, we feed the ground truth
            iter_count += 1
            ## end update the input fields for the next iteration
            ## -------------------------------------------
        outs = torch.cat(outs, dim=1)
        for i in range(len(normalization_factor)):
            normalization_factor[i] = normalization_factor[i].repeat(1, self.out_channels, 1, 1)
        normalization_factor = torch.cat(normalization_factor, dim=1)
        return outs, normalization_factor