"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-02 21:12:39
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-12-25 17:34:29
"""
from functools import lru_cache
from typing import Tuple

import torch
from torch import nn
from torch.functional import Tensor
from torch.types import Device

__all__ = ["KNOConv2d", "EigenKNOConv2d", "FourierKNOConv2d", "FourierKNOConv2dFixSz", "dct1d", "idct1d"]

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


@lru_cache(maxsize=32)
def get_precomputed_expk(N, device):
    k = torch.arange(N, device=device, dtype=torch.float).mul_(np.pi / (2 * N))
    expk = torch.exp(1j * k)
    return expk


def dct1d(x, norm=None, dim=-1):
    if dim != -1:
        x = x.transpose(dim, -1)
    x_shape = x.shape

    N = x_shape[-1]
    # x = x.contiguous().view(-1, N)

    v = torch.cat([x[..., ::2], x[..., 1::2].flip([-1])], dim=-1)

    Vc = torch.fft.fft(v, dim=-1)  # add this line

    # k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    # W_r = torch.cos(k)
    # W_i = torch.sin(k)
    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    expk = get_precomputed_expk(N, x.device)
    V = Vc.imag * (2 * expk.imag) + Vc.real * (2 * expk.real)

    if norm == "ortho":
        V[..., 0] /= np.sqrt(N) * 2
        V[..., 1:] /= np.sqrt(N / 2) * 2

    # V = 2 * V.view(*x_shape)

    if dim != -1:
        V = V.transpose(dim, -1)

    return V


def idct1d(X, norm=None, dim=-1):
    if dim != -1:
        X = X.transpose(dim, -1)

    x_shape = X.shape
    
    N = x_shape[-1]

    # X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    X_v = X / 2

    if norm == "ortho":
        X_v[..., 0] *= np.sqrt(N) * 2
        X_v[..., 1:] *= np.sqrt(N / 2) * 2

    # k = (
    #     torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
    #     * np.pi
    #     / (2 * N)
    # )
    # W_r = torch.cos(k)
    # W_i = torch.sin(k)
    expk = get_precomputed_expk(N, X.device)

    V_t_r = X_v
    V_t_i = torch.cat(
        [torch.zeros_like(X_v[..., :1]), -X_v.flip([-1])[..., :-1]], dim=-1
    )

    V_t = torch.complex(V_t_r.contiguous(), V_t_i)
    V = V_t * expk

    # V_r = V_t_r * W_r - V_t_i * W_i
    # V_i = V_t_r * W_i + V_t_i * W_r

    # V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    # v = torch.irfft(V, 1, onesided=False)                             # comment this line
    # v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)  # add this line

    v = torch.fft.irfft(V, dim=-1, n=V.shape[-1])
    x = v.new_zeros(v.shape)
    x[..., ::2] = v[..., : N - (N // 2)]
    # x[..., 1::2] = v.flip([-1])[..., : N // 2]
    x[..., 1::2] = v[..., -(N // 2) :].flip([-1])

    if dim != -1:
        x = x.transpose(dim, -1)

    return x


def dctn(x, norm="ortho", dim=(-2, -1)):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    if type(dim) is int:
        dim = (dim,)
    for d in dim[::-1]:
        x = dct1d(x, norm=norm, dim=d)
    return x


def idctn(x, norm="ortho", dim=(-2, -1)):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    if type(dim) is int:
        dim = (dim,)
    for d in dim[::-1]:
        x = idct1d(x, norm=norm, dim=d)
    return x


class KNOConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        https://arxiv.org/pdf/2010.08895.pdf
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        (
            self.n_mode_1,
            self.n_mode_2,
        ) = n_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.device = device

        self.scale = 1 / (in_channels * out_channels)
        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
        self.weight = nn.Parameter(
            self.scale
            * torch.zeros(
                [self.in_channels, self.out_channels, *self.n_modes], dtype=torch.cfloat
            )
        )

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight.real)

    def get_zero_padding(self, size, device):
        bs, h, w = size[0], size[-2], size[-1] // 2 + 1
        return torch.zeros(
            bs, self.out_channels, h, w, dtype=torch.cfloat, device=device
        )

    def forward(self, x: Tensor, r: float) -> Tensor:
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        input = x
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Multiply relevant Fourier modes
        out_ft = self.get_zero_padding(x.size(), x.device)
        # out_ft = x.clone()
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        n_mode_1 = min(out_ft.size(-2) // 2, self.n_mode_1)
        n_mode_2 = min(out_ft.size(-1), self.n_mode_2)
        # print(out_ft.shape, n_mode_1, n_mode_2, self.weight_1.shape)
        x_ft_tmp = x_ft
        for _ in range(r):
            out_ft[..., :n_mode_1, :n_mode_2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft_tmp[..., :n_mode_1, :n_mode_2], self.weight
            )
            x_ft_tmp = out_ft.clone()

        x_ft_tmp = x_ft
        for _ in range(r):
            out_ft[:, :, -n_mode_1:, :n_mode_2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft_tmp[:, :, -n_mode_1:, :n_mode_2], self.weight
            )
            x_ft_tmp = out_ft.clone()

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x + input


class EigenKNOConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        https://arxiv.org/pdf/2010.08895.pdf
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        (
            self.n_mode_1,
            self.n_mode_2,
        ) = n_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.device = device

        self.scale = 1 / (in_channels * out_channels)
        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
        self.weight = nn.Parameter(
            self.scale
            * torch.zeros(
                [*self.n_modes, self.in_channels, self.out_channels],
                dtype=torch.cfloat,
                device=self.device,
            )
        )
        self.kernel = nn.Parameter(
            self.scale
            * torch.zeros(
                [*self.n_modes, self.out_channels, 1],
                dtype=torch.cfloat,
                device=self.device,
            )
        )

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight.real)
        u, s, vh = torch.linalg.svd(self.weight)
        self.kernel.data.copy_(s[..., None])
        self.weight.data.copy_(u.matmul(vh))

    def get_zero_padding(self, size, device):
        bs, h, w = size[0], size[-2], size[-1] // 2 + 1
        return torch.zeros(
            bs, self.out_channels, h, w, dtype=torch.cfloat, device=device
        )

    def forward(self, x: Tensor, r: float) -> Tensor:
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Multiply relevant Fourier modes
        out_ft = self.get_zero_padding(x.size(), x.device)
        # out_ft = x.clone()
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        n_mode_1 = min(out_ft.size(-2) // 2, self.n_mode_1)
        n_mode_2 = min(out_ft.size(-1), self.n_mode_2)
        # print(out_ft.shape, n_mode_1, n_mode_2, self.weight_1.shape)

        # u, _, vh = torch.linalg.svd(self.weight) # [*modes, o, o]
        # u = u.matmul(vh) # map to unitary
        u = self.weight
        ## matrix exponent is easy after eigen decomposition
        ## we use unitary projection as a reparametrization to force eigen decomposition
        weight = u.matmul(self.kernel.pow(r).mul(u.mH))

        out_ft[..., :n_mode_1, :n_mode_2] = torch.einsum(
            "bixy,xyio->boxy", x_ft[..., :n_mode_1, :n_mode_2], weight
        )
        out_ft[:, :, -n_mode_1:, :n_mode_2] = torch.einsum(
            "bixy,xyio->boxy", x_ft[:, :, -n_mode_1:, :n_mode_2], weight
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x


class FourierKNOConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        transform: str = "dft",
        img_size: int = 256,
        forward_transform=True,
        inverse_transform=True,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        https://arxiv.org/pdf/2010.08895.pdf
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._forawrd_transform = forward_transform
        self._inverse_transform = inverse_transform
        self.n_modes = n_modes
        (
            self.n_mode_1,
            self.n_mode_2,
        ) = n_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.device = device
        self.transform = transform
        assert transform in ["dft", "dct"], "transform must be dft or dct"

        if transform == "dft":
            assert forward_transform == True and inverse_transform == True, "dft must have both forward and inverse transform"
        self.scale = 1 / (in_channels * out_channels)

        ## for matmul
        # self.scale = ((1 / (in_channels)) * 256**2 / (2 * self.n_mode_1 * self.n_mode_2))**0.5
        ## for element-wise mul
        # if transform == "dct":
        #     self.scale = (img_size**2 / (2 * self.n_mode_1 * self.n_mode_2)) ** 0.5
        # else:
        #     self.scale = (
        #         img_size**2 / (4 * self.n_mode_1 * self.n_mode_2 + 4)
        #     ) ** 0.5
        # self.scale = 1 / (in_channels)
        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
        if self.transform == "dft":
            self.kernel = nn.Parameter(
                torch.empty(
                    [2, 1, self.out_channels, *self.n_modes, 2],
                    device=self.device,
                )
            )
        else:
            self.kernel = nn.Parameter(
                torch.empty(
                    [2, 1, self.out_channels, *self.n_modes],
                    device=self.device,
                )
            )

    def reset_parameters(self) -> None:
        self.kernel.data.copy_(
            torch.randn_like(self.kernel.data).mul_(self.scale).add_(1)
        )

    def get_zero_padding(self, size, device):
        bs, h, w = (
            size[0],
            size[-2],
            (size[-1] // 2 + 1 if self.transform == "dft" else size[-1]),
        )
        return torch.zeros(
            bs,
            self.out_channels,
            h,
            w,
            dtype=torch.cfloat if self.transform == "dft" else torch.float,
            device=device,
        )

    def forward_transform(self, x: Tensor, dim=(-2, -1), enable = True) -> Tensor:
        if enable:
            with torch.cuda.amp.autocast(enabled=False):
                if self.transform == "dft":
                    if x.is_complex():
                        x_ft = torch.fft.fftn(x, dim=dim, norm="ortho")
                    else:
                        x_ft = torch.fft.rfftn(x, dim=dim, norm="ortho")
                elif self.transform == "dct":
                    x_ft = dctn(x, dim=dim, norm="ortho")
                else:
                    raise NotImplementedError
        else:
            x_ft = x
        return x_ft

    def inverse_transform(
        self, x_ft: Tensor, s=None, dim=(-2, -1), real: bool = True, enable = True
    ) -> Tensor:
        if enable:
            with torch.cuda.amp.autocast(enabled=False):
                if self.transform == "dft":
                    if real:
                        x = torch.fft.irfftn(x_ft, s=s, dim=dim, norm="ortho")
                    else:
                        x = torch.fft.ifftn(x_ft, dim=dim, norm="ortho")
                elif self.transform == "dct":
                    x = idctn(x_ft, dim=dim, norm="ortho")
                else:
                    raise NotImplementedError
        else:
            x = x_ft
        return x

    def forward(self, x: Tensor, r: float) -> Tensor:
        if self.transform == "dft":
            kernel = torch.view_as_complex(self.kernel)
        else:
            kernel = self.kernel
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if 0 and self.n_mode_1 >= x.size(-2) // 2 and self.n_mode_2 >= x.size(-1) // 2:
            ## full mode, we can use 3d fft
            # x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
            x_ft = self.forward_transform(x, dim=(-3, -2, -1))
            out_ft = self.get_zero_padding(x.size(), x.device)
            n_mode_1 = min(x_ft.shape[-2] // 2, self.n_mode_1)
            n_mode_2 = min(x_ft.shape[-1], self.n_mode_2)
            kernel = kernel.pow(r)
            out_ft[..., :n_mode_1, :n_mode_2] = x_ft[..., :n_mode_1, :n_mode_2].mul(
                kernel[0]
            )
            out_ft[:, :, -n_mode_1:, :n_mode_2] = x_ft[..., -n_mode_1:, :n_mode_2].mul(
                kernel[1]
            )
            # x = torch.fft.irfftn(
            #     out_ft,
            #     s=(x.size(-3), x.size(-2), x.size(-1)),
            #     dim=[-3, -2, -1],
            #     norm="ortho",
            # )
            x = self.inverse_transform(
                out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), dim=[-3, -2, -1]
            )
        else:
            # x_ft = torch.fft.rfft2(x, norm="ortho")
            x_ft = self.forward_transform(x, dim=(-2, -1), enable=self._forawrd_transform)

            # Multiply relevant Fourier modes
            out_ft = self.get_zero_padding(x.size(), x.device)
            # out_ft = x.clone()
            # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
            n_mode_1 = min(x_ft.shape[-2] // (2 if self.transform == "dft" else 1), self.n_mode_1)
            n_mode_2 = min(x_ft.shape[-1], self.n_mode_2)
            # print(out_ft.shape, n_mode_1, n_mode_2, self.weight_1.shape)

            kernel = kernel.pow(r)

            # print(x_ft.shape, kernel.shape, n_mode_1, n_mode_2, self.forward_transform(x_ft[..., :n_mode_1, :n_mode_2], dim=-3).shape)
            if self.transform == "dft":
                out_ft[..., :n_mode_1, :n_mode_2] = self.inverse_transform(
                    self.forward_transform(x_ft[..., :n_mode_1, :n_mode_2], dim=-3).mul(
                        kernel[0]
                    ),
                    dim=-3,
                    real=False,
                )
                out_ft[:, :, -n_mode_1:, :n_mode_2] = self.inverse_transform(
                    self.forward_transform(x_ft[..., -n_mode_1:, :n_mode_2], dim=-3).mul(
                        kernel[1]
                    ),
                    dim=-3,
                    real=False,
                )
            else:
                out_ft[..., :n_mode_1, :n_mode_2] = self.inverse_transform(
                    self.forward_transform(x_ft[..., :n_mode_1, :n_mode_2], dim=-3).mul(
                        kernel[0]
                    ),
                    dim=-3,
                    real=False,
                )

            # Return to physical space
            # x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
            x = self.inverse_transform(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1), enable=self._inverse_transform)
        return x

    def __repr__(self) -> str:
        str = f"FourierKNOConv2d("
        str += f"in_channels={self.in_channels}, "
        str += f"out_channels={self.out_channels}, "
        str += f"n_modes={self.n_modes}, "
        str += f"transform={self.transform}, "
        str += f"fw_trans={self._forawrd_transform}, "
        str += f"inv_trans={self._inverse_transform})"
        return str

class FourierKNOConv2dFixSz(nn.Module):
    def __init__(
        self,
        width: int,
        n_modes: Tuple[int],
        transform: str = "dft",
        img_size: int = 256,
        forward_transform=True,
        inverse_transform=True,
        device: Device = torch.device("cuda:0"),
        interpolate: bool = False,
    ) -> None:
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        https://arxiv.org/pdf/2010.08895.pdf
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        """

        self.width = width
        self._forawrd_transform = forward_transform
        self._inverse_transform = inverse_transform
        self.interpolate = interpolate
        self.n_modes = n_modes
        (
            self.n_mode_1,
            self.n_mode_2,
        ) = n_modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.device = device
        self.transform = transform
        assert transform in ["dft", "dct"], "transform must be dft or dct"

        if transform == "dft":
            assert forward_transform == True and inverse_transform == True, "dft must have both forward and inverse transform"
        self.scale = 1 / (width * width)

        ## for matmul
        # self.scale = ((1 / (in_channels)) * 256**2 / (2 * self.n_mode_1 * self.n_mode_2))**0.5
        ## for element-wise mul
        # if transform == "dct":
        #     self.scale = (img_size**2 / (2 * self.n_mode_1 * self.n_mode_2)) ** 0.5
        # else:
        #     self.scale = (
        #         img_size**2 / (4 * self.n_mode_1 * self.n_mode_2 + 4)
        #     ) ** 0.5
        # self.scale = 1 / (in_channels)
        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
        if self.transform == "dft":
            self.kernel = nn.Parameter(
                torch.empty(
                    [2, 1, self.width, *self.n_modes, 2],
                    device=self.device,
                )
            )
        else:
            self.kernel = nn.Parameter(
                torch.empty(
                    [2, 1, self.width, *self.n_modes],
                    device=self.device,
                )
            )

    def reset_parameters(self) -> None:
        self.kernel.data.copy_(
            torch.randn_like(self.kernel.data).mul_(self.scale).add_(1)
        )

    def get_zero_padding(self, size, device):
        bs, f, h, w = (
            size[0],
            size[-3],
            size[-2],
            (size[-1] // 2 + 1 if self.transform == "dft" else size[-1]),
        )
        return torch.zeros(
            bs,
            f,
            h,
            w,
            dtype=torch.cfloat if self.transform == "dft" else torch.float,
            device=device,
        )

    def forward_transform(self, x: Tensor, dim=(-2, -1), enable = True) -> Tensor:
        if enable:
            with torch.cuda.amp.autocast(enabled=False):
                if self.transform == "dft":
                    if x.is_complex():
                        x_ft = torch.fft.fftn(x, dim=dim, norm="ortho")
                    else:
                        x_ft = torch.fft.rfftn(x, dim=dim, norm="ortho")
                elif self.transform == "dct":
                    x_ft = dctn(x, dim=dim, norm="ortho")
                else:
                    raise NotImplementedError
        else:
            x_ft = x
        return x_ft

    def inverse_transform(
        self, x_ft: Tensor, s=None, dim=(-2, -1), real: bool = True, enable = True
    ) -> Tensor:
        if enable:
            with torch.cuda.amp.autocast(enabled=False):
                if self.transform == "dft":
                    if real:
                        x = torch.fft.irfftn(x_ft, s=s, dim=dim, norm="ortho")
                    else:
                        x = torch.fft.ifftn(x_ft, dim=dim, norm="ortho")
                elif self.transform == "dct":
                    x = idctn(x_ft, dim=dim, norm="ortho")
                else:
                    raise NotImplementedError
        else:
            x = x_ft
        return x

    def forward(self, x: Tensor, r: float) -> Tensor:
        _, frames, _, _ = x.shape
        # if self.transform == "dft":
        #     kernel = torch.view_as_complex(self.kernel) #(2, 1, self.width, *self.n_modes, 2)
        # else:
        #     kernel = self.kernel
        kernel = self.kernel

        x_ft = self.forward_transform(x, dim=(-2, -1), enable=self._forawrd_transform)

        # Multiply relevant Fourier modes
        out_ft = self.get_zero_padding(x.size(), x.device)
        # out_ft = x.clone()
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        n_mode_1 = min(x_ft.shape[-2] // (2 if self.transform == "dft" else 1), self.n_mode_1)
        n_mode_2 = min(x_ft.shape[-1], self.n_mode_2)
        # print(out_ft.shape, n_mode_1, n_mode_2, self.weight_1.shape)
        kernel_cal = kernel.pow(r)
        padded_kernel_cal = torch.cat([kernel_cal, kernel_cal[:, :, 0, ...].unsqueeze(dim=2)], dim=2) # 2，1， 128+1, 80, 80, 2
        padded_kernel_cal = padded_kernel_cal.squeeze(1).view(2, self.width+1, -1)
        padded_kernel_cal = padded_kernel_cal.permute(0, 2, 1)
        padded_kernel_cal = F.interpolate(padded_kernel_cal, size=(frames+1), mode='linear', align_corners=True) # 2, 80*80*2, frames+1
        kernel_cal = padded_kernel_cal[:, :, :-1] # 2, 80*80*2, frames
        kernel_cal = kernel_cal.permute(0, 2, 1) # 2, frames, 80*80*2
        kernel_cal = kernel_cal.view(2, 1, frames, *self.n_modes, 2) # 2, 1, frames, 80, 80, 2
        kernel_cal = kernel_cal.contiguous()
        kernel_cal = torch.view_as_complex(kernel_cal) if self.transform == "dft" else kernel_cal
        # print(x_ft.shape, kernel.shape, n_mode_1, n_mode_2, self.forward_transform(x_ft[..., :n_mode_1, :n_mode_2], dim=-3).shape)
        if self.transform == "dft":
            out_ft[..., :n_mode_1, :n_mode_2] = self.inverse_transform(
                self.forward_transform(x_ft[..., :n_mode_1, :n_mode_2], dim=-3).mul(
                    kernel_cal[0]
                ),
                dim=-3,
                real=False,
            )
            out_ft[:, :, -n_mode_1:, :n_mode_2] = self.inverse_transform(
                self.forward_transform(x_ft[..., -n_mode_1:, :n_mode_2], dim=-3).mul(
                    kernel_cal[1]
                ),
                dim=-3,
                real=False,
            )
        else:
            out_ft[..., :n_mode_1, :n_mode_2] = self.inverse_transform(
                self.forward_transform(x_ft[..., :n_mode_1, :n_mode_2], dim=-3).mul(
                    kernel_cal[0]
                ),
                dim=-3,
                real=False,
            )

        # Return to physical space
        # x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        x = self.inverse_transform(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1), enable=self._inverse_transform)
        return x

    def __repr__(self) -> str:
        str = f"FourierKNOConv2dFixSz("
        str += f"width={self.width}, "
        str += f"n_modes={self.n_modes}, "
        str += f"transform={self.transform}, "
        str += f"fw_trans={self._forawrd_transform}, "
        str += f"inv_trans={self._inverse_transform})"
        return str