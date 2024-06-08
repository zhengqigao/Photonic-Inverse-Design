"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-20 17:00:10
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-20 17:00:22
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import nn
from torch.functional import Tensor
from torch.types import Device

__all__ = ["PermittivityEncoder"]


class PermittivityEncoder(nn.Module):
    def __init__(
        self,
        size: Tuple[int],
        regions: Tensor,
        valid_range: Tuple[float] = [1, 5, 0, 1],  # [min_real, max_real, min_imag, max_imag] of permittivity
        device: Device = torch.device("cuda:0"),
    ):
        """Leranable permutivity with controlling region encoding

        Args:
            size (Tuple[int]): Pixel-wise dimensions of the rectangular field
            regions (Tensor): Array of rectangular region description, each contains [y, x, size_y, size_x] in unit of pixel. Starting from 0.
        """
        super().__init__()
        self.size = size
        self.regions = regions
        self.valid_range = valid_range
        self.device = device

        self.build_parameters()
        self.reset_parameters()

    def build_parameters(self) -> None:
        ## num_regions learnable permittivity values
        self.weight = nn.Parameter(torch.empty(self.size[0], dtype=torch.cfloat, device=self.device))

        ## generate flattened gathering indices for each control region, the repeating times is the region area
        ## e.g., [e0,e1,e2].gather([0,0,1,1,1,1,2,2,2,2]) = [e0,e0,e1,e1,e1,e1,e2,e2,e2,e2]
        self.gathering_indices = []
        for i, (size_y, size_x) in enumerate(self.regions[:, 2:], 0):
            self.gathering_indices.extend([i] * int(size_y * size_x))
        self.gathering_indices = torch.tensor(self.gathering_indices).to(torch.long).to(self.device)

        ## generate flattened scattering indices for each control region
        ## e.g., [1,1,1,1].scatter(0, [1,3], [e1, e2]) = [1, e1, 1, e2]
        self.scattering_indices = []
        for i, (y, x, size_y, size_x) in enumerate(self.regions, 0):
            X = torch.arange(x, x + size_x)
            Y = torch.arange(y, y + size_y)
            Y, X = torch.meshgrid(Y, X)
            # convert from 2D coordinates to 1D absolute coordinate
            indices = Y.flatten() * self.size[1] + X.flatten()
            self.scattering_indices.append(indices)
        self.scattering_indices = torch.cat(self.scattering_indices, 0).to(torch.long).to(self.device)
        self.permittivity_field = torch.ones(self.size, dtype=torch.cfloat, device=self.device)

    def reset_parameters(self, bg_permittivity: float = 1) -> None:
        """reset background permittivity

        Args:
            bg_permittivity (float, optional): permittivity of the background material. Defaults to 1.
        """
        self.weight.data.fill_(-3)
        self.permittivity_field.fill_(bg_permittivity)

    def build_weight(self):
        weight_real = (
            self.weight.real.sigmoid().mul(self.valid_range[1] - self.valid_range[0]).add(self.valid_range[0])
        )
        weight_imag = (
            self.weight.imag.sigmoid().mul(self.valid_range[3] - self.valid_range[2]).add(self.valid_range[2])
        )
        weight = torch.complex(weight_real[self.gathering_indices], weight_imag[self.gathering_indices])

        weight = self.permittivity_field.view(-1).scatter(0, self.scattering_indices, weight).view(self.size)
        return weight

    def forward(self, wavelength: float = None) -> Tensor:
        """Epsilon_r(r, omega) is a complex-valued frequency-domain frequency-dependent field. We jointly model position and wavelength in Epsilon_r.\\
            Epsilon_r = epsilon' - j * epsilon'', epsilon'' >=0

        Args:
            wavelength (float): wavelength in unit of um

        Returns:
            Tensor: Complex-valued frequency-domain frequency-dependent relative permittivity field
        """
        weight = self.build_weight()
        ## TODO Will add wavelength-dependent modeling. Currently assume dispersion can be ignored within a certain range.
        return weight
