"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-20 17:02:17
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-20 17:02:17
"""
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import nn
from torch.functional import Tensor

__all__ = ["FullFieldRegressionHead"]


class FullFieldRegressionHead(nn.Module):
    def __init__(self, size: Tuple[int], grid_step: float) -> None:
        super().__init__()
        self.size = size  # pixel dimension
        self.grid_step = grid_step  # in unit of um
        ## discretized propagation vectors
        self.z_vector = torch.arange(size[1]) * grid_step

    def forward_scatter(
        self, scattered_field_envelop: Tensor, incident_field_envelop: Tensor, wavelength: float
    ) -> Tensor:
        """(scattered E envelop + incident E envelop) * e^(j*k_0*z) = full field

        Args:
            scattered_field_envelop (Tensor): The envelops of the frequency-domain scatter electric field
            incident_field_envelop (Tensor): The envelops of the frequency-domain incident electric field
            wavelength (float): Vacuum wavelength in unit of um

        Returns:
            Tensor: frequency-domain full electric field
        """
        self.z_vector = self.z_vector.to(scattered_field_envelop.device)
        return (scattered_field_envelop + incident_field_envelop).mul(
            torch.exp(self.z_vector.mul(1j * 2 * np.pi / wavelength))
        )

    def forward_full(self, field_envelop: Tensor, epsilon: Tensor, wavelength: float) -> Tensor:
        """E envelop * e^(j*k_0*z) = full field

        Args:
            field_envelop (Tensor): The envelops of the frequency-domain full electric field
            wavelength (float): Vacuum wavelength in unit of um

        Returns:
            Tensor: frequency-domain full electric field
        """
        # field_envelop [bs, 1, h, w] complex
        self.z_vector = self.z_vector.to(field_envelop.device)
        # print(field_envelop.shape, self.z_vector.shape, wavelength.shape)
        return field_envelop.mul(
            torch.exp(
                self.z_vector.mul(1j * 2 * np.pi / wavelength.unsqueeze(1).unsqueeze(1) * epsilon.sqrt())
            )
        )
