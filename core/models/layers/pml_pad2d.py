"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-20 17:03:42
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-20 17:03:42
"""

from torch import nn
from torch.functional import Tensor

__all__ = ["PerfectlyMatchedLayerPad2d"]


class PerfectlyMatchedLayerPad2d(nn.Module):
    def __init__(
        self,
        pml_width: float,
        buffer_width: float,
        grid_step: float,
        pml_permittivity: float = 0,
        buffer_permittivity: float = -1e10,
    ) -> None:
        super().__init__()
        self.pml_width = pml_width
        self.buffer_width = buffer_width
        self.padding_size = [round(pml_width / grid_step) + round(buffer_width / grid_step)] * 4
        self.buffer_permittivity = buffer_permittivity
        self.pml_permittivity = pml_permittivity
        self.buffer_padding_op = nn.ConstantPad2d(
            [round(buffer_width / grid_step)] * 4, value=buffer_permittivity
        )
        self.pml_padding_op = nn.ConstantPad2d([round(pml_width / grid_step)] * 4, value=pml_permittivity)

    def forward(self, x: Tensor) -> Tensor:
        """Pad the permittivity in the buffer and PML region

        Args:
            x (Tensor): Complex-valued permittivity field

        Returns:
            Tensor: Padded complex-valued permittivity field
        """
        return self.pml_padding_op(self.buffer_padding_op(x))

    def trim(self, x: Tensor) -> Tensor:
        yl = self.padding_size[2]
        yh = x.size(-2) - self.padding_size[3]
        xl = self.padding_size[0]
        xh = x.size(-1) - self.padding_size[1]
        return x[..., yl:yh, xl:xh]
