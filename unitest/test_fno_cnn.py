'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-19 00:17:20
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-01-19 00:17:20
'''
import torch
from core.models import FNO2d, PermittivityEncoder


def test_fno2d():
    wavelength = 1.550
    device = torch.device("cuda:0")
    domain_size = (20, 100)
    grid_step = wavelength / 20
    domain_size = (4*grid_step, 4*grid_step)
    domain_size_pixel = [round(i / grid_step) for i in domain_size]
    layer = FNO2d(
        in_channels=2,
        dim=1,
        kernel_list=[1],
        kernel_size_list=[1],
        padding_list=[0],
        hidden_list=[128],
        mode_list=[(4, 4)],
        act_func="GELU",
        domain_size=domain_size,
        grid_step=grid_step,
        device=device,
    ).to(device)
    x = torch.randn(1, 1, *domain_size_pixel, dtype=torch.cfloat, device=device)
    y = layer(x, wavelength=wavelength)
    print(y.shape)
    print(y)


def test_permitivity_encoder():
    device = torch.device("cuda:0")
    h, w = 8, 8
    encoder = PermittivityEncoder(
        size=(h, w),
        regions=torch.tensor([[2, 2, 2, 2], [5, 5, 2, 2]]),
        valid_range=(2, 4, 0.001, 0.01),
        device=device,
    ).to(device)
    permitivity = encoder()
    print(permitivity)


if __name__ == "__main__":
    test_fno2d()
    # test_permitivity_encoder()
