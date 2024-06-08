'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-17 18:50:45
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-01-17 18:50:45
'''
import torch
from core.models.layers import FNOConv2d


def test_fno_conv2d():
    device = torch.device("cuda:0")
    layer = FNOConv2d(8, 8, (4, 4), device).to(device)
    x = torch.randn(1, 1, 32, 32, device=device)
    y = layer(x)
    print(y)


if __name__ == "__main__":
    test_fno_conv2d()
