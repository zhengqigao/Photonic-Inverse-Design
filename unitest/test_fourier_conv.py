'''
Date: 2024-04-19 13:08:33
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-19 13:09:54
FilePath: /NeurOLight_Local/unitest/test_fourier_conv.py
'''
"""
Date: 2024-04-13 14:58:33
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-13 14:59:55
FilePath: /NeurOLight_Local/unitest/test_fourier_conv2d.py
"""

import torch
from core.models.layers import FourierConv3d, FourierConv2d
from pyutils.config import configs
from core import builder

def test_fourier_conv3d():
    device = torch.device("cuda:0")
    layer = FourierConv3d(
        8,
        8,
        (5, 7, 7),
        padding=True,
        r=1,
        is_causal=True,
        mask_shape="cone",
        device=device,
    ).to(device)
    layer.reset_parameters()
    x = torch.randn(1, 1, 32, 32, 32, device=device)
    y = layer(x)
    print(y)

def test_fourier_conv2d():
    device = torch.device("cuda:0")
    layer = FourierConv2d(
        8,
        8,
        (7, 7),
        padding=True,
        r=1,
        is_causal=True,
        mask_shape="cylinder",
        device=device,
    ).to(device)
    layer.reset_parameters()
    x = torch.randn(1, 8, 32, 32, device=device)
    y = layer(x)
    print(y)

def test_fourier_cnn():
    device = torch.device("cuda:0")
    x = torch.randn(1, 59, 256, 256, device=device)
    configs.load("./configs/fdtd/cnn/train_random/train.yml", recursive=True)
    print(configs)
    model = builder.make_model(device, configs.run.random_state)
    print(model)
    y = model(x)


    configs.model.backbone_cfg.conv_cfg = dict(type="Conv2d", padding_mode="replicate")
    configs.model.backbone_cfg.kernel_list = [96]

    model = builder.make_model(device, configs.run.random_state)
    print(model)
    y = model(x)

if __name__ == "__main__":
    test_fourier_conv2d()
    test_fourier_conv3d()
    test_fourier_cnn()
