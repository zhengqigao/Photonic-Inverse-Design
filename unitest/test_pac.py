'''
Date: 2024-05-12 14:37:52
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-05-12 19:00:23
FilePath: /NeurOLight_Local/unitest/test_pac.py
'''
"""
Date: 2024-05-10 16:39:05
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-05-10 16:40:14
FilePath: /NeurOLight_Local/unitest/test_pac.py
"""

import torch
from pyutils.general import TimerCtx, print_stat, TorchTracemalloc
from pyutils.torch_train import set_torch_deterministic

from core.models.layers.pac import PacConv2d


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = PacConv2d(32, 32, 9, padding=8, native_impl=True, dilation=2)
        self.conv2 = PacConv2d(32, 32, 9, padding=8, native_impl=True, dilation=2)
        # self.conv1 = PacConv2d(32, 32, 17, padding=8, native_impl=True, dilation=1)
        # self.conv2 = PacConv2d(32, 32, 17, padding=8, native_impl=True, dilation=1)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, f):
        return self.conv2(self.conv1(x, f), f)


def test_pac():
    device = "cuda:0"
    set_torch_deterministic(0)
    x = torch.randn(1, 32, 64, 64, device=device, requires_grad=True)
    f = torch.randn(1, 32, 64, 64, device=device, requires_grad=True)
    conv = Model().to(device)
    with TorchTracemalloc(True):
        y = conv(x, f)
    torch.cuda.empty_cache()
    y = conv(x, f)
    print_stat(y)
    y.sum().backward()

    torch.cuda.synchronize()
    with TimerCtx() as t:
        for _ in range(5):
            y = conv(x, f)
        torch.cuda.synchronize()
    print(t.interval / 5)


if __name__ == "__main__":
    test_pac()
