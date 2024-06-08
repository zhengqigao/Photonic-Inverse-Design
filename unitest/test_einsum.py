'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-10 16:56:20
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-05-10 16:32:04
'''
import torch
from pyutils.general import TimerCtx
from einops import einsum
device=torch.device("cuda:0")
x = torch.randn(12, 32, 80, 191).to(device)
w = torch.randn(32, 32, 191).to(device)
w1 = torch.randn(191, 32, 32).to(device)
for _ in range(5):
    y = torch.einsum("bixy,ioy->boxy", x, w)
torch.cuda.synchronize()
with TimerCtx() as t:
    for _ in range(5):
        y = torch.einsum("bixy,ioy->boxy", x, w)
    torch.cuda.synchronize()
print(t.interval / 5)


for _ in range(5):
    y = einsum(x, w, "b i x y, i o y -> b o x y")
torch.cuda.synchronize()
with TimerCtx() as t:
    for _ in range(5):
        y = einsum(x, w, "b i x y, i o y -> b o x y")
    torch.cuda.synchronize()
print(t.interval / 5)

for _ in range(5):
    # [12, 80, 191, 1, 32] x [191, 32, 32] -> [12, 80, 191, 1, 32]
    y = x.unsqueeze(1).permute(0, 3,4,1,2).matmul(w1).squeeze(-2).permute(0,3,1,2)
torch.cuda.synchronize()
with TimerCtx() as t:
    for _ in range(5):
        # [12, 80, 191, 1, 32] x [191, 32, 32] -> [12, 80, 191, 1, 32]
        y = x.unsqueeze(1).permute(0, 3,4,1,2).matmul(w1).squeeze(-2).permute(0,3,1,2)
    torch.cuda.synchronize()
print(t.interval / 5)
