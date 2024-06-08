'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-21 14:03:06
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-01-21 14:04:39
'''
import torch
import time
N = 32
W = torch.randn(N, N).cuda()
x = torch.randn(N, 1).cuda()
torch.cuda.synchronize()
for i in range(10):
    y = W.matmul(x)
start = time.time()
for i in range(100):
    y = W.matmul(x)
print((time.time()-start)/100)
