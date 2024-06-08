'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-10 18:35:01
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-10 20:23:41
'''

import wave
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from core.utils import CurlLoss
raw_data = torch.load("./data/mmi2x2/processed/training.pt")
wavelength, epsilon, fields = raw_data
print(wavelength.shape, epsilon.shape, fields.shape)
Ex, Ey, Ez, Hx, Hy, Hz = [i[0:1, 0:1] for i in fields.chunk(6, dim=1)]

epsilon = epsilon[0:1,0:1]
kernel_x = torch.tensor([-1.,1.]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
kernel_y = torch.tensor([-1.,1.]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
Ex_eps = F.pad(Ex*epsilon, pad=(0,1,0,0))
Ey_eps = F.pad(Ey*epsilon, pad=(0,0,0,1))
# dx_Ex = torch.complex(F.conv2d(Ex_eps.real, kernel_x), F.conv2d(Ex_eps.imag, kernel_x))
dy_Ey = torch.complex(F.conv2d(Ey_eps.real, kernel_y), F.conv2d(Ey_eps.imag, kernel_y))
beta = 2*np.pi * epsilon**0.5 / 1.55
div =  dy_Ey + 1j*beta * Ex*epsilon
print(div.abs().mean())
print(Ex.mul(epsilon).abs().mean())

loss = CurlLoss(grid_step=0.1, wavelength=1.55)
err = loss(fields[0:1, [0,1,5]], epsilon)
print(err)
