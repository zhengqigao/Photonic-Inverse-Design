'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 15:49:44
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-27 23:15:54
'''
from data.mmi.device_shape import MMI_NxM, mmi_3x3_L
from core.models.pde_base import PDE_NN_BASE
import torch
import matplotlib.pyplot as plt
from core.utils import plot_compare
from core.datasets.mixup import MixupAll
def plot_mixup():
    raw_data = torch.load("./data/mmi/random_size/test.pt")
    # print(raw_data)
    wavelength, grid_step, epsilon, fields, eps_min, eps_max = raw_data
    print(wavelength.shape, grid_step.shape, epsilon.shape, fields.shape)
    wavelength = wavelength[0]
    grid_step = grid_step[0]
    eps = epsilon[0,:,0] # [3, 80, 384]
    mode = epsilon[0,:,1] # [3, 80, 384]
    target = fields[0,:,0] # [3, 80, 384]
    plot_compare(
        wavelength,
        grid_step,
        eps,
        mode,
        target,
        "./plot/mixup_single_fields.png",
        norm=False,
    )
    mixup = MixupAll(mode="elem")
    epsilon, fields = mixup(epsilon[0:1], fields[0:1], random_state=43, vflip=False)
    eps = epsilon[0,:,0] # [3, 80, 384]
    mode = epsilon[0,:,1] # [3, 80, 384]
    target = fields[0,:,0] # [3, 80, 384]
    plot_compare(
        wavelength,
        grid_step,
        eps,
        mode,
        target,
        "./plot/mixup_mixed_fields.png",
        norm=False,
    )
if __name__ == "__main__":
    plot_mixup()
