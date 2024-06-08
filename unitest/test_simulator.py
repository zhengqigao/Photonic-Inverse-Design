'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-19 23:13:24
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-01-27 01:10:58
'''

import numpy as np
import torch
from core.models import FNO2d
from core.utils import IntensityMSELoss, FaradayResidue2d
import matplotlib.pyplot as plt
from core.models.simulator import Simulation2D
from core.models.constant import *
def test_simulator():
    import cv2
    device=torch.device("cuda:0")
    wavelength = 1.0

    # img = torch.load("unitest/wave_1.pt").transpose(-1,-2)
    # img = img[0:1, -1, 10:-10, 10:-10]
    img = cv2.imread('unitest/e_field_1.png')
    N = 200
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
    img = cv2.resize(img, (N, N))
    h, w = img.shape[0], img.shape[1]
    img = torch.from_numpy(img).to(device).unsqueeze(0).unsqueeze(0)
    h, w = img.shape[-2:]
    img = img.to(torch.cfloat)
    print(h, w)

    epsilon = cv2.imread('unitest/epsilon_field_1.png')
    epsilon = cv2.cvtColor(epsilon, cv2.COLOR_BGR2GRAY)/255
    epsilon = cv2.resize(epsilon, (N, N))
    epsilon = torch.from_numpy(epsilon).to(device)

    epsilon = (epsilon.sub(0.5).sign()+1)/2*0.53+1
    epsilon = epsilon.to(torch.cfloat)
    epsilon.data.fill_(1)

    grid_step = 0.05
    domain_size = (h*grid_step, w*grid_step)
    domain_size_pixel = [round(i / grid_step) for i in domain_size]
    print(domain_size_pixel)
    simulator = Simulation2D(
        mode="Hz",
        device_length = N,
        npml = 0,
        use_dirichlet_bcs=True,
    )

    permittivities, src_x, src_y, Ex, Ey, Hz = simulator.solve(epsilon.real.cpu().numpy()*eps_si, omega=OMEGA_1550, clip_buffers=True, src_x = 1)

    intensity = np.abs(Ey)
    phase = np.angle(Ey)
    def normalize(x):
        x_mean = np.mean(x)
        x_std = np.std(x)
        x = (x - x_mean) / x_std
        return x
    img = np.concatenate([normalize(epsilon.real.cpu().numpy()), normalize(intensity), normalize(phase)], axis=1)
    plt.imshow(img, cmap="viridis")
    plt.savefig("simu_Hz_Ey.png")


def test_simulator_mmi():
    import cv2
    device = torch.device("cuda:0")
    wavelength, epsilon, field = torch.load("./data/mmi2x2/processed/training.pt")
    epsilon = epsilon.to(device)
    wavelength = wavelength.to(device).item()
    h, w = epsilon.shape[-2:]
    epsilon = epsilon[0,0, :,:h].real
    grid_step = 0.1
    domain_size = (h * grid_step, w * grid_step)
    domain_size_pixel = [round(i / grid_step) for i in domain_size]

    simulator = Simulation2D(
        mode="Hz",
        device_length = h,
        npml = 0,
        use_dirichlet_bcs=True,
    )

    permittivities, src_x, src_y, Ex, Ey, Hz = simulator.solve2(epsilon.cpu().numpy(), omega=OMEGA_1550, clip_buffers=True, src_x = [1,10], src_y=[150,175])

    intensity = np.abs(Ex)
    phase = np.angle(Ex)
    def normalize(x):
        x_mean = np.mean(x)
        x_std = np.std(x)
        x = (x - x_mean) / x_std
        return x
    img = np.concatenate([normalize(epsilon.cpu().numpy()), normalize(intensity), normalize(phase)], axis=1)
    plt.imshow(img, cmap="viridis")
    plt.savefig("simu_Hz_Ex_mmi.png")


if __name__ == "__main__":
    # test_simulator()
    test_simulator_mmi()
