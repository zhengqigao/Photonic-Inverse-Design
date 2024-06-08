'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-15 16:49:27
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-01-20 03:53:59
'''
import torch
import numpy as np
from core.utils import FaradayResidue2d
import matplotlib.pyplot as plt
from core.models.simulator import Simulation2D, maxwell_residual_2d_complex
from core.models.constant import *

def test_faradayresidue2d():
    device=torch.device("cuda:0")
    E = torch.ones(1, 2, 8, 8, device=device)
    epsilon = torch.ones(1, 2, 8, 8, device=device)
    loss = FaradayResidue2d(wavelength = 1550)
    y = loss(E, epsilon)
    print(y)


def validate_faraday_loss():
    N = 64
    grid_step = 0.05
    wavelength = 1.550
    epsilon = torch.zeros(1,N,N, dtype=torch.float)
    epsilon.data.fill_(eps_si)
    simulator = Simulation2D(
        mode="Hz",
        device_length = N,
        npml = 0,
        use_dirichlet_bcs=True,
    )

    permittivities, src_x, src_y, Ex, Ey, Hz = simulator.solve(epsilon, omega=OMEGA_1550, clip_buffers=True, src_x = 1)
    source = torch.zeros(1,1,N,N, dtype=torch.cfloat)
    source[..., N//2, 1] = -1j*MU0 * 1e6 * wavelength
    device = torch.device("cpu:0")
    Ex = torch.from_numpy(Ex).to(torch.cfloat).unsqueeze(0).unsqueeze(0)

    sim = Simulation2D(
            device_length=N,
            npml=2,
        )
    curl_curl_op, eps_op = sim.get_operators()

    curl_curl_re = torch.tensor(np.asarray(np.real(curl_curl_op)), device=device).float()
    curl_curl_im = torch.tensor(np.asarray(np.imag(curl_curl_op)), device=device).float()
    print(curl_curl_re)
    print(curl_curl_re.shape)
    print(curl_curl_im)

    res_re, res_im = maxwell_residual_2d_complex(Ex.real, Ex.imag, epsilon, curl_curl_re, curl_curl_im)
    print((res_re.square()+res_im.square()).mean())
    loss = FaradayResidue2d(wavelength = 1550, grid_step=grid_step)
    y = loss(Ex, epsilon.to(torch.cfloat).unsqueeze(0), source)
    print(y)


def learn_faraday():
    import cv2
    device=torch.device("cuda:0")
    # img = cv2.imread('unitest/e_field_1.png')
    # h, w = img.shape[0], img.shape[1]
    # img = torch.from_numpy(img[..., :-1]).to(device).permute(2,0,1).unsqueeze(0)
    img = torch.load("unitest/wave_1.pt").transpose(-1,-2)
    img = img[0:1, -1, 10:-10, 10:-10]
    h, w = img.shape[-2:]
    img = img.to(torch.cfloat)


    N = 128
    E = torch.nn.Parameter(torch.randn(1, 1, h, w, device=device, dtype=torch.cfloat))
    E.data.copy_(img)
    E_init = E.data[0,0]
    intensity = E_init.abs().cpu().numpy()
    phase = E_init.angle().cpu().numpy()
    init_img = np.concatenate([intensity/np.max(intensity), phase/3.14], axis=1)

    epsilon = torch.ones(1, 1, h, w, device=device, dtype=torch.cfloat)*1.5
    epsilon -= 1j*0.00001
    # wavelength = 1/15/0.707*1e2 # nm
    wavelength = 1.550 # nm
    # grid_step = 1.550/7 # nm
    # grid_step = 1.550/9.43 # nm
    grid_step = 1.550/10 # nm
    # grid_step = 1.550/11.07 # nm
    # loss = FaradayResidue2d(wavelength = 1550)
    loss = FaradayResidue2d(grid_step = grid_step, wavelength = wavelength)
    optimizer = torch.optim.Adam([E], lr=2e-4)
    y = loss(E, epsilon)
    for i in range(10000):
        y = loss(E, epsilon)
        optimizer.zero_grad()
        y.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"step: {i}, loss = {y.item()}")
    E = E.data[0, 0]
    intensity = E.abs().cpu().numpy()
    phase = E.angle().cpu().numpy()
    img = np.concatenate([intensity/np.max(intensity), phase/3.14], axis=1)
    plt.imshow(np.concatenate([init_img, img], axis=0), cmap="gray")
    plt.savefig("intensity_phase.png")
    # plt.imshow(phase, cmap="gray")
    # plt.savefig("intensity.png")
if __name__ == "__main__":
    # test_faradayresidue2d()
    # learn_faraday()
    validate_faraday_loss()
