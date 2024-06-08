"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-16 21:19:03
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-30 21:12:29
"""
import numpy as np
import torch
from angler import Simulation
from data.mmi.device_shape import MMI_NxM
from pyutils.general import TimerCtx
from core.models import FFNO2d, FNO2d


def _test_runtime_angler(device):
    lambda0 = 1.55e-6  # free space wavelength (m)
    c0 = 299792458  # speed of light in vacuum (m/s)
    omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
    pol = "Hz"  # polarization (either 'Hz' or 'Ez')
    source_amp = 1e-9  # amplitude of modal source (make around 1 for nonlinear effects)
    eps_map = device.epsilon_map
    neff_si = 3.48
    simulation = Simulation(omega, eps_map, device.grid_step, device.NPML, pol)
    simulation.add_mode(
        neff=neff_si,
        direction_normal="x",
        center=device.in_port_centers_px[1],
        width=int(2 * device.in_port_width_px[1]),
        scale=source_amp,
    )
    simulation.setup_modes()
    (Ex, Ey, Hz) = simulation.solve_fields()


def test_runtime_ours():
    L = 12
    device = torch.device("cuda:0")

    scales = [1, 2, 4, 8, 16, 32, 64, 128]
    times = []
    domains = []
    bs = 1
    for scale in scales:
        size = (12 * scale ** 0.5 + 3 * 2, 3 * scale ** 0.5)
        domains.append(size[0] * size[1])
        shape = int(size[1] / 0.05), int(size[0] / 0.05)
        print(shape)
        mode = (int(shape[0] * 0.5), int(shape[1] * 0.182))
        print(mode)
        model = FFNO2d(
            in_channels=4,
            out_channels=2,
            dim=64,
            kernel_list=[64] * L,
            kernel_size_list=[1] * L,
            padding_list=[1] * L,
            hidden_list=[256],
            mode_list=[mode] * L,
            domain_size=(8, 38.4),
            grid_step=0.1,
            aug_path=False,
            device=device,
        ).to(device)
        # model = FNO2d(
        #     in_channels=4,
        #     out_channels=2,
        #     dim=64,
        #     kernel_list=[64]*L,
        #     kernel_size_list=[1]*L,
        #     padding_list=[0]*L,
        #     hidden_list=[256],
        #     mode_list=[(shape[0]//4, shape[1]//4)]*L,
        #     domain_size=(8, 38.4),
        #     grid_step=0.1,
        #     device=device,
        # ).to(device)

        model.eval()
        data = torch.randn(bs, 2, shape[0], shape[1], device=device, dtype=torch.cfloat)
        wavelength = torch.randn(bs, 1, device=device)
        grid_step = torch.randn(bs, 2, device=device)
        with torch.no_grad():
            for _ in range(2):
                y = model(data, wavelength, grid_step).detach()
        torch.cuda.synchronize()
        with TimerCtx() as t:
            with torch.no_grad():
                for _ in range(3):
                    y = model(data, wavelength, grid_step).detach()
            torch.cuda.synchronize()
        tt = t.interval / 3
        print(tt)
        times.append(tt)
        del model
        torch.cuda.empty_cache()
    print(domains)
    print(times)


def test_runtime_angler():
    scales = [1, 2, 4, 8, 16, 32, 64, 128]
    times = []
    domains = []
    for scale in scales:
        N = 3
        wl = 1.55
        index_si = 3.48
        size = (12 * scale ** 0.5, 3 * scale ** 0.5)
        port_len = 3
        domains.append((size[0] + port_len * 2) * size[1])
        mmi = MMI_NxM(
            N,
            N,
            box_size=size,  # box [length, width], um
            wg_width=(wl / index_si / 2, wl / index_si / 2),  # in/out wavelength width, um
            port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
            port_len=port_len,  # length of in/out waveguide from PML to box. um
            border_width=0.25,  # space between box and PML. um
            grid_step=0.05,  # isotropic grid step um
            NPML=(30, 30),  # PML pixel width. pixel
        )
        with TimerCtx() as t:
            _test_runtime_angler(mmi)
        times.append(t.interval)
    print(domains)
    print(times)


def test_runtime_ours_mmi3x3():
    L = 12
    device = torch.device("cuda:0")

    times = []
    domains = []
    bs = 1
    model = FFNO2d(
        in_channels=4,
        out_channels=2,
        dim=64,
        kernel_list=[64] * L,
        kernel_size_list=[1] * L,
        padding_list=[1] * L,
        hidden_list=[256],
        mode_list=[(40, 70)] * L,
        domain_size=(8, 38.4),
        grid_step=0.1,
        aug_path=False,
        conv_stem=True,
        ffn=True,
        # ffn_dwconv=False,
        device=device,
    ).to(device)

    model.eval()
    data = torch.randn(bs, 2, 80, 384, device=device, dtype=torch.cfloat)
    wavelength = torch.randn(bs, 1, device=device)
    grid_step = torch.randn(bs, 2, device=device)
    with torch.no_grad():
        for _ in range(2):
            y = model(data, wavelength, grid_step).detach()
    torch.cuda.synchronize()
    with TimerCtx() as t:
        with torch.no_grad():
            for _ in range(5):
                y = model(data, wavelength, grid_step).detach()  # .sum(dim=0)

        torch.cuda.synchronize()
    tt = t.interval / 5
    print(tt)
    times.append(tt)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # test_runtime_angler()
    # test_runtime_ours()
    test_runtime_ours_mmi3x3()
