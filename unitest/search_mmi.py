"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-05-10 17:58:39
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-05-10 22:29:12
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from angler.simulation import Simulation
from data.mmi.device_shape import MMI_NxM, mmi_3x3_L
from pyutils.general import ensure_dir


def L_pi(width, n_r, lambda_0):
    return 4 * n_r * width ** 2 / (3 * lambda_0)

def mmi4x4():
    N = 4
    n_port = 0
    # size = (26, 8)
    size = (27, 8)
    neff_sio2 = 1.445
    neff_si = 3.475
    wavelength = 1.55
    grid_step = 0.1
    # for l in np.linspace(27.25, 27.25, 1):
    # for l in np.linspace(27.25*2, 27.25*2, 1):
    for l in np.linspace(319, 319, 1):
        for w in np.linspace(14, 14, 1):
            # l = L_pi(w, neff_si, wavelength) * 3 / N - 2
            size = (l, w)
            print(size)
            mmi = MMI_NxM(
                N,
                N,
                box_size=size,
                wg_width=(0.48, 0.48),
                port_diff=(size[1] / N, size[1] / N),
                port_len=3,
                border_width=0.25,
                taper_len=10,
                taper_width=1.3,
                grid_step=grid_step,
                NPML=[30, 30],
                eps_bg=neff_sio2 ** 2,
                eps_r=neff_si ** 2,
            )
            # eps = mmi.trim_pml(mmi.epsilon_map)
            wavelength = 1.55
            c0 = 299792458  # speed of light in vacuum (m/s)
            source_amp = 1e-9  # amplitude of modal source (make around 1 for nonlinear effects)
            # neff_si = 3.48
            lambda0 = wavelength / 1e6  # free space wavelength (m)
            omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
            simulation = Simulation(omega, eps_r=mmi.trim_pml(mmi.epsilon_map), dl=grid_step, NPML=[30, 30], pol="Ez")
            simulation.add_mode(
                neff=neff_si,
                direction_normal="x",
                center=mmi.in_port_centers_px[n_port],
                width=int(2 * mmi.in_port_width_px[n_port]),
                scale=source_amp,
            )
            # simulation.add_mode(
            #     neff=neff_si,
            #     direction_normal="x",
            #     center=mmi.in_port_centers_px[n_port+1],
            #     width=int(2 * mmi.in_port_width_px[n_port+1]),
            #     scale=source_amp,
            # )
            simulation.setup_modes()
            # (Ex, Ey, Hz) = simulation.solve_fields()
            (Hx, Hy, Ez) = simulation.solve_fields()
            ## draw fields
            # simulation.fields["Ez"] = Ez
            simulation.plt_abs(outline=True, cbar=False)
            ensure_dir("./unitest/search_mmi4x4")
            plt.savefig(
                f"./unitest/search_mmi4x4/mmi{mmi.num_in_ports}x{mmi.num_in_ports}_size-{size[0]:.2f}x{size[1]:.2f}_p-{n_port}_simu.png",
                dpi=400,
            )
            
def mmi5x5():
    N = 5
    n_port = 2
    # size = (26, 8)
    size = (27, 8)
    neff_sio2 = 1.445
    neff_si = 3.475
    wavelength = 1.55
    grid_step = 0.05
    # for l in np.linspace(27.25, 27.25, 1):
    # for l in np.linspace(27.25*2, 27.25*2, 1):
    for l in np.linspace(55.5, 55.5, 1):
        for w in np.linspace(8, 8, 1):
            w = 6
            l = L_pi(w, neff_si, wavelength) * 3 / N - 2
            size = (l, w)
            print(size)
            mmi = MMI_NxM(
                N,
                N,
                box_size=size,
                wg_width=(1, 1),
                port_diff=(size[1] / N, size[1] / N),
                port_len=3,
                border_width=0.25,
                grid_step=grid_step,
                NPML=[30, 30],
                eps_bg=neff_sio2 ** 2,
                eps_r=neff_si ** 2,
            )
            # eps = mmi.trim_pml(mmi.epsilon_map)
            wavelength = 1.55
            c0 = 299792458  # speed of light in vacuum (m/s)
            source_amp = 1e-9  # amplitude of modal source (make around 1 for nonlinear effects)
            neff_si = 3.48
            lambda0 = wavelength / 1e6  # free space wavelength (m)
            omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
            simulation = Simulation(omega, eps_r=mmi.epsilon_map, dl=grid_step, NPML=[30, 30], pol="Ez")
            simulation.add_mode(
                neff=neff_si,
                direction_normal="x",
                center=mmi.in_port_centers_px[n_port],
                width=int(2 * mmi.in_port_width_px[n_port]),
                scale=source_amp,
            )
            simulation.setup_modes()
            # (Ex, Ey, Hz) = simulation.solve_fields()
            (Hx, Hy, Ez) = simulation.solve_fields()
            ## draw fields
            # simulation.fields["Ez"] = Ez
            simulation.plt_re(outline=False)
            ensure_dir("./unitest/search_mmi5x5")
            plt.savefig(
                f"./unitest/search_mmi5x5/mmi{mmi.num_in_ports}x{mmi.num_in_ports}_size-{size[0]:.2f}x{size[1]:.2f}_p-{n_port}_simu.png",
                dpi=300,
            )


def mmi8x8():
    N = 8
    n_port = 3
    wavelength = 1.55
    # size = (26, 8)
    size = (27, 8)
    neff_sio2 = 1.445
    neff_si = 3.475
    for l in np.linspace(107.6, 107.6, 1):
        for w in np.linspace(9.6, 9.6, 1):
            # w = 6
            # l = L_pi(w, neff_si, wavelength) * 3
            size = (l, w)
            print(size)
            grid_step = 0.05
            mmi = MMI_NxM(
                N,
                N,
                box_size=size,
                wg_width=(0.4, 0.4),
                port_diff=(size[1] / N, size[1] / N),
                port_len=3,
                border_width=0.25,
                grid_step=grid_step,
                NPML=[30, 30],
                eps_bg=neff_sio2 ** 2,
                eps_r=neff_si ** 2,
            )
            # eps = mmi.trim_pml(mmi.epsilon_map)
            wavelength = 1.55
            c0 = 299792458  # speed of light in vacuum (m/s)
            source_amp = 1e-9  # amplitude of modal source (make around 1 for nonlinear effects)
            # neff_si = 3.48
            # neff_si = 3.5
            lambda0 = wavelength / 1e6  # free space wavelength (m)
            omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
            simulation = Simulation(omega, eps_r=mmi.epsilon_map, dl=grid_step, NPML=[30, 30], pol="Ez")
            simulation.add_mode(
                neff=neff_si,
                direction_normal="x",
                center=mmi.in_port_centers_px[n_port],
                width=int(2 * mmi.in_port_width_px[n_port]),
                scale=source_amp,
            )
            simulation.setup_modes()
            # (Ex, Ey, Hz) = simulation.solve_fields()
            (Hx, Hy, Ez) = simulation.solve_fields()
            ## draw fields
            # simulation.fields["Ez"] = Ez
            simulation.plt_re(outline=False)
            ensure_dir(f"./unitest/search_mmi{N}x{N}")
            plt.savefig(
                f"./unitest/search_mmi{N}x{N}/mmi{mmi.num_in_ports}x{mmi.num_in_ports}_size-{size[0]:.2f}x{size[1]:.2f}_p-{n_port}_simu.png",
                dpi=400,
            )


if __name__ == "__main__":
    # mmi5x5()
    mmi4x4()
    # mmi8x8()
