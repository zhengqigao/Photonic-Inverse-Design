'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-08 20:19:21
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-12-15 16:49:59
'''
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from data.mmi.device_shape import mmi_3x3, MMI_NxM

def smatrix_eme():
    raw_data = torch.load("./data/mmi2x2/processed/training.pt")
    wavelength, epsilon, fields = raw_data
    print(wavelength.shape, epsilon.shape, fields.shape)
    Ex, Ey, Ez, Hx, Hy, Hz = [i[0, 0] for i in fields.chunk(6, dim=1)]
    fig, axes = plt.subplots(3, 1)
    r1 = Ex.real.abs().max()
    r2 = Ey.real.abs().max()
    r3 = Hz.real.abs().max()
    h1 = axes[0].imshow(Ex.real, vmin=-r1, vmax=r1, cmap="RdBu")
    h2 = axes[1].imshow(Ey.real, vmin=-r2, vmax=r2, cmap="RdBu")
    h3 = axes[2].imshow(Hz.real, vmin=-r3, vmax=r3, cmap="RdBu")
    for j in range(3):
        divider = make_axes_locatable(axes[j])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar([h1, h2, h3][j], label="Ey", ax=axes[j], cax=cax)
    plt.savefig("./eme_simu.png", dpi=200)
    for name, field in {"Ex": Ex, "Ey": Ey, "Hz": Hz}.items():
        t22 = field[54:59, -5].mean() / field[54:59, 5].mean()
        t12 = field[22:27, -5].mean() / field[54:59, 5].mean()
        beta = 2 * np.pi * epsilon.real.max().item() ** 0.5 / wavelength[0, 0].item()
        ref_phase = ((Ex.shape[1] - 10) * 0.1 * beta) % (2 * np.pi)
        print(f"Field {name}")
        print("S22: ", t22.item(), "S12: ", t12.item())
        print("|S22|: ", t22.abs().item(), "|S12|: ", t12.abs().item())
        print("Arg(S22): ", t22.angle().item()%(2*np.pi), "Arg(S12): ", t12.angle().item()%(2*np.pi))
        print("Arg(S22)-ref: ", t22.angle().item() - ref_phase, "Arg(S12): ", t12.angle().item() - ref_phase)
        print("Arg(S22)-Arg(S12): ", t22.angle().item() - t12.angle().item())
        print("")

    t22 = sum(f[56, -5].abs().square().item() for f in [Ex, Ey, Hz]) / sum(f[56, 5].abs().square().item() for f in [Ex, Ey, Hz])
    t12 = sum(f[24, -5].abs().square().item() for f in [Ex, Ey, Hz]) / sum(f[56, 5].abs().square().item() for f in [Ex, Ey, Hz])
    print(f"S22^2: {t22}")
    print(f"S12^2: {t12}")

def smatrix_angler():
    import numpy as np
    import matplotlib.pylab as plt
    import copy

    # add angler to path (not necessary if pip installed)
    import sys
    import torch

    # import the main simulation and optimization classes
    from angler import Simulation

    # define the similation constants
    lambda0 = 1.55e-6                   # free space wavelength (m)
    c0 = 299792458                      # speed of light in vacuum (m/s)
    omega = 2 * np.pi * c0 / lambda0    # angular frequency (2pi/s)
    dl = 0.1                            # grid size (L0) um
    NPML = [10, 10]                     # number of pml grid points on x and y borders
    pol = "Hz"                          # polarization (either 'Hz' or 'Ez')
    source_amp = 1e-9                   # amplitude of modal source (make around 1 for nonlinear effects)
    ext_wg = 10                         # length os extended input waveguide (um)
    wg = 0.4                            # waveguide width (um)
    port1 = 2.5                         # input waveguide center location (um)
    port2 = 5.6                         # input waveguide center location (um)
    mode_center_1 = 2.5
    mode_center_2 = 5.7
    mode_width = 0.5
    pads = ((14, 38, 2, 3.5), (14, 38, 4, 5.5))  # pad location (xl,xh,yl,yh) (um)
    eps_range = (11.9, 12.3)            # tuning range of Si permittivity

    # define permittivity of three port system
    # eps_r, design_region = N_port(N, L, H, w, d, l, spc, dl, NPML, eps_m)
    device = torch.device("cuda:0")
    wavelength, epsilon, _ = torch.load("./data/mmi2x2/processed/training.pt")
    print(wavelength.shape)
    eps_r = epsilon[0, 0].t().numpy().real
    print(eps_r.shape)
    # print(eps_r[1, :])
    # exit(0)
    ## add waveguide
    eps_si = eps_r.max()
    eps_sio2 = eps_r.min()

    ## binarize permittivities on SiO2 and Si boundary
    # eps_r[eps_r > (eps_si + eps_sio2) / 2] = eps_si
    # eps_r[eps_r <= (eps_si + eps_sio2) / 2] = eps_sio2

    ## extension of input waveguide
    eps_i = np.zeros((int(ext_wg / dl), eps_r.shape[1])) + eps_sio2
    start, end = round((port1 - wg / 2) / dl), round((port1 + wg / 2) / dl)-1
    print(start, end)
    eps_i[:, round((port1 - wg / 2) / dl) : round((port1 + wg / 2) / dl)-1] = eps_si
    eps_i[:, [start-1, end]] = 6.177296

    start, end = round((port2 - wg / 2) / dl)+1, round((port2 + wg / 2) / dl)
    print(start, end)
    eps_i[:, round((port2 - wg / 2) / dl)+1 : round((port2 + wg / 2) / dl)] = eps_si
    eps_i[:, [start-1, end]] = 6.177296
    eps_r = np.concatenate([eps_i, eps_r], axis=0)
    (Nx, Ny) = eps_r.shape

    n_points = 5


    simulation = Simulation(omega, eps_r, dl, NPML, pol)

    # plot the permittivity distribution
    # simulation.plt_eps(outline=False)
    # plt.savefig("angler_mmi_eps.png", dpi=300)
    # exit(0)

    l = 0.5
    ny = mode_center_2

    # set the input waveguide modal source
    simulation.add_mode(
        neff=np.sqrt(eps_si),
        direction_normal="x",
        center=[NPML[0] + int(l / 2 / dl), ny / dl],
        width=int(2 * mode_width / dl),
        scale=source_amp,
    )
    simulation.setup_modes()
    center = [NPML[0] + int(l / 2 / dl), ny / dl]
    width = int(2 * mode_width / dl)
    inds_x = [center[0], center[0]+1]
    inds_y = [int(center[1]-width/2), int(center[1]+width/2)]
    eigen_mode = simulation.src[inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]]
    
    fig, axes = plt.subplots(1, 1)
    axes.imshow(np.abs(simulation.src), cmap="RdBu")
    plt.savefig("./angler_simu_eigen_mode.png", dpi=200)
    
    print("Eigen mode source:", eigen_mode)
    print("largest mode source:", np.max(np.abs(simulation.src)))

    # set source and solve for electromagnetic fields
    (Ex, Ey, Hz) = simulation.solve_fields()
    SCALE = np.sum(np.abs(Hz * simulation.src)**2)**0.5
    print("Normalization SCALE: ", SCALE)
    eigen_mode /= SCALE
    Ex, Ey, Hz = [torch.from_numpy(i).to(torch.cfloat).t()[:,100:] for i in (Ex, Ey, Hz)]
    fig, axes = plt.subplots(3, 1)
    r1 = Ex.real.abs().max()
    r2 = Ey.real.abs().max()
    r3 = Hz.real.abs().max()
    h1 = axes[0].imshow(Ex.real, vmin=-r1, vmax=r1, cmap="RdBu")
    h2 = axes[1].imshow(Ey.real, vmin=-r2, vmax=r2, cmap="RdBu")
    h3 = axes[2].imshow(Hz.real, vmin=-r3, vmax=r3, cmap="RdBu")
    for j in range(3):
        divider = make_axes_locatable(axes[j])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar([h1, h2, h3][j], label=pol, ax=axes[j], cax=cax)
    plt.savefig("./angler_simu.png", dpi=200)
    # for name, field in {"Ex": Ex, "Ey": Ey, "Hz": Hz}.items():
    #     t22 = field[54:59, -5].mean() / field[54:59, 5].mean()
    #     t12 = field[22:27, -5].mean() / field[54:59, 5].mean()
    #     beta = 2 * np.pi * epsilon.real.max().item() ** 0.5 / wavelength[0, 0].item()
    #     ref_phase = ((Ex.shape[1] - 10) * 0.1 * beta) % (2 * np.pi)
    #     print(f"Field {name}")
    #     print("S22: ", t22.item(), "S12: ", t12.item())
    #     print("|S22|: ", t22.abs().item(), "|S12|: ", t12.abs().item())
    #     print("Arg(S22): ", t22.angle().item() - ref_phase, "Arg(S12): ", t12.angle().item() - ref_phase)
    #     print("Arg(S22)-Arg(S12): ", t22.angle().item() - t12.angle().item())
    #     print("")
    
    # s parameter extraction
    eigen_mode = torch.from_numpy(eigen_mode[0]) # / simulation.W_in
    print(eigen_mode)
    for name, field in {"Ex": Ex, "Ey": Ey, "Hz": Hz}.items():
        t22 = field[52:62, -5].dot(eigen_mode.conj()) / field[52:62, 5].dot(eigen_mode.conj())
        # t22 = field[54:59, -5].mean() / field[54:59, 5].mean()
        # t12 = field[22:27, -5].mean() / field[54:59, 5].mean()
        t12 = field[20:30, -5].dot(eigen_mode.conj()) / field[52:62, 5].dot(eigen_mode.conj())
        beta = 2 * np.pi * epsilon.real.max().item() ** 0.5 / wavelength[0, 0].item()
        ref_phase = ((Ex.shape[1] - 10) * 0.1 * beta) % (2 * np.pi)
        print(f"Field {name}")
        print("S22: ", t22.item(), "S12: ", t12.item())
        print("|S22|: ", t22.abs().item(), "|S12|: ", t12.abs().item())
        print("Arg(S22): ", t22.angle().item() - ref_phase, "Arg(S12): ", t12.angle().item() - ref_phase)
        print("Arg(S22)-Arg(S12): ", t22.angle().item() - t12.angle().item())
        print("")


def test_transfer_matrix_angler():
    size = (319, 14)
    N = 4
    n_port = 0
    neff_sio2 = 1.445
    neff_si = 3.475
    wavelength = 1.55
    grid_step = 0.1
    mmi = MMI_NxM(
                N,
                N,
                box_size=size,
                wg_width=(1.3, 1.3),
                port_diff=(size[1] / N, size[1] / N),
                port_len=3,
                border_width=0.25,
                grid_step=grid_step,
                NPML=[30, 30],
                eps_bg=neff_sio2 ** 2,
                eps_r=neff_si ** 2,
            )
    w, h = (300, 2)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    print(mmi)
    # matrix = mmi.extract_transfer_matrix(mmi.epsilon_map, wavelength=1.55, pol="Hz")
    # print(matrix)
    for delta_neff in np.arange(0, 0.03001, 0.006):
        eps = [3.48**2 + delta_neff, 3.48**2, 3.48**2, 3.48**2]
        eps_map = mmi.set_pad_eps(eps)
        matrix = mmi.extract_transfer_matrix(eps_map, wavelength=1.55, pol="Ez")
        print("\n", eps)
        print(torch.from_numpy(matrix))
    
   
if __name__ == "__main__":
    # smatrix_eme()
    # smatrix_angler()
    test_transfer_matrix_angler()
