'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-03 14:40:12
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-03 16:14:50
'''
'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-27 01:21:54
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-03 13:16:50
'''
import numpy as np
import matplotlib.pylab as plt
import copy

# add angler to path (not necessary if pip installed)
import sys
import torch

from core.utils import CurlLoss, TMPropagateFieldLoss
sys.path.append("..")

# import the main simulation and optimization classes
from angler import Simulation, Optimization

# import some structure generators
from angler.structures import three_port, two_port
from angler.derivatives import unpack_derivs

# define the similation constants
lambda0 = 1.55e-6              # free space wavelength (m)
c0 = 3e8                    # speed of light in vacuum (m/s)
omega = 2*np.pi*c0/lambda0  # angular frequency (2pi/s)
dl = 0.1                    # grid size (L0)
NPML = [10, 10]             # number of pml grid points on x and y borders
# NPML = [0, 0]             # number of pml grid points on x and y borders
pol = 'Hz'                  # polarization (either 'Hz' or 'Ez')
source_amp = 1e-9           # amplitude of modal source (make around 1 for nonlinear effects)

# define material constants
n_index = 2.44              # refractive index
eps_m = n_index**2          # relative permittivity

# define permittivity of three port system
# eps_r, design_region = N_port(N, L, H, w, d, l, spc, dl, NPML, eps_m)
device = torch.device("cuda:0")
wavelength, epsilon, field = torch.load("./data/mmi2x2/processed/training.pt")
eps_r = epsilon[0,0].t().numpy().real
print(eps_r.shape)
## add waveguide
eps_si = eps_r.max()
eps_sio2 = eps_r.min()
print(eps_si, eps_sio2)
extra_l = 100
eps_i = np.zeros((extra_l, eps_r.shape[1])) + eps_sio2
wg = 4
eps_i[:,23:23+wg] = eps_si
eps_i[:,54:54+wg] = eps_si
eps_r = np.concatenate([eps_i, eps_r], axis=0)


ds = 1
eps_r = eps_r[:, ::ds]
eps_r[eps_r>8] = eps_si
eps_r[eps_r<7.999] = eps_sio2
name = "angler_mmi_simu_0.1.png"
# eps_r[250:450, 30:51] = eps_si+0.1
(Nx, Ny) = eps_r.shape
# nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points

def simulate(nys, amps, pic_name):
    np.random.seed(0)
    import random
    random.seed(0)
    # make a new simulation object
    simulation = Simulation(omega, eps_r.copy(), dl, NPML, pol)

    print("Computed a domain with {} grids in x and {} grids in y".format(Nx,Ny))
    print("The simulation has {} grids per free space wavelength".format(int(lambda0/dl/simulation.L0)))

    # plot the permittivity distribution
    # simulation.plt_eps(outline=False)
    # plt.savefig("angler_mmi_eps.png")

    N = 2
    l = 2
    d = 10
    w = dl * wg / ds
    # ny = 59 / ds
    # ny = 55.5 / ds

    # set the input waveguide modal source
    print(nys)
    for ny, amp in zip(nys, amps):
        simulation.add_mode(neff=np.sqrt(eps_si), direction_normal='x', center=[NPML[0]+int(l/2/dl), ny], width=int(2*w/dl), scale=amp)
    simulation.setup_modes()

    # set source and solve for electromagnetic fields
    (Ex, Ey, Hz) = simulation.solve_fields()
    simulation.plt_re()
    plt.savefig(f"{pic_name}.png")
    return Ex, Ey, Hz
Ex_list = []
Ey_list = []
Hz_list = []
for i, (nys, amps) in enumerate(zip([[26,44], [56],  [26,56,44]], [[1e-9, 0.9e-9], [1.5e-9], [1e-9, 1.5e-9, 0.9e-9]])):
# for i, (nys, amps) in enumerate(zip([[26], [26], [26], [26]], [[1e-9], [1e-9], [1e-9], [1e-9]])):
    Ex, Ey, Hz = simulate(nys, amps, f"mode_{i}")
    Ex_list.append(Ex)
    Ey_list.append(Ey)
    Hz_list.append(Hz)
print(np.mean(np.abs(Ex_list[0])))
err = np.mean(np.abs(sum(Ex_list[:-1]) - Ex_list[-1]))
print(np.mean(np.abs(Ex_list[0])), np.mean(np.abs(Ex_list[1])))
# err = np.mean(np.abs(Ex_list[0] - Ex_list[1]/1.00))
print([Ex[0,0] for Ex in Ex_list])
print(err)


