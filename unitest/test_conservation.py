'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-07 17:38:32
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-07 19:04:00
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
from angler.linalg import grid_average

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
eps_r[250:450, 30:51] = eps_si+0.1
(Nx, Ny) = eps_r.shape
# nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points

# make a new simulation object
simulation = Simulation(omega, eps_r, dl, NPML, pol)

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
ny = 56 / ds

# set the input waveguide modal source
simulation.add_mode(neff=np.sqrt(eps_si), direction_normal='x', center=[NPML[0]+int(l/2/dl), ny], width=int(2*w/dl), scale=source_amp)
simulation.setup_modes()

# set source and solve for electromagnetic fields
(Ex, Ey, Hz) = simulation.solve_fields()
print(Ex.shape)
inds_x = [600, 601]
inds_y = [56-wg//2, 56+wg//2]
def prob(inds_x, inds_y):
    Hz_x = grid_average(Hz[inds_x[0]:inds_x[1]+1, inds_y[0]:inds_y[1]+1], 'x')[:-1, :-1]
    Sx = 1/2*np.real(Ey[inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]]*np.conj(Hz_x))
    flux = dl*np.sum(Sx)
    return flux

# inds_x = [600, 601]
# inds_y = [56-wg, 56+wg]
# print(prob(inds_x, inds_y))

# inds_x = [600, 601]
# inds_y = [25-wg, 25+wg]
# print(prob(inds_x, inds_y))





# inds_x = [15, 16]
# inds_y = [0, 79]
# print(prob(inds_x, inds_y))

# inds_x = [20, 21]
# inds_y = [0, 79]
# print(prob(inds_x, inds_y))

# inds_x = [25, 26]
# inds_y = [0, 79]
# print(prob(inds_x, inds_y))

# inds_x = [598, 599]
# inds_y = [0, 79]
# print(prob(inds_x, inds_y))

# inds_x = [600, 601]
# inds_y = [0, 79]
# print(prob(inds_x, inds_y))

for start in range(20, 600, 20):
    inds_x = [start, start+1]
    inds_y = [0, 79]
    print(prob(inds_x, inds_y))

