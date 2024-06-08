import jax
import numpy as np
import jax.experimental.sparse.linalg
import scipy.sparse as sp
from pyutils.general import TimerCtx
from angler import Simulation, Optimization
import torch
from angler.linalg import solver_complex2real, solver_direct
import pypardiso
from pypardiso.scipy_aliases import pypardiso_solver as ps
import jax.scipy.sparse.linalg
import scikits.umfpack
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cuda")
# define the similation constants
lambda0 = 1.55e-6  # free space wavelength (m)
c0 = 3e8  # speed of light in vacuum (m/s)
omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
dl = 0.1  # grid size (L0)
NPML = [10, 10]  # number of pml grid points on x and y borders
# NPML = [0, 0]             # number of pml grid points on x and y borders
pol = "Hz"  # polarization (either 'Hz' or 'Ez')
source_amp = 1e-9  # amplitude of modal source (make around 1 for nonlinear effects)

# define material constants
n_index = 2.44  # refractive index
eps_m = n_index**2  # relative permittivity

# define permittivity of three port system
# eps_r, design_region = N_port(N, L, H, w, d, l, spc, dl, NPML, eps_m)
device = torch.device("cuda:0")
# wavelength, epsilon, field = torch.load("./data/mmi2x2/processed/training.pt")
from data.mmi.device_shape import mmi_3x3_L_random

device = mmi_3x3_L_random()
eps_r = device.epsilon_map
# eps_r = epsilon[0,0].t().numpy().real
# print(eps_r.shape)
# ## add waveguide
# eps_si = eps_r.max()
# eps_sio2 = eps_r.min()
# print(eps_si, eps_sio2)
# extra_l = 100
# eps_i = np.zeros((extra_l, eps_r.shape[1])) + eps_sio2
# wg = 4
# eps_i[:, 23 : 23 + wg] = eps_si
# eps_i[:, 54 : 54 + wg] = eps_si
# eps_r = np.concatenate([eps_i, eps_r], axis=0)


# ds = 1
# eps_r = eps_r[:, ::ds]
# eps_r[eps_r>8] = eps_si
# eps_r[eps_r<7.999] = eps_sio2
name = "angler_mmi_simu_0.1.png"
# eps_r[250:450, 30:51] = eps_si+0.1
(Nx, Ny) = eps_r.shape
# nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points

# make a new simulation object
simulation = Simulation(omega, eps_r, dl, NPML, pol)
A = simulation.A


A = sp.rand(300, 300, density=0.05, format='csr', dtype=np.complex128)

b = np.random.rand(A.shape[0], 1).astype(np.complex128)

def pypardiso_spsolve_complex(A, b):

    b = b.astype(np.complex128)[..., np.newaxis].view(np.float64).transpose().reshape(-1)
    A = sp.vstack((sp.hstack((A.real, -A.imag)),
                       sp.hstack((A.imag, A.real))))
    A = sp.rand(600, 600, density=0.05, format='csr')
    b = np.random.rand(600)
    print(A.shape, b.shape)
    print(A, b)
    print(A.dtype, b.dtype)
    x = pypardiso.spsolve(A, b)
    return x


with TimerCtx() as t:
    # x1 = solver_direct(A, b, solver="pardiso")
    x1 = solver_complex2real(A, b, solver="pardiso")
    x1 = pypardiso_spsolve_complex(A, b)
print(x1)
print(t.interval)
# with TimerCtx() as t:
#     x3 = jax.experimental.sparse.linalg.spsolve(
#         A.data, A.indices, A.indptr, b, tol=1e-10
#     ).block_until_ready()
#     # x3 = jax.scipy.sparse.linalg.gmres(A, b)   
# print(t.interval)
# print(x3)

with TimerCtx() as t:
    x4 = scikits.umfpack.spsolve(A, b)
print(t.interval)
# print(x4)

with TimerCtx() as t:
    x2 = sp.linalg.spsolve(A, b)
print(t.interval)
# print(x2)


