'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-26 22:17:36
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-07 16:11:10
'''
from angler import Simulation
from angler.derivatives import unpack_derivs
import numpy as np
N=8
omega = 1.12e15

perms = np.ones((N,N), dtype = np.float64)
dl = 1
sim = Simulation(omega, perms, dl, [1, 1], "Hz", L0 = 1e-6)

Dyb, Dxb, Dxf, Dyf = unpack_derivs(sim.derivs)

curl_curl = (Dxf @ Dxb + Dyf @ Dyb)
with np.printoptions(precision=2):
    res = curl_curl.todense().tolist()
    for l in res:
        print(l)

