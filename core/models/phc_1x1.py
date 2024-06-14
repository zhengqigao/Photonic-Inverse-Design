import os
import matplotlib

matplotlib.rcParams["text.usetex"] = False
import math
import numpy as np
import torch
import torch.nn as nn
from .layers.phc_1x1_fdtd import PhC_1x1
# Determine the path to the directory containing device.py
device_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/fdtd'))

eps_sio2 = 1.44**2
eps_si = 3.48**2

__all__ = ["Repara_PhC_1x1"]

class Repara_PhC_1x1(nn.Module):
    ''' this class is used to optimize the PhC_1x1 device

    1. init a latent vector that represents the device
    2. reparameterize the latent vector to another vector in a differentiable way
    3. generate the permittivity from the latent vector
    4. calculate the transmittion efficiency of the device as the optimization objective
    5. use the adjoint method to calculate the gradient of the objective w.r.t. the permittivity
    6. calculate the gradient of the objective w.r.t. the latent vector
    7. optimize the latent vector using the gradient
    '''
    def __init__(
            self, 
            device_cfg, 
            sim_cfg, 
            purturbation=False,
            num_rows_perside=6,
            num_cols=8,
        ):
        super(Repara_PhC_1x1, self).__init__()
        self.device_cfg = device_cfg
        for key in self.device_cfg:
            if type(self.device_cfg[key]) == str:
                self.device_cfg[key] = eval(self.device_cfg[key])
        self.purturbation = purturbation
        self.a = torch.tensor(1.0)
        self.box_size = torch.tensor(device_cfg["box_size"], dtype=torch.float32)
        self.num_rows_perside = num_rows_perside
        self.num_cols = num_cols
        self.init_parameters()
        self.sim_cfg = sim_cfg
        for key in self.sim_cfg:
            if type(self.sim_cfg[key]) == str:
                self.sim_cfg[key] = eval(self.sim_cfg[key])
        self.resolution = sim_cfg["resolution"]

    def init_parameters(self):
        self.wd = torch.sqrt(torch.tensor(3)) # one is the mean and the other is the std
        self.s1 = torch.sqrt(torch.tensor(3))
        self.s2 = self.s1 + torch.sqrt(torch.tensor(3))
        init_position_up_side = torch.zeros(self.num_rows_perside, self.num_cols, 3) # the last dimension is for x, y, sigma, the std for Gaussian splatting
        for i in range(self.num_rows_perside):
            if i == 0:
                init_position_up_side[i, :, 1] = self.s1/2
            elif i == 1:
                init_position_up_side[i, :, 1] = self.s2/2
            else:
                init_position_up_side[i, :, 1] = self.s2/2 + (i - 1) * self.a / 2 * torch.sqrt(torch.tensor(3))
        for j in range(self.num_cols):
            init_position_up_side[:, j, 0] = j * self.a - (self.num_cols - 1) * self.a / 2

        for k in range(self.num_rows_perside):
            if k % 2 == 0:
                init_position_up_side[k, :, 0] -= self.a / 2

        init_position_down_side = init_position_up_side.clone()
        for i in range(self.num_rows_perside):
            init_position_down_side[i, :, 1] = -self.wd/2 - i * self.a / 2 * torch.sqrt(torch.tensor(3))
        init_position_up_side[:, :, 2] = 0.1
        init_position_down_side[:, :, 2] = 0.1
        self.hole_position = nn.Parameter(torch.cat((init_position_up_side, init_position_down_side), dim=0)) # [x_coordinats, y_coordinats, 2]
    
    @staticmethod
    def gaussian(x, y, x0, y0, A, sigma):
        return A * torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    def build_permittivity(self, backward=False):
        if self.purturbation:
            if not backward:
                position_purturbation = 0.07*torch.randn((self.hole_position.shape[0], self.hole_position.shape[1], 2), requires_grad=False)
                self.position_purturbation = position_purturbation
            else:
                position_purturbation = self.position_purturbation
            hole_position = self.hole_position[:, :, :2] + position_purturbation
        else:
            hole_position = self.hole_position[:, :, :2]

        # TODO make this a module
        # Define grid
        x = torch.linspace(-self.box_size[0]/2, self.box_size[0]/2, int(self.box_size[0] * self.resolution) + 1)
        y = torch.linspace(-self.box_size[1]/2, self.box_size[1]/2, int(self.box_size[1] * self.resolution) + 1)
        X, Y = torch.meshgrid(x, y)

        Z = torch.zeros_like(X)
        # TODO no for loop
        for i in range(hole_position.shape[0]):
            for j in range(hole_position.shape[1]):
                Z += self.gaussian(X, Y, hole_position[i, j, 0], hole_position[i, j, 1], 1, self.hole_position[i, j, 2])
        # TODO LSE to pick the max value, this is to approx the max value so that not too sparse
        # wirelength to approx
        permittivity = Z

        # # normalize the permittivity with the maximum value
        # permittivity = permittivity / torch.max(permittivity)

        # update device config with the new permittivity
        self.device_cfg['box_size'] = [int(self.box_size[0] * self.resolution)/self.resolution, int(self.box_size[1] * self.resolution)/self.resolution]

        self.permittivity_tensor_size = permittivity.shape

        return permittivity
    
    @staticmethod
    def binarize_projection(permittivity, T):
        result = torch.sigmoid((permittivity - 0.5) / T)
        return result

    def calculate_objective_and_gradient(self, permittivity, mode = "fdtd"):
        if mode == "fdtd":
            f0, grad = self._calculate_objective_and_gradient_fdtd(permittivity)
        else:
            raise NotImplementedError
            
        return f0, grad

    def _calculate_objective_and_gradient_fdtd(self, permittivity):
        device = PhC_1x1(**self.device_cfg)
        device.update_permittivity(permittivity, self.device_cfg["box_size"][0], self.device_cfg["box_size"][1])
        device.add_source(0)
        device.create_simulation(**self.sim_cfg)
        device.create_objective(0, 0)
        device.create_optimzation()
        f0, grad = device.obtain_objective_and_gradient()
        return f0, grad
    
    def backward(self, loss_list):
        # Compute gradients of permittivity w.r.t. the design variables
        # design_vars = [self.hole_position, self.box_size]
        # TODO: how to optimze the box_size? it still look strange to me
        # if the gradient at point (x, y) is -1, it means that the permittivity at (x, y) should be decreased
        # but in which way the box_size should be changed accordingly?
        for i in range(len(loss_list)):
            loss_list[i].backward() # first compute and accumulate the gradient of the loss w.r.t. the design variables
        f0, grad_permittivity = self.calculate_objective_and_gradient(self.permittivity) # obtain the gradient of the objective w.r.t. the permittivity

        if isinstance(grad_permittivity, np.ndarray): # make sure the gradient is torch tensor
            grad_permittivity = torch.tensor(grad_permittivity, dtype=torch.float32)

        if len(grad_permittivity.shape) == 2: # summarize the gradient along different frquencies
            grad_permittivity = torch.sum(grad_permittivity, dim=-1)

        grad_permittivity = grad_permittivity.view(*self.permittivity.shape) # reshape the gradient to the shape of the permittivity
        # grad_permittivity = torch.randn(self.permittivity_tensor_size) # for test

        design_vars = [self.hole_position]
        grad_latent = torch.autograd.grad(outputs=self.permittivity, inputs=design_vars, grad_outputs=grad_permittivity, allow_unused=True)

        # Assign the gradients back to the design variables
        for var, grad in zip(design_vars, grad_latent):
            if var.grad is None:
                var.grad = grad
            else:
                var.grad += grad
            print("this is the gradient of the hole position: ", var.grad)

    def forward(self, T):
        permittivity = self.build_permittivity() # update the permittivity and change the device config that fits the permittivity size
        self.permittivity = self.binarize_projection(permittivity, T)
        return self.hole_position

if __name__ == "__main__":
    init_device_cfg = dict(
        num_in_ports = 1,
        num_out_ports = 1,
        box_size = [19.8, 12],
        wg_width = (1.7320508076, 1.7320508076),
        port_diff = (4, 4),
        port_len = 3,
        taper_width = 1.7320508076,
        taper_len = 2,
        eps_r = 12.1104,
        eps_bg = 2.0736,
    )
    sim_cfg = dict(
        resolution = 20,
        border_width = [0, 1],
        PML = (2, 2),
        record_interval = 0.3,
        store_fields = ['Ez'],
        until = 250,
        stop_when_decay = False,
    )
    model = Repara_PhC_1x1(
            device_cfg=init_device_cfg, 
            sim_cfg=sim_cfg, 
            purturbation=False,
            num_rows_perside=6,
            num_cols=8,
        )
    f0, grad, hole_position = model(0.01)
    print(f0)
    print(grad)
    model.backward(grad)