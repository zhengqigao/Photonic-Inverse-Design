"""
Description:
Author: Pingchuan Ma (pingchua@asu.edu)
Date: 2024-02-06 20:46:39
LastEditors: Pingchuan Ma (pingchua@asu.edu)
LastEditTime: 2024-02-06 20:46:39
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
from torch.nn.utils import weight_norm
from .layers import *

__all__ = ['PhyCRNet']

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

# define the high-order finite difference kernels
lapl_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]

partial_y = [[[[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [1/12, -8/12, 0, 8/12, -1/12],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]]]

partial_x = [[[[0, 0, 1/12, 0, 0],
               [0, 0, -8/12, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 8/12, 0, 0],
               [0, 0, -1/12, 0, 0]]]]

# generalized version
def initialize_weights(module):
    ''' starting from small initialized parameters '''
    if isinstance(module, nn.Conv2d):
        c = 0.1
        module.weight.data.uniform_(-c*np.sqrt(1 / np.prod(module.weight.shape[:-1])),
                                     c*np.sqrt(1 / np.prod(module.weight.shape[:-1])))
     
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class ConvLSTMCell(nn.Module):
    ''' Convolutional LSTM '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding):

        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = 3
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.num_features = 4

        # padding for hidden state
        self.padding = int((self.hidden_kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='zeros')

        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='zeros')

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='zeros')

        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='zeros')

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='zeros')

        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='zeros')

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='zeros')

        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='zeros')        

        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden_tensor(self, prev_state):
        ''' initialize hidden states '''
        ## the input shape of the prev_state is (32, 2, 128, 32, 32)
        return (Variable(prev_state[:, 0, ...].squeeze()).cuda(), Variable(prev_state[:, 1, ...].squeeze()).cuda())


class encoder_block(nn.Module):
    ''' encoder with CNN '''
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding):
        
        super(encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size  
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.conv = weight_norm(nn.Conv2d(self.input_channels, 
            self.hidden_channels, self.input_kernel_size, self.input_stride, 
            self.input_padding, bias=True, padding_mode='circular'))

        self.act = nn.ReLU()

        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.act(self.conv(x))

class PhyCRNet(nn.Module):
    ''' physics-informed convolutional-recurrent neural networks '''
    def __init__(self, input_channels, hidden_channels, 
        input_kernel_size, input_stride, input_padding, dt, 
        num_layers, upscale_factor, step=1, effective_step=[1], out_frames=100):

        super(PhyCRNet, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells 
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.dt = dt
        self.upscale_factor = upscale_factor
        self.out_frames = out_frames

        # number of layers
        self.num_encoder = num_layers[0]
        self.num_convlstm = num_layers[1]

        # encoder - downsampling  
        for i in range(self.num_encoder):
            name = 'encoder{}'.format(i)
            cell = encoder_block(
                input_channels = self.input_channels[i], 
                hidden_channels = self.hidden_channels[i], 
                input_kernel_size = self.input_kernel_size[i],
                input_stride = self.input_stride[i],
                input_padding = self.input_padding[i])

            setattr(self, name, cell)
            self._all_layers.append(cell)            
            
        # ConvLSTM
        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            name = 'convlstm{}'.format(i)
            cell = ConvLSTMCell(
                input_channels = self.input_channels[i],
                hidden_channels = self.hidden_channels[i],
                input_kernel_size = self.input_kernel_size[i],
                input_stride = self.input_stride[i],
                input_padding = self.input_padding[i])
        
            setattr(self, name, cell)
            self._all_layers.append(cell)  

        # output layer
        self.output_layer = nn.Conv2d(2, 8, kernel_size = 5, stride = 1, 
                                      padding=2, padding_mode='circular')

        # pixelshuffle - upscale
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)   

        # initialize weights
        self.apply(initialize_weights)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, grid_step, wavelength, initial_state=None, begining: bool = True):
        bs, frames, H, W = x.shape
        if begining:
            ##
            self.initial_state = torch.randn(bs, 2, 128, 32, 32)## generate a random initial state
        else:
            self.initial_state = initial_state
        outputs = torch.zeros(bs, frames-9, H, W).to(x.device)
        srcs = x[:, -self.out_frames:, ...]
        E0 = x[:, 1:-self.out_frames, ...]
        eps = x[:, 0:1, ...]
        internal_state = []
        second_last_state = []

        x0 = E0.clone()
        for step in range(self.step):
            if step != self.step - 1:
                srct = srcs[:, step*8: (step+1)*8, ...] # bs, 8, H, W
            else:
                srct = srcs[:, step*8:, ...] # bs, 4, H, W
                zero_padding = torch.zeros(bs, 4, H, W).to(x.device)
                srct = torch.cat([srct, zero_padding], dim=1)
            x0 = torch.cat([x0, eps, srct], dim=1) # bs, 10, H, W
            # encoder
            for i in range(self.num_encoder):
                name = 'encoder{}'.format(i)
                x0 = getattr(self, name)(x0) # bs, 128, 32, 32
                
            # convlstm
            for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                name = 'convlstm{}'.format(i)
                if step == 0:
                    (h, c) = getattr(self, name).init_hidden_tensor(prev_state = self.initial_state)
                        # prev_state = self.initial_state[i - self.num_encoder])  
                    internal_state.append((h,c))
                
                # one-step forward
                (h, c) = internal_state[i - self.num_encoder]
                x0, new_c = getattr(self, name)(x0, h, c)
                internal_state[i - self.num_encoder] = (x0, new_c)                               

            # output
            x0 = self.pixelshuffle(x0)
            x0 = self.output_layer(x0)

            # residual connection
            if step == 0:
                x0 = E0+ self.dt * x0
            else:
                x0 = outputs[:, (step-1)*8:step*8, ...] + self.dt * x0

            if step == (self.step - 2):
                second_last_state = internal_state.copy()
                
            if step in self.effective_step and step != (self.step - 1):
                outputs[:, step*8: (step+1)*8, ...] = x0
            elif step == (self.step - 1):
                outputs[:, step*8: (step+1)*8-4, ...] = x0[:, 0:4, ...]

        return outputs, second_last_state