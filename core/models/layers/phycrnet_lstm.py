"""
Description:
Author: Pingchuan Ma (pingchua@asu.edu)
Date: 2024-02-06 20:46:39
LastEditors: Pingchuan Ma (pingchua@asu.edu)
LastEditTime: 2024-02-06 20:46:39
"""

'''PhyCRNet for solving spatiotemporal PDEs'''

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

__all__ = ["ConvLSTMCell", "encoder_block"]

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

# specific parameters for burgers equation
def initialize_weights(module):

    if isinstance(module, nn.Conv2d):
        #nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        c = 1 #0.5
        module.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), 
            c*np.sqrt(1 / (3 * 3 * 320)))
     
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
            bias=True, padding_mode='circular')

        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='circular')

        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True, padding_mode='circular')

        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True, padding_mode='circular')

        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=1, bias=False, 
            padding_mode='circular')       

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
        return (Variable(prev_state[0]).cuda(), Variable(prev_state[1]).cuda())


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