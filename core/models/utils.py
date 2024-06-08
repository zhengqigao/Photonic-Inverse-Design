'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-20 16:55:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-13 15:36:39
'''
import math
from functools import lru_cache
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import einsum
from pyutils.activation import Swish
from pyutils.torch_train import set_torch_deterministic
from timm.models.layers import DropPath
from torch import nn
from torch.functional import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.types import Device

from core.utils import plot_compare, print_stat

from .constant import *
from .layers import FourierConv3d
from .layers.activation import SIREN
from .pde_base import PDE_NN_BASE


def conv_output_size(in_size, kernel_size, padding=0, stride=1, dilation=1):
    return math.floor((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def get_last_n_frames(packed_sequence, n=100):
    # Unpack the sequence
    padded_sequences, lengths = pad_packed_sequence(packed_sequence, batch_first=True)
    last_frames = []

    for i, length in enumerate(lengths):
        # Calculate the start index for slicing
        start = max(length - n, 0)
        # Extract up to the last n frames
        last_n = padded_sequences[i, start:length, :]
        last_frames.append(last_n)
    last_frames = torch.stack(last_frames, dim=0)
    return last_frames

