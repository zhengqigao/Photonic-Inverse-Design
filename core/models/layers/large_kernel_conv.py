"""
Date: 2024-04-24 21:30:00
LastEditors: Pingchuan Ma && pingchua@asu.edu
LastEditTime: 2024-04-24 21:30:00
FilePath: /NeurOLight_Local/core/models/layers/large_kernel_conv.py
"""

from functools import lru_cache
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import einsum
from mmengine.registry import MODELS
from pyutils.general import logger
from torch import nn
from torch.functional import Tensor
from torch.nn.modules.utils import _ntuple
from torch.types import Device

__all__ = ["LargeKernelConv2d"]

## the large kernel convolution is inherited from nn.Conv2d
## the only different is that the large kernel convolution will have to override the load_state_dict method
## so that it could load weight of different shape
@MODELS.register_module()
class LargeKernelConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + 'weight'
        if weight_key in state_dict:
            current_shape = state_dict[weight_key].shape
            expected_shape = (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
            # Check if current weights need to be padded to match expected dimensions
            if current_shape != expected_shape:
                # Calculate padding to reach expected kernel size
                padding = [(e - c) // 2 for e, c in zip(expected_shape[2:], current_shape[2:])]
                # Assume we need to modify the weight dimensions or values
                weight = state_dict[weight_key]
                # Perform your custom transformations here
                transformed_weight = F.pad(weight, (padding[1], padding[1], padding[0], padding[0]), "constant", 0)
                state_dict[weight_key] = transformed_weight

        # Call the base implementation to load the transformed state
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
