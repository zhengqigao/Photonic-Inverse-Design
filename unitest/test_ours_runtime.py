import argparse
import os
from typing import Callable, Dict, Iterable, List, Tuple
import torch.cuda.amp as amp
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter, logger as lg
from pyutils.loss import KLLossMixed
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler
import torch.fft
from core import builder
from core.datasets.mixup import Mixup, MixupAll
from core.utils import CurlLoss, PoyntingLoss, plot_compare, print_stat, DeterministicCtx, normalize, make_axes_locatable
import wandb
import datetime
import h5py
import yaml
from torch import Tensor
from itertools import islice
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from pyutils.general import TimerCtx, print_stat, TorchTracemalloc

def test_speed(model, cfg, device, img_size):
    sf = 0.75*(256/534.75)
    h = round(img_size[0]*sf)
    w = round(img_size[1]*sf)
    h = 168
    w = 168
    model.eval()
    # in_channels = model.in_channels
    in_channels = 171
    dummy_data = torch.randn(1, in_channels, h, w).to(device)
    dummy_src_mask = torch.ones(1, 1, h, w).to(device)
    dummy_padding_mask = torch.ones(1, 1, h, w).to(device)
    with torch.no_grad():
        for _ in range(3):
            output, normalization_factor = model(dummy_data, src_mask=dummy_src_mask, padding_mask=dummy_padding_mask)
    torch.cuda.synchronize()
    with torch.no_grad():
        with TimerCtx() as t:
            for _ in range(3):
                output, normalization_factor = model(dummy_data, src_mask=dummy_src_mask, padding_mask=dummy_padding_mask)
            torch.cuda.synchronize()
    tt = t.interval / 3
    print("test time: ", tt)
    torch.cuda.empty_cache()
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if "backbone_cfg" in configs.model.keys():
        if configs.model.backbone_cfg.conv_cfg.type == "Conv2d" or configs.model.backbone_cfg.conv_cfg.type == "LargeKernelConv2d":
            if "r" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["r"]
            if "is_causal" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["is_causal"]
            if "mask_shape" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["mask_shape"]
            if "enable_padding" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["enable_padding"]

    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )
    lg.info(model)

    test_speed(model, configs, device, (256, 256))
    quit()
    device_id = range(0, 12)
    file_names = []
    for i in device_id:
        if i <= 9:
            file_name = f"mmi_1x1_L_determine_20-000{i}-p0.h5"
            file_names.append(file_name)
        else:
            file_name = f"mmi_1x1_L_determine_20-00{i}-p0.h5"
            file_names.append(file_name)
    for file_name in file_names:
        print("this is the file name: ", file_name)
        with h5py.File("./data/fdtd/raw_diff_res/"+file_name, "r") as f:
            Ez = torch.from_numpy(
                f["Ez"][:][()]
            ).float()
            img_size = Ez.shape[-2], Ez.shape[-1]
        test_speed(model, configs, device, img_size)

if __name__ == "__main__":
    main()