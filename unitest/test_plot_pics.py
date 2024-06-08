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
from core.utils import DeterministicCtx, normalize, make_axes_locatable, plot_compare, print_stat
import matplotlib.pyplot as plt
from torch import Tensor
import h5py
import yaml
import random

def plot_pics(test_loader, path = './plot/paper_figs'):
    rand_idx = len(test_loader.dataset) // test_loader.batch_size - 1
    rand_idx = [random.randint(0, rand_idx) for _ in range(4)]
    for batch_idx, (raw_data, raw_target) in enumerate(test_loader):

        data = torch.cat([raw_data["eps"], raw_data["Ez"], raw_data["source"]], dim=1)
        target = raw_target["Ez"]

        if batch_idx in rand_idx:
            # plot raw_data["eps"] to the path by matplotlib
            eps = raw_data["eps"].squeeze().cpu().numpy().transpose(-1, -2)
            print("this is the hight of the device: ", eps.shape[-1])
            print("this is the width of the device: ", eps.shape[-2])
            eps = eps > (eps.max()+eps.min()) / 2
            plt.imshow(eps, cmap='gray')
            plt.colorbar()
            plt.savefig(f"{path}/eps_{batch_idx}.png")
            plt.close()
            # plot raw_target["Ez"][:, -1] to the path by matplotlib
            target = raw_target["Ez"].squeeze(0)[0].cpu().numpy().transpose(-1, -2)
            plt.imshow(target.squeeze(), cmap = "RdBu_r")
            plt.colorbar()
            plt.savefig(f"{path}/target_{batch_idx}.png")
            plt.close()
            # plot raw_data["source"] to the path by matplotlib
            source = raw_data["source"].squeeze()[-1].cpu().numpy().transpose(-1, -2).squeeze()
            plt.imshow(source.squeeze(), cmap = "RdBu_r")
            plt.colorbar()
            plt.savefig(f"{path}/source_{batch_idx}.png")
            plt.close()

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

    _, _, test_loader = builder.make_dataloader()
    plot_pics(test_loader)



if __name__ == "__main__":
    main()