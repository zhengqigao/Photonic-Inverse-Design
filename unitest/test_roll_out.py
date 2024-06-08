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
from core.utils import DeterministicCtx

def test_roll_out(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: Criterion,
    device: torch.device,
    cfg: dict,
) -> None:
    model.eval()
    total_loss = []
    with torch.no_grad(), DeterministicCtx(42):
        for i, (raw_data, raw_target) in enumerate(test_loader):
            for key, d in raw_data.items():
                raw_data[key] = d.to(device, non_blocking=True)
            for key, t in raw_target.items():
                raw_target[key] = t.to(device, non_blocking=True)

            src_list = raw_data["source"].chunk(cfg.dataset.out_frames//cfg.model.out_frames, dim=1)
            target_list = raw_target["Ez"].chunk(cfg.dataset.out_frames//cfg.model.out_frames, dim=1)
            data = torch.cat([raw_data["eps"], raw_data["Ez"][:, :10], raw_data["source"][:, :cfg.model.out_frames]], dim=1)
            num_iters = raw_target["Ez"].shape[1] // cfg.model.out_frames
            result = []
            loss_list = []
            with amp.autocast(enabled=False):
                for iter_time in range(num_iters):
                    output, normalization_factor = model(data, target_list[iter_time], grid_step=raw_data["grid_step"], print_info=False, src_mask=raw_data["src_mask"], padding_mask=raw_data["padding_mask"])
                    prediction = output*normalization_factor
                    result.append(prediction)
                    local_loss = criterion(output, target_list[iter_time]/normalization_factor, raw_data["padding_mask"])
                    loss_list.append(local_loss.item())
                    ## update the source and the input fields for the next iteration
                    if iter_time == 0:
                        input_fields = torch.cat([raw_data["Ez"][:, :10], prediction], dim=1)[:, -model.in_frames:]
                    else:
                        input_fields = torch.cat([input_fields, prediction], dim=1)[:, -model.in_frames:]
                    if iter_time == num_iters-1:
                        break # no need to update the data any more
                    else:
                        data = torch.cat([raw_data["eps"], input_fields, src_list[iter_time+1]], dim=1)
                avg_loss = (np.array(loss_list).mean())
                total_loss.append(avg_loss)
        total_avg_loss = np.array(total_loss).mean()
        print("Here we go! This is the total_avg_loss", total_avg_loss, flush=True)

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

    _, _, test_loader = builder.make_dataloader()

    test_criterion = builder.make_criterion(
        configs.test_criterion.name, configs.test_criterion
    ).to(device)


    load_model(
        model,
        configs.checkpoint.restore_checkpoint,
        ignore_size_mismatch=int(configs.checkpoint.no_linear),
    )
    print("model loaded successfully!", flush=True)
    test_roll_out(model, test_loader, test_criterion, device, configs)


if __name__ == "__main__":
    main()