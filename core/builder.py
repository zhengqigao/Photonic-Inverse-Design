from typing import Dict, Tuple

import torch
import torch.nn as nn
from neuralop.models import FNO
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device
from torch.utils.data import Sampler
import random

from core.models import *

from .utils import (
    DAdaptAdam,
    NormalizedMSELoss,
    maskedNMSELoss,
    NL2NormLoss,
    maskedNL2NormLoss,
    TemperatureScheduler,
    DistanceLoss,
)

__all__ = [
    "make_dataloader",
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def collate_fn_keep_spatial_res(batch):
    data, targets = zip(*batch)
    new_size = []
    for item in data:
        if item["device_type"].item() == 1 or item["device_type"].item() == 5: # which means it is a mmi and resolution = 20
            newSize = (int(round(item["Ez"].shape[-2]*item["scaling_factor"].item()*0.75)),
                    int(round(item["Ez"].shape[-1]*item["scaling_factor"].item()*0.75)),)
        else:
            newSize = (int(round(item["Ez"].shape[-2]*item["scaling_factor"].item())),
                    int(round(item["Ez"].shape[-1]*item["scaling_factor"].item())),)
        new_size.append(newSize)
    Hight = [int(round(item["Ez"].shape[-2]*item["scaling_factor"].item())) for item in data]
    Width = [int(round(item["Ez"].shape[-1]*item["scaling_factor"].item())) for item in data]
    maxHight = max(Hight)
    maxWidth = max(Width)
    if maxWidth % 2 == 1:
        maxWidth += 1 ## make sure the width is even so that there won't be any mismatch between fourier and inverse fourier
    # Pad all items to the max length and max width using zero padding
    # eps should use background value to padding
    # fields should use zero padding
    # weight masks should use zero padding
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.interpolate(item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0) # dummy batch dim and then remove it
        item["eps"] = torch.nn.functional.interpolate(item["eps"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["source"] = torch.nn.functional.interpolate(item["source"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["mseWeight"] = torch.nn.functional.interpolate(item["mseWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["src_mask"] = torch.nn.functional.interpolate(item["src_mask"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["epsWeight"] = torch.nn.functional.interpolate(item["epsWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["padding_mask"] = torch.ones_like(item["eps"], device=item["eps"].device)
    
    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.interpolate(item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)

    hightPatchSize_bot = [(maxHight-item["Ez"].shape[-2])//2 for item in data]
    hightPatchSize_top = [maxHight-item["Ez"].shape[-2]-(maxHight-item["Ez"].shape[-2])//2 for item in data]
    widthPatchSize_left = [(maxWidth-item["Ez"].shape[-1])//2 for item in data]
    widthPatchSize_right = [maxWidth-item["Ez"].shape[-1]-(maxWidth-item["Ez"].shape[-1])//2 for item in data]
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.pad(item["Ez"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["eps"] = torch.nn.functional.pad(item["eps"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=item["eps_bg"].item())
        item["padding_mask"] = torch.nn.functional.pad(item["padding_mask"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["source"] = torch.nn.functional.pad(item["source"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["mseWeight"] = torch.nn.functional.pad(item["mseWeight"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["src_mask"] = torch.nn.functional.pad(item["src_mask"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["epsWeight"] = torch.nn.functional.pad(item["epsWeight"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        
    
    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.pad(item["Ez"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)

    Ez_data = torch.stack([item["Ez"] for item in data], dim=0)
    source_data = torch.stack([item["source"] for item in data], dim=0)
    eps_data = torch.stack([item["eps"] for item in data], dim=0)
    padding_mask_data = torch.stack([item["padding_mask"] for item in data], dim=0)
    mseWeight_data = torch.stack([item["mseWeight"] for item in data], dim=0)
    src_mask_data = torch.stack([item["src_mask"] for item in data], dim=0)
    epsWeight_data = torch.stack([item["epsWeight"] for item in data], dim=0)
    device_type_data = torch.stack([item["device_type"] for item in data], dim=0)
    eps_bg_data = torch.stack([item["eps_bg"] for item in data], dim=0)
    grid_step_data = torch.stack([item["grid_step"] for item in data], dim=0)

    Ez_target = torch.stack([item["Ez"] for item in targets], dim=0)
    std_target = torch.stack([item["std"] for item in targets], dim=0)
    mean_target = torch.stack([item["mean"] for item in targets], dim=0)
    stdDacayRate = torch.stack([item["stdDacayRate"] for item in targets], dim=0)

    raw_data = {
        "Ez": Ez_data,
        "source": source_data,
        "eps": eps_data,
        "padding_mask": padding_mask_data,
        "mseWeight": mseWeight_data,
        "src_mask": src_mask_data,
        "epsWeight": epsWeight_data,
        "device_type": device_type_data,
        "eps_bg": eps_bg_data,
        "grid_step": grid_step_data,
    }
    raw_targets = {
        "Ez": Ez_target,
        "std": std_target,
        "mean": mean_target,
        "stdDacayRate": stdDacayRate,
    }

    return raw_data, raw_targets

def collate_fn_keep_spatial_res_pad_to_256(batch):
    # Extract all items for each key and compute the max length
    data, targets = zip(*batch)
    new_size = []
    for item in data:
        if item["device_type"].item() == 1 or item["device_type"].item == 5: # train seperately for metaline, so no need to consider the resolution mismatch
            newSize = (int(round(item["Ez"].shape[-2]*item["scaling_factor"].item()*0.75)),
                    int(round(item["Ez"].shape[-1]*item["scaling_factor"].item()*0.75)),)
        else:
            newSize = (int(round(item["Ez"].shape[-2]*item["scaling_factor"].item())),
                    int(round(item["Ez"].shape[-1]*item["scaling_factor"].item())),)
        new_size.append(newSize)
    maxHight = 256
    maxWidth = 256
    if item["device_type"].item() == 5: # which means it is a metaline
        maxHight = 168
        maxWidth = 168
    # Pad all items to the max length and max width using zero padding
    # eps should use background value to padding
    # fields should use zero padding
    # weight masks should use zero padding
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.interpolate(item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0) # dummy batch dim and then remove it
        item["eps"] = torch.nn.functional.interpolate(item["eps"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["source"] = torch.nn.functional.interpolate(item["source"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["mseWeight"] = torch.nn.functional.interpolate(item["mseWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["src_mask"] = torch.nn.functional.interpolate(item["src_mask"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["epsWeight"] = torch.nn.functional.interpolate(item["epsWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["padding_mask"] = torch.ones_like(item["eps"], device=item["eps"].device)
    
    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.interpolate(item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)

    hightPatchSize_bot = [(maxHight-item["Ez"].shape[-2])//2 for item in data]
    hightPatchSize_top = [maxHight-item["Ez"].shape[-2]-(maxHight-item["Ez"].shape[-2])//2 for item in data]
    widthPatchSize_left = [(maxWidth-item["Ez"].shape[-1])//2 for item in data]
    widthPatchSize_right = [maxWidth-item["Ez"].shape[-1]-(maxWidth-item["Ez"].shape[-1])//2 for item in data]
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.pad(item["Ez"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["eps"] = torch.nn.functional.pad(item["eps"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=item["eps_bg"].item())
        item["padding_mask"] = torch.nn.functional.pad(item["padding_mask"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["source"] = torch.nn.functional.pad(item["source"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["mseWeight"] = torch.nn.functional.pad(item["mseWeight"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["src_mask"] = torch.nn.functional.pad(item["src_mask"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["epsWeight"] = torch.nn.functional.pad(item["epsWeight"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        
    
    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.pad(item["Ez"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)

    Ez_data = torch.stack([item["Ez"] for item in data], dim=0)
    source_data = torch.stack([item["source"] for item in data], dim=0)
    eps_data = torch.stack([item["eps"] for item in data], dim=0)
    padding_mask_data = torch.stack([item["padding_mask"] for item in data], dim=0)
    mseWeight_data = torch.stack([item["mseWeight"] for item in data], dim=0)
    src_mask_data = torch.stack([item["src_mask"] for item in data], dim=0)
    epsWeight_data = torch.stack([item["epsWeight"] for item in data], dim=0)
    device_type_data = torch.stack([item["device_type"] for item in data], dim=0)
    eps_bg_data = torch.stack([item["eps_bg"] for item in data], dim=0)
    grid_step_data = torch.stack([item["grid_step"] for item in data], dim=0)

    Ez_target = torch.stack([item["Ez"] for item in targets], dim=0)
    std_target = torch.stack([item["std"] for item in targets], dim=0)
    mean_target = torch.stack([item["mean"] for item in targets], dim=0)
    stdDacayRate = torch.stack([item["stdDacayRate"] for item in targets], dim=0)

    raw_data = {
        "Ez": Ez_data,
        "source": source_data,
        "eps": eps_data,
        "padding_mask": padding_mask_data,
        "mseWeight": mseWeight_data,
        "src_mask": src_mask_data,
        "epsWeight": epsWeight_data,
        "device_type": device_type_data,
        "eps_bg": eps_bg_data,
        "grid_step": grid_step_data,
    }
    raw_targets = {
        "Ez": Ez_target,
        "std": std_target,
        "mean": mean_target,
        "stdDacayRate": stdDacayRate,
    }

    return raw_data, raw_targets

class MySampler(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.indices = sorted(range(len(data_source)), key=lambda x: data_source[x][0]["area"].item())
        self.shuffle = shuffle
        # self.indices is a list of indices sorted by the area of the devices

    def __iter__(self):
        if self.shuffle:
            group_size = 2500
            group_num = len(self.indices) // group_size
            for i in range(group_num+1):
                group_indices = self.indices[i*group_size:(i+1)*group_size] if i != group_num else self.indices[i*group_size:]
                random.shuffle(group_indices)
                if i != group_num:
                    self.indices[i*group_size:(i+1)*group_size] = group_indices
                else:
                    self.indices[i*group_size:] = group_indices
        else:
            pass
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def make_dataloader(
    name: str = None,
    splits=["train", "valid", "test"],
    train_noise_cfg=None,
    out_frames=None,
) -> Tuple[DataLoader, DataLoader]:
    name = (name or configs.dataset.name).lower()
    if name == "fdtd":
        pass
    else:
        train_dataset, test_dataset = get_dataset(
            name,
            configs.dataset.img_height,
            configs.dataset.img_width,
            dataset_dir=configs.dataset.root,
            transform=configs.dataset.transform,
        )
        validation_dataset = None

    if train_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res":
        train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res,
                            sampler = MySampler(train_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif train_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res_pad_to_256":
        train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res_pad_to_256,
                            sampler = MySampler(train_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif train_dataset is not None and configs.dataset.batch_strategy == "resize_and_padding_to_square":
        train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset,
                            batch_size=configs.run.batch_size,
                            shuffle=int(configs.dataset.shuffle),
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                        )
    else:
        train_loader = None

    if validation_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res":
        validation_loader = torch.utils.data.DataLoader(
                            dataset=validation_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res,
                            sampler = MySampler(validation_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif validation_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res_pad_to_256":
        validation_loader = torch.utils.data.DataLoader(
                            dataset=validation_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res_pad_to_256,
                            sampler = MySampler(validation_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif validation_dataset is not None and configs.dataset.batch_strategy == "resize_and_padding_to_square":
        validation_loader = torch.utils.data.DataLoader(
                            dataset=validation_dataset,
                            batch_size=configs.run.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                        )
    else:
        validation_loader = None

    if test_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res":
        test_loader = torch.utils.data.DataLoader(
                            dataset=test_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res,
                            sampler = MySampler(test_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif test_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res_pad_to_256":
        test_loader = torch.utils.data.DataLoader(
                            dataset=test_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res_pad_to_256,
                            sampler = MySampler(test_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif test_dataset is not None and configs.dataset.batch_strategy == "resize_and_padding_to_square":
        test_loader = torch.utils.data.DataLoader(
                            dataset=test_dataset,
                            batch_size=configs.run.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                        )
    else:
        test_loader = None

    return train_loader, validation_loader, test_loader


def make_model(device: Device, random_state: int = None, **kwargs) -> nn.Module:
    if "repara_phc_1x1" in configs.model.name.lower():
        model = eval(configs.model.name)(
            device_cfg=configs.model.device_cfg, 
            sim_cfg=configs.model.sim_cfg, 
            purturbation=configs.model.purturbation,
            num_rows_perside=configs.model.num_rows_perside,
            num_cols=configs.model.num_cols,
        )
    else:
        raise NotImplementedError(f"Not supported model name: {configs.model.name}")
    return model


def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "dadaptadam":
        optimizer = DAdaptAdam(
            params,
            lr=configs.lr,
            betas=getattr(configs, "betas", (0.9, 0.999)),
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer, name: str = None, config_file: dict = {}) -> Scheduler:
    name = (name or config_file.name).lower()
    if name == "temperature": # this temperature scheduler is a cosine annealing scheduler
        scheduler = TemperatureScheduler(
            initial_T=float(configs.temp_scheduler.lr),
            final_T=float(configs.temp_scheduler.lr_min),
            total_steps=int(configs.run.n_epochs),
        )
    elif name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(configs.run.n_epochs),
            eta_min=float(configs.lr_scheduler.lr_min),
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=configs.scheduler.lr_gamma
        )
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    cfg = cfg or configs.criterion
    if name == "mse":
        criterion = nn.MSELoss()
    elif name == "nmse":
        criterion = NormalizedMSELoss()
    elif name == "nl2norm":
        criterion = NL2NormLoss()
    elif name == "masknl2norm":
        criterion = maskedNL2NormLoss(weighted_frames=cfg.weighted_frames, weight=cfg.weight, if_spatial_mask=cfg.if_spatial_mask)
    elif name == "masknmse":
        criterion = maskedNMSELoss(weighted_frames=cfg.weighted_frames, weight=cfg.weight, if_spatial_mask=cfg.if_spatial_mask)
    elif name == "distanceloss":
        criterion = DistanceLoss(min_distance=cfg.min_distance)
    else:
        raise NotImplementedError(name)
    return criterion
