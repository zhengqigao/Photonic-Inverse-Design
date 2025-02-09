#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Callable, Dict, Iterable, List
import torch.cuda.amp as amp
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter, logger as lg
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
from core.utils import plot_compare, DeterministicCtx
import wandb
import datetime

def train_phc(
    model,
    optimizer: Optimizer,
    lr_scheduler: Scheduler,
    temp_scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    lossv: List,
    device: torch.device = torch.device("cuda:0"),
    plot: bool = False,
    grad_scaler=None,
) -> None:
    # the model here is a photonic crystal device
    # first we will use the meep to calculate the transmission efficiency of the device
    # and then we use the adjoint method to calculate the gradient of the transmission efficiency w.r.t the permittivity
    # then we use the autograd to calculate parital permittivity over partial design variables
    # and then we use the optimizer to update the permittivity
    # the scheduler is used to update the learning rate and the temperature of the binarization
    # TODO finish the training frame of the photonic crystal device
    print("this is the epoch", epoch, flush=True)
    torch.autograd.set_detect_anomaly(True)
    model.train()
    step = epoch
    distance_meter = AverageMeter("distance")

    with amp.autocast(enabled=grad_scaler._enabled):
        temp = temp_scheduler.step()
        print("begin forward call ...")
        f0, grad, hole_position = model(temp)
        print("finish the obj and grad calculation", flush=True)
        if type(output) == tuple:
            output, aux_output = output
        else:
            aux_output = None
        regression_loss = criterion(hole_position)
        distance_meter.update(regression_loss.item())
        loss = regression_loss

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    model.backward(grad)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    step += 1

    log = "Train Epoch: {} | Loss: {:.4e} Regression Loss: {:.4e}".format(
        epoch,
        loss.data.item(),
        regression_loss.data.item(),
    )
    log += f" optimization objective: {f0}"
    lg.info(log)

    mlflow.log_metrics({"train_loss": loss.item()}, step=step)
    wandb.log(
        {
            "train_running_loss": loss.item(),
            "global_step": step,
        },
    )

        # break
    lr_scheduler.step()
    avg_regression_loss = distance_meter.avg
    lg.info(f"Train Regression Loss: {avg_regression_loss:.4e}")
    wandb.log(
        {
            "train_loss": avg_regression_loss,
            "opt_objective": float(f0),
            "epoch": epoch,
            "lr": get_learning_rate(optimizer),
        },
    )
    lossv.append(regression_loss.item() - float(f0))
    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_train.png")
        # plot the permittivity
    return None

def main() -> None:
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

    if int(configs.run.deterministic) == True:
        set_torch_deterministic(int(configs.run.random_state))

    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )
    lg.info(model)

    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
    optimizer = builder.make_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    lr_scheduler = builder.make_scheduler(optimizer, config_file=configs.lr_scheduler)
    temp_scheduler = builder.make_scheduler(optimizer, name="temperature", config_file=configs.temp_scheduler)
    
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=10,
        metric_name="err",
        format="{:.4f}",
    )

    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    wandb.login()
    tag = wandb.util.generate_id()
    group = f"{datetime.date.today()}"
    name = f"{configs.run.wandb.name}-{datetime.datetime.now().hour:02d}{datetime.datetime.now().minute:02d}{datetime.datetime.now().second:02d}-{tag}"
    configs.run.pid = os.getpid()
    run = wandb.init(
        project=configs.run.wandb.project,
        # entity=configs.run.wandb.entity,
        group=group,
        name=name,
        id=tag,
        # Track hyperparameters and run metadata
        config=configs,
    )

    lossv = [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {name} starts. Group: {group}, Run ID: ({run.id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        if (
            int(configs.checkpoint.resume)
            and len(configs.checkpoint.restore_checkpoint) > 0
        ):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )

        for epoch in range(1, int(configs.run.n_epochs) + 1):
            train_phc(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                temp_scheduler=temp_scheduler,
                epoch=epoch,
                criterion=criterion,
                lossv=lossv,
                device=device,
                plot=configs.plot.train,
                grad_scaler=grad_scaler,
            )
            if epoch > int(configs.run.n_epochs) - 21:
                saver.save_model(
                    model,
                    lossv[-1],
                    epoch=epoch,
                    path=checkpoint,
                    save_model=False,
                    print_msg=True,
                )
        wandb.finish()
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")

if __name__ == "__main__":
    main()
