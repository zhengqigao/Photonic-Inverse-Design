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

def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Dict,
    mixup_fn: Callable = None,
    device: torch.device = torch.device("cuda:0"),
    plot: bool = False,
    grad_scaler=None,
) -> None:
    torch.autograd.set_detect_anomaly(True)
    model.train()
    step = epoch * len(train_loader)

    mse_meter = AverageMeter("mse")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    aux_output_weight = getattr(configs.criterion, "aux_output_weight", 0)
    accum_iter = getattr(configs.run, "grad_accum_step", 1)

    # poynting_loss = PoyntingLoss(configs.model.grid_step, wavelength=1.55)
    data_counter = 0
    total_data = len(train_loader.dataset)
    for batch_idx, (raw_data, raw_target) in enumerate(train_loader):
        #     break
        # for batch_idx, _ in enumerate(train_loader):
        for key, d in raw_data.items():
            raw_data[key] = d.to(device, non_blocking=True)
        for key, t in raw_target.items():
            raw_target[key] = t.to(device, non_blocking=True)

        data = torch.cat([raw_data["eps"], raw_data["Ez"], raw_data["source"]], dim=1)
        target = raw_target["Ez"]

        data_counter += data.shape[0]
        # print(data.shape)
        target = target.to(device, non_blocking=True)
        if mixup_fn is not None:
            data, target = mixup_fn(data, target)

        with amp.autocast(enabled=grad_scaler._enabled):
            output, normalization_factor = model(data, src_mask=raw_data["src_mask"], padding_mask=raw_data["padding_mask"])
            if type(output) == tuple:
                output, aux_output = output
            else:
                aux_output = None
            regression_loss = criterion(output, target/normalization_factor, raw_data["mseWeight"])
            mse_meter.update(regression_loss.item())
            loss = regression_loss
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                if name == "curl_loss":
                    fields = torch.cat([target[:, 0:1]], output, target[:, 2:3], dim=1)
                    aux_loss = weight * aux_criterion(fields, data[:, 0:1])
                elif name == "tv_loss":
                    aux_loss = weight * aux_criterion(output, target)
                elif name == "poynting_loss":
                    aux_loss = weight * aux_criterion(output, target)
                elif name == "rtv_loss":
                    aux_loss = weight * aux_criterion(output, target)
                loss = loss + aux_loss
                aux_meters[name].update(aux_loss.item())
            # TODO aux output loss
            if aux_output is not None and aux_output_weight > 0:
                aux_output_loss = aux_output_weight * F.mse_loss(
                    aux_output, target.abs()
                )  # field magnitude learning
                loss = loss + aux_output_loss
            else:
                aux_output_loss = None

            loss = loss / accum_iter

        grad_scaler.scale(loss).backward()

        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            grad_scaler.unscale_(optimizer)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} Regression Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                regression_loss.data.item(),
            )
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            if aux_output_loss is not None:
                log += f" aux_output_loss: {aux_output_loss.item()}"
            lg.info(log)

            mlflow.log_metrics({"train_loss": loss.item()}, step=step)
            wandb.log(
                {
                    "train_running_loss": loss.item(),
                    "global_step": step,
                },
            )

        # break
    scheduler.step()
    avg_regression_loss = mse_meter.avg
    lg.info(f"Train Regression Loss: {avg_regression_loss:.4e}")
    mlflow.log_metrics(
        {"train_regression": avg_regression_loss, "lr": get_learning_rate(optimizer)},
        step=epoch,
    )
    wandb.log(
        {
            "train_loss": avg_regression_loss,
            "epoch": epoch,
            "lr": get_learning_rate(optimizer),
        },
    )

    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_train.png")
        plot_compare(
            epsilon=data[0, 0:1],
            input_fields=data[0, 1 : -target.shape[1]],
            pred_fields=output[0]*normalization_factor[0],
            target_fields=target[0],
            filepath=filepath,
            pol="Ez",
            norm=False,
        )

def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = True,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    with torch.no_grad(), DeterministicCtx(42):
        for i, (raw_data, raw_target) in enumerate(validation_loader):
            for key, d in raw_data.items():
                raw_data[key] = d.to(device, non_blocking=True)
            for key, t in raw_target.items():
                raw_target[key] = t.to(device, non_blocking=True)

            # data = torch.cat([raw_data["eps"], raw_data["Ez"], raw_data["source"], raw_data["src_mask"]], dim=1)
            data = torch.cat([raw_data["eps"], raw_data["Ez"], raw_data["source"]], dim=1)
            target = raw_target["Ez"]
            target = target.clone()
            # print(data.shape)
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(data, target)

            with amp.autocast(enabled=False):
                output, normalization_factor = model(data, src_mask=raw_data["src_mask"], padding_mask=raw_data["padding_mask"])
                if type(output) == tuple:
                    output, aux_output = output
                else:
                    aux_output = None
                val_loss = criterion(output, target/normalization_factor, raw_data["mseWeight"])
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nValidation set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"val_loss": mse_meter.avg}, step=epoch)
    wandb.log(
        {
            "val_loss": mse_meter.avg,
            "epoch": epoch,
        },
    )

    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_valid.png")
        plot_compare(
            epsilon=data[1, 0:1] if data.shape[0] > 1 else data[0, 0:1],
            input_fields=data[1, 1 : -target.shape[1]] if data.shape[0] > 1 else data[0, 1 : -target.shape[1]],
            pred_fields=output[1]*normalization_factor[1] if data.shape[0] > 1 else output[0]*normalization_factor[0],
            target_fields=target[1] if data.shape[0] > 1 else target[0],
            filepath=filepath,
            pol="Ez",
            norm=False,
        )

def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    with torch.no_grad(), DeterministicCtx(42):
        for i, (raw_data, raw_target) in enumerate(test_loader):
            for key, d in raw_data.items():
                raw_data[key] = d.to(device, non_blocking=True)
            for key, t in raw_target.items():
                raw_target[key] = t.to(device, non_blocking=True)

            # data = torch.cat([raw_data["eps"], raw_data["Ez"], raw_data["source"], raw_data["src_mask"]], dim=1)
            data = torch.cat([raw_data["eps"], raw_data["Ez"], raw_data["source"]], dim=1)
            target = raw_target["Ez"]
            # print(data.shape)
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(data, target)

            with amp.autocast(enabled=False):
                output, normalization_factor = model(data, src_mask=raw_data["src_mask"], padding_mask=raw_data["padding_mask"])
                if type(output) == tuple:
                    output, aux_output = output
                else:
                    aux_output = None
                val_loss = criterion(output, target/normalization_factor, raw_data["mseWeight"])
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nTest set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"test_loss": mse_meter.avg}, step=epoch)
    wandb.log(
        {
            "test_loss": mse_meter.avg,
            "epoch": epoch,
        },
    )

    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_test.png")
        plot_compare(
            epsilon=data[0, 0:1],
            input_fields=data[0, 1 : -target.shape[1]],
            pred_fields=output[0]*normalization_factor[0],
            target_fields=target[0],
            filepath=filepath,
            pol="Ez",
            norm=False,
        )

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
