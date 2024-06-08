import argparse
import os
from typing import Callable, Dict, Iterable, List, Tuple
import torch.cuda.amp as amp
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
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
from core.utils import CurlLoss, PoyntingLoss, plot_compare


def extract_gradients(model: nn.Module):
    grads = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.is_complex():
                grad = torch.view_as_real(p.grad)
            else:
                grad = p.grad
            grads[name] = grad.data.clone().flatten()
    total_grad = torch.cat(list(grads.values()), dim=0)
    return grads, total_grad


def cosine_distance(
    grads_1: Dict[str, Tensor], grads_2: Dict[str, Tensor]
) -> Dict[str, float]:
    dist = np.zeros([len(grads_1)])
    for idx, name in enumerate(grads_1):
        dist[idx] = (
            (1 - F.cosine_similarity(grads_1[name], grads_2[name], dim=0)).cpu().item()
        )
    return dist


def l2_distance(
    grads_1: Dict[str, Tensor],
    total_grad_1: Tensor,
    grads_2: Dict[str, Tensor],
    total_grad_2: Tensor,
) -> Dict[str, float]:
    dist = np.zeros([len(grads_1)])
    for idx, name in enumerate(grads_1):
        dist[idx] = (
            (
                grads_1[name]
                .sub(grads_2[name])
                .square_()
                .sum()
                .div(grads_1[name].norm(2).square())
            )
            .cpu()
            .item()
        )
    total_dist = (
        (
            total_grad_1.sub(total_grad_2)
            .square_()
            .sum()
            .div(total_grad_1.norm(2).square())
        )
        .cpu()
        .item()
    )

    return dist, total_dist


def angular_similarity(
    grads_1: Dict[str, Tensor],
    total_grad_1: Tensor,
    grads_2: Dict[str, Tensor],
    total_grad_2: Tensor,
) -> Dict[str, float]:
    dist = np.zeros([len(grads_1)])
    for idx, name in enumerate(grads_1):
        dist[idx] = (
            (
                1
                - F.cosine_similarity(grads_1[name], grads_2[name], dim=0).clamp_(-1, 1)
                .acos_()
                .div_(np.pi)
            )
            .cpu()
            .item()
        )
    total_dist = (
        (1 - F.cosine_similarity(total_grad_1, total_grad_2, dim=0).clamp_(-1, 1).acos_().div_(np.pi))
        .cpu()
        .item()
    )
    return dist, total_dist


def train(
    model: nn.Module,
    train_loader: DataLoader,
    ori_train_loader: DataLoader,
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
    model.train()
    step = epoch * len(train_loader)

    accum_arg_dist = []
    accum_l2_dist = []
    accum_total_arg_dist = []
    accum_total_l2_dist = []

    mse_meter = AverageMeter("mse")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    aux_output_weight = getattr(configs.criterion, "aux_output_weight", 0)

    # poynting_loss = PoyntingLoss(configs.model.grid_step, wavelength=1.55)
    data_counter = 0
    ori_data_counter = 0
    total_data = len(train_loader.dataset)
    for batch_idx, (ori_batch, batch) in enumerate(zip(ori_train_loader, train_loader)):
        ori_data, ori_target = ori_batch
        data, target = batch
        #     break
        # for batch_idx, _ in enumerate(train_loader):

        ori_data = ori_data.to(device, non_blocking=True)
        ori_data_counter += ori_data.shape[0]
        ori_target = ori_target.to(device, non_blocking=True)
 
        data = data.to(device, non_blocking=True)
        data_counter += data.shape[0]
        target = target.to(device, non_blocking=True)

        with amp.autocast(enabled=grad_scaler._enabled):
            output = model(ori_data)
            if type(output) == tuple:
                output, aux_output = output
            else:
                aux_output = None
            regression_loss = criterion(output, ori_target)
            mse_meter.update(regression_loss.item())
            loss = regression_loss
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                if name == "tv_loss":
                    aux_loss = weight * aux_criterion(output, ori_target)

                loss = loss + aux_loss
                aux_meters[name].update(aux_loss.item())
            # TODO aux output loss
            if aux_output is not None and aux_output_weight > 0:
                aux_output_loss = aux_output_weight * F.mse_loss(
                    aux_output, ori_target.abs()
                )  # field magnitude learning
                loss = loss + aux_output_loss
            else:
                aux_output_loss = None

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)

        grads_ori, total_grad_ori = extract_gradients(model)

        # extract noisy gradient
        with amp.autocast(enabled=grad_scaler._enabled):
            output = model(data)
            if type(output) == tuple:
                output, aux_output = output
            else:
                aux_output = None
            regression_loss = criterion(output, target)
            mse_meter.update(regression_loss.item())
            loss = regression_loss
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                if name == "tv_loss":
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

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)

        grads, total_grad = extract_gradients(model)

        # calculate distance
        arg_dist, total_arg_dist = angular_similarity(
            grads_ori, total_grad_ori, grads, total_grad
        )
        l2_dist, total_l2_dist = l2_distance(grads_ori, total_grad_ori, grads, total_grad)

        accum_arg_dist.append(arg_dist)
        accum_l2_dist.append(l2_dist)
        accum_total_arg_dist.append(total_arg_dist)
        accum_total_l2_dist.append(total_l2_dist)



        grad_scaler.step(optimizer)
        grad_scaler.update()

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
        # break
    scheduler.step()



    accum_arg_dist = np.stack(accum_arg_dist, 0)
    accum_l2_dist = np.stack(accum_l2_dist, 0)
    accum_total_arg_dist = np.stack(accum_total_arg_dist, 0)
    accum_total_l2_dist = np.stack(accum_total_l2_dist, 0)
    layer_avg_arg_dist, layer_std_arg_dist = np.mean(accum_arg_dist, 0), np.std(accum_arg_dist, 0)
    layer_avg_l2_dist, layer_std_l2_dist = np.mean(accum_l2_dist, 0), np.std(accum_l2_dist, 0)
    total_avg_arg_dist, total_std_arg_dist = np.mean(accum_total_arg_dist), np.std(accum_total_arg_dist)
    total_avg_l2_dist, total_std_l2_dist = np.mean(accum_total_l2_dist), np.std(accum_total_l2_dist)
    mlflow.log_metrics(
        {
            "total_avg_arg": total_avg_arg_dist,
            "total_std_arg": total_std_arg_dist,
            "total_avg_l2": total_avg_l2_dist,
            "total_std_l2": total_std_l2_dist,
        },
        step=epoch,
    )

    for i in range(len(layer_avg_arg_dist)):
        mlflow.log_metrics(
            {
                f"l{i}_avg_arg": layer_avg_arg_dist[i],
                f"l{i}_std_arg": layer_std_arg_dist[i],
                f"l{i}_avg_l2": layer_avg_l2_dist[i],
                f"l{i}_std_l2": layer_std_l2_dist[i],
            },
            step=epoch,
        )



    avg_regression_loss = mse_meter.avg
    lg.info(f"Train Regression Loss: {avg_regression_loss:.4e}")
    mlflow.log_metrics(
        {"train_regression": avg_regression_loss, "lr": get_learning_rate(optimizer)},
        step=epoch,
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
    with torch.no_grad():
        for i, (data, target) in enumerate(validation_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(data, target, random_state=i, vflip=False)

            output = model(data)

            val_loss = criterion(output, target)
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nValidation set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"val_loss": mse_meter.avg}, step=epoch)


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
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(
                    data, target, random_state=i + 10000, vflip=False
                )
            output = model(data)
            # print(output.shape)

            val_loss = criterion(output, target)
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nTest set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"test_loss": mse_meter.avg}, step=epoch)


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

    train_loader, validation_loader, test_loader = builder.make_dataloader()
    ori_train_loader, _, _ = builder.make_dataloader(train_noise_cfg={})
    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )

    optimizer = builder.make_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
    if configs.aux_criterion is not None:
        aux_criterions = {
            name: [builder.make_criterion(name, cfg=config), float(config.weight)]
            for name, config in configs.aux_criterion.items()
            if float(config.weight) > 0
        }
    else:
        aux_criterions = {}
    print(aux_criterions)
    mixup_config = configs.dataset.augment
    if mixup_config is not None:
        mixup_fn = MixupAll(**mixup_config)
        test_mixup_fn = MixupAll(**configs.dataset.test_augment)
    else:
        mixup_fn = test_mixup_fn = None
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=4,
        metric_name="err",
        format="{:.4f}",
    )

    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(
        {
            "exp_name": configs.run.experiment,
            "exp_id": experiment.experiment_id,
            "run_id": mlflow.active_run().info.run_id,
            "init_lr": configs.optimizer.lr,
            "checkpoint": checkpoint,
            "restore_checkpoint": configs.checkpoint.restore_checkpoint,
            "pid": os.getpid(),
        }
    )

    lossv, accv = [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
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

            lg.info("Validate resumed model...")
            test(model, validation_loader, 0, criterion, lossv, accv, False, device)

        for epoch in range(1, int(configs.run.n_epochs) + 1):
            train(
                model,
                train_loader,
                ori_train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                aux_criterions,
                mixup_fn,
                device,
                plot=configs.plot.train,
                grad_scaler=grad_scaler,
            )

            if validation_loader is not None:
                validate(
                    model,
                    validation_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                    plot=configs.plot.valid,
                )
            if epoch > int(configs.run.n_epochs) - 21:
                test(
                    model,
                    test_loader,
                    epoch,
                    criterion,
                    [],
                    [],
                    device,
                    mixup_fn=test_mixup_fn,
                    plot=configs.plot.test,
                )
                saver.save_model(
                    model,
                    lossv[-1],
                    epoch=epoch,
                    path=checkpoint,
                    save_model=False,
                    print_msg=True,
                )
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
