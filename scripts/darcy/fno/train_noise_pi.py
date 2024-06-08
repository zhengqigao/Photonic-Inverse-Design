"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-09-28 15:38:44
"""
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "darcy"
model = "fno"
exp_name = "train_noise"
root = f"log/{dataset}/{model}/{exp_name}_pinn"
script = "train_flow.py"
config_file = f"configs/{dataset}/{model}/{exp_name}/train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    (
        loss,
        loss_norm,
        aux_loss,
        lr,
        in_std,
        tar_std,
        in_scale,
        tar_scale,
        in_prec,
        tar_prec,
        epochs,
        id,
    ) = args
    with open(
        os.path.join(
            root,
            f"darcy-pinn_istd-{in_std}_tstd-{tar_std}_is-{in_scale}_ts-{tar_scale}_ip-{in_prec}_tp-{tar_prec}_aux-{aux_loss}_id-{id}.log",
        ),
        "w",
    ) as wfid:
        exp = [
            f"--plot.interval=50",
            f"--plot.dir_name={model}_{exp_name}_{loss}_{id}",
            f"--run.log_interval=50",
            f"--run.random_state={41+id}",
            f"--criterion.name={loss}",
            f"--criterion.norm={loss_norm}",
            f"--aux_criterion.{aux_loss}.weight=1",
            f"--optimizer.lr={lr}",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            f"--train_noise.input_gaussian.std={in_std}",
            f"--train_noise.target_gaussian.std={tar_std}",
            f"--train_noise.input_downsample.scale_factor={in_scale}",
            f"--train_noise.target_downsample.scale_factor={tar_scale}",
            f"--train_noise.input_quant.prec={in_prec}",
            f"--train_noise.target_quant.prec={tar_prec}",
            f"--run.n_epochs={epochs}",
            f"--checkpoint.model_comment=istd-{in_std}_tstd-{tar_std}_is-{in_scale}_ts-{tar_scale}_ip-{in_prec}_tp-{tar_prec}",
            f"--dataset.encode_input=False",
            f"--dataset.encode_output=False",
            f"--dataset.positional_encoding=False",
            f"--dataset.in_channels=1",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    # scan gaussian noise
    tasks1 = [
        ["mae", True, "darcy_loss", 0.005, 0.0, 0.0, 1, 1, "fp32", "fp32", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0.03, 0.03, 1, 1, "fp32", "fp32", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0.05, 0.05, 1, 1, "fp32", "fp32", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0.07, 0.07, 1, 1, "fp32", "fp32", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0.09, 0.09, 1, 1, "fp32", "fp32", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0.11, 0.11, 1, 1, "fp32", "fp32", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0.15, 0.15, 1, 1, "fp32", "fp32", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0.2, 0.2, 1, 1, "fp32", "fp32", 50, 1],
    ][::-1]

    # scam downsampling factor
    tasks2 = [
        ["mae", True, "darcy_loss", 0.005, 0, 0, 0.8, 0.8, "fp32", "fp32", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0, 0, 0.7, 0.7, "fp32", "fp32", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0, 0, 0.6, 0.6, "fp32", "fp32", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0, 0, 0.5, 0.5, "fp32", "fp32", 50, 1],
    ][::-1]

    # scan quantization precision
    tasks3 = [
        ["mae", True, "darcy_loss", 0.005, 0, 0, 1, 1, "fp16", "fp16", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0, 0, 1, 1, "bfp16", "bfp16", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0, 0, 1, 1, "int16", "int16", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0, 0, 1, 1, "int8", "int8", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0, 0, 1, 1, "int7", "int7", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0, 0, 1, 1, "int6", "int6", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0, 0, 1, 1, "int5", "int5", 50, 1],
        ["mae", True, "darcy_loss", 0.005, 0, 0, 1, 1, "int4", "int4", 50, 1],
    ]

    with Pool(8) as p:
        p.map(task_launcher, tasks1 + tasks2 + tasks3)
    logger.info(f"Exp: {configs.run.experiment} Done.")
