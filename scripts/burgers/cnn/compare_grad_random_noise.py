'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-09-19 17:47:16
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "burgers"
model = "cnn"
exp_name = "train_noise"
root = f"log/{dataset}/{model}/{exp_name}_grad"
script = 'compare_gradient_flow.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    loss, loss_norm, lr, in_std, tar_std, in_scale, tar_scale, in_prec, tar_prec, epochs, id = args
    with open(os.path.join(root, f'burgers_istd-{in_std}_tstd-{tar_std}_is-{in_scale}_ts-{tar_scale}_ip-{in_prec}_tp-{tar_prec}_id-{id}.log'), 'w') as wfid:
        exp = [
            f"--plot.interval=20",
            f"--plot.dir_name={model}_{exp_name}_{loss}_{id}",
            f"--run.log_interval=20",
            f"--run.random_state={41+id}",
            f"--criterion.name={loss}",
            f"--criterion.norm={loss_norm}",
            f"--optimizer.lr={lr}",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            f"--train_noise.input_gaussian.std={in_std}",
            f"--train_noise.target_gaussian.std={tar_std}",
            f"--train_noise.input_downsample.scale_factor={in_scale}",
            f"--train_noise.target_downsample.scale_factor={tar_scale}",
            f"--train_noise.input_quant.prec={in_prec}",
            f"--train_noise.target_quant.prec={tar_prec}",
            f"--run.n_epochs={epochs}",
            f"--dataset.shuffle=False",
            f"--checkpoint.model_comment=istd-{in_std}_tstd-{tar_std}_is-{in_scale}_ts-{tar_scale}_ip-{in_prec}_tp-{tar_prec}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    # scan gaussian noise
    tasks1 = [
        ["mae", True, 0.005, 0.0,   0.0, 1,   1,   "fp32", "fp32", 20, 1],
        ["mae", True, 0.005, 0.03, 0.03, 1,   1,   "fp32", "fp32", 20, 1],
        ["mae", True, 0.005, 0.05, 0.05, 1,   1,   "fp32", "fp32", 20, 1],
        ["mae", True, 0.005, 0.07, 0.07, 1,   1,   "fp32", "fp32", 20, 1],
        ["mae", True, 0.005, 0.09, 0.09, 1,   1,   "fp32", "fp32", 20, 1],
        ["mae", True, 0.005, 0.11, 0.11, 1,   1,   "fp32", "fp32", 20, 1],
        ["mae", True, 0.005, 0.15, 0.15, 1,   1,   "fp32", "fp32", 20, 1],
        ["mae", True, 0.005, 0.2,   0.2, 1,   1,   "fp32", "fp32", 20, 1],
    ][::-1]

    # scam downsampling factor
    tasks2 = [
        ["mae", True, 0.005, 0, 0, 0.8,0.8,   "fp32", "fp32", 20, 1],
        ["mae", True, 0.005, 0, 0, 0.7,0.7,   "fp32", "fp32", 20, 1],
        ["mae", True, 0.005, 0, 0, 0.6,0.6,   "fp32", "fp32", 20, 1],
        ["mae", True, 0.005, 0, 0, 0.5,0.5,   "fp32", "fp32", 20, 1],
    ][::-1]


    # scan quantization precision
    tasks3 = [
        ["mae", True, 0.005, 0,    0,    1,   1,   "fp16", "fp16", 20, 1],
        ["mae", True, 0.005, 0,    0,    1,   1,   "bfp16","bfp16",20, 1],
        ["mae", True, 0.005, 0,    0,    1,   1,   "int16","int16",20, 1],
        ["mae", True, 0.005, 0,    0,    1,   1,   "int8", "int8", 20, 1],
        ["mae", True, 0.005, 0,    0,    1,   1,   "int7", "int7", 20, 1],
        ["mae", True, 0.005, 0,    0,    1,   1,   "int6", "int6", 20, 1],
        ["mae", True, 0.005, 0,    0,    1,   1,   "int5", "int5", 20, 1],
        ["mae", True, 0.005, 0,    0,    1,   1,   "int4", "int4", 20, 1],
    ]

    with Pool(8) as p:
        p.map(task_launcher, tasks1+tasks2+tasks3)
    logger.info(f"Exp: {configs.run.experiment} Done.")
