'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: JeremieMelo jqgu@utexas.edu
LastEditTime: 2023-09-16 12:18:06
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "mmi"
model = "fno"
exp_name = "train_random_noise"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    mixup, loss, loss_norm, aux_loss, aux_loss_w, aux_loss_norm, lr, enc, n_data, n_layer, in_std, tar_std, in_scale, tar_scale, in_prec, tar_prec, epochs, id = args
    data_list = [f"rHz_{i}" for i in range(n_data)]
    with open(os.path.join(root, f'rHz{n_data}_istd-{in_std}_tstd-{tar_std}_is-{in_scale}_ts-{tar_scale}_ip-{in_prec}_tp-{tar_prec}_id-{id}.log'), 'w') as wfid:
        exp = [
            f"--dataset.pol_list={str(data_list)}",
            f"--dataset.processed_dir=random_size{n_data}",
            f"--dataset.test_ratio=0.1",
            f"--plot.interval=50",
            f"--plot.dir_name={model}_{exp_name}_rHz{n_data}_mixup-{mixup}_{loss}_{enc}_nl-{n_layer}_tv-{aux_loss_w:.4f}_{id}",
            f"--run.log_interval=50",
            f"--run.random_state={41+id}",
            f"--dataset.augment.prob={mixup}",
            f"--criterion.name={loss}",
            f"--criterion.norm={loss_norm}",
            f"--aux_criterion.{aux_loss}.weight={aux_loss_w}",
            f"--aux_criterion.{aux_loss}.norm={aux_loss_norm}",
            f"--optimizer.lr={lr}",
            f"--model.pos_encoding={enc}",
            f"--dataset.train_valid_split_ratio=[0.8, 0.2]",
            f"--model.kernel_list={[32]*n_layer}",
            f"--model.kernel_size_list={[1]*n_layer}",
            f"--model.padding_list={[0]*n_layer}",
            f"--model.mode_list={[(10, 32)]*n_layer}",
            f"--train_noise.input_gaussian.std={in_std}",
            f"--train_noise.target_gaussian.std={tar_std}",
            f"--train_noise.input_downsample.scale_factor={in_scale}",
            f"--train_noise.target_downsample.scale_factor={tar_scale}",
            f"--train_noise.input_quant.prec={in_prec}",
            f"--train_noise.target_quant.prec={tar_prec}",
            f"--run.n_epochs={epochs}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    # tasks = [[1, "cmae", True, "tv_loss", 0.001, True, 0.001, "exp", 5, 5, 1]] # deep, bsconv stem, augpath, pre_norm after f_conv, xx+xy, xx*xy cat
    # tasks = [[1, "cmae", True, "tv_loss", 0.001, True, 0.001, "exp", 5, 5, 2]] # deep, bsconv stem, augpath, pre_norm after f_conv, xx+xy, xx*xy cat
    tasks = [
        # [1, "cmae", True, "tv_loss", 0.001, True, 0.001, "exp", 5, 5, 0.001, 0.03, 1,   1,   "fp32", "fp32", 1],
        # [1, "cmae", True, "tv_loss", 0.001, True, 0.001, "exp", 5, 5, 0.001, 0.05, 1,   1,   "fp32", "fp32", 1],
        # [1, "cmae", True, "tv_loss", 0.001, True, 0.001, "exp", 5, 5, 0,    0,    0.8, 0.8, "fp32", "fp32", 1],
        # [1, "cmae", True, "tv_loss", 0.001, True, 0.001, "exp", 5, 5, 0,    0,    0.6, 0.6, "fp32", "fp32", 1],
        # [1, "cmae", True, "tv_loss", 0.001, True, 0.001, "exp", 5, 5, 0,    0,    1,   1,   "int8", "int8", 1],
        [1, "cmae", True, "tv_loss", 0.001, True, 0.001, "exp", 5, 5, 0,    0,    1,   1,   "int8", "int8", 2],
    ]

    # scan gaussian noise
    tasks = [
        [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0.0, 0.0, 1,   1,   "fp32", "fp32", 100, 1],
        # [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0.03, 0.03, 1,   1,   "fp32", "fp32", 100, 1],
        # [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0.05, 0.05, 1,   1,   "fp32", "fp32", 100, 1],
        # [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0.07, 0.07, 1,   1,   "fp32", "fp32", 100, 1],
        # [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0.09, 0.09, 1,   1,   "fp32", "fp32", 100, 1],
        # [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0.11, 0.11, 1,   1,   "fp32", "fp32", 100, 1],
        # [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0.15, 0.15, 1,   1,   "fp32", "fp32", 100, 1],
        # [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0.2,   0.2, 1,   1,   "fp32", "fp32", 100, 1],
    ][::-1]

    # scam downsampling factor
    # tasks = [
    #     # [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0, 0, 1,    1,   "fp32", "fp32", 100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0, 0, 0.8,0.8,   "fp32", "fp32", 100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0, 0, 0.7,0.7,   "fp32", "fp32", 100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0, 0, 0.6,0.6,   "fp32", "fp32", 100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0, 0, 0.5,0.5,   "fp32", "fp32", 100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0, 0, 0.4,0.4,   "fp32", "fp32", 100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0, 0, 0.333,0.333,   "fp32", "fp32", 100, 1],
    #     # [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0, 0, 0.25,0.25,   "fp32", "fp32", 100, 1],
    # ][::-1]


    # scan quantization precision
    # tasks = [
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0,    0,    1,   1,   "fp16", "fp16", 100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0,    0,    1,   1,   "bfp16","bfp16",100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0,    0,    1,   1,   "int16","int16",100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0,    0,    1,   1,   "int8", "int8", 100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0,    0,    1,   1,   "int7", "int7", 100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0,    0,    1,   1,   "int6", "int6", 100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0,    0,    1,   1,   "int5", "int5", 100, 1],
    #     [1, "cmae", True, "tv_loss", 0.00, True, 0.001, "exp", 5, 5, 0,    0,    1,   1,   "int4", "int4", 100, 1],
    # ]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
