'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-18 00:48:16
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-18 12:31:33
'''
"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-17 19:50:02
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-17 19:59:37
"""
import os
import numpy as np
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "mmi"
model = "ffno"
exp_name = "train_random"
root = f"log/{dataset}/{model}/{exp_name}_mixup_test"
script = "test.py"
config_file = f"configs/{dataset}/{model}/{exp_name}/train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    test_mode, mixup, ratio, ckpt, id = args
    n_layer = 8
    data_list = [f"rHz_{i}" for i in range(10)]
    with open(
        os.path.join(root, f"rHz10_{test_mode}_mixup-{mixup}_data-{ratio:.2f}_id-{id}.log"), "w"
    ) as wfid:
        exp = [
            f"--dataset.pol_list={str(data_list)}",
            f"--dataset.processed_dir=random_size10",
            f"--dataset.test_ratio=0.1",
            f"--plot.interval=50",
            f"--plot.dir_name=test_{model}_{exp_name}_rHz10_mixup-0_cmae_exp_nl-{n_layer}_{test_mode}_{id}",
            f"--run.log_interval=50",
            f"--run.random_state={41+id}",
            f"--dataset.augment.prob=0",
            f"--dataset.augment.random_vflip_ratio=0",
            f"--criterion.name=cmae",
            f"--criterion.norm=True",
            f"--aux_criterion.tv_loss.weight=0",
            f"--aux_criterion.tv_loss.norm=False",
            f"--optimizer.lr=0.002",
            f"--model.pos_encoding=exp",
            f"--model.kernel_list={[64]*n_layer}",
            f"--model.kernel_size_list={[1]*n_layer}",
            f"--model.padding_list={[1]*n_layer}",
            f"--model.mode_list={[(40, 70)]*n_layer}",
            f"--model.with_cp=False",
            f"--checkpoint.resume=1",
            f"--checkpoint.restore_checkpoint={ckpt}",
            f"--run.test_mode={test_mode}",
        ]
        logger.info(f"running command {' '.join(pres + exp)}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    machine = os.uname()[1]

    tasks = [
        ("mm", 0, 0.1, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.10_mixup-0_err-0.8967_epoch-272.pt", 1),
        ("mm", 0, 0.2, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.20_mixup-0_err-0.8848_epoch-138.pt", 1),
        ("mm", 0, 0.4, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.40_mixup-0_err-0.8874_epoch-60.pt", 1),
        ("mm", 0, 0.5, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.50_mixup-0_err-0.8860_epoch-46.pt", 1),
        ("mm", 0, 0.6, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.60_mixup-0_err-0.8822_epoch-39.pt", 1),
        ("mm", 0, 0.7, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.70_mixup-0_err-0.8734_epoch-32.pt", 1),
        ("mm", 0, 0.8, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.80_mixup-0_err-0.8794_epoch-29.pt", 1),
        ("mm", 0, 0.9, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.80_mixup-0_err-0.8794_epoch-29.pt", 1),
    ]

    # tasks = [
    #     ("sm", 0, 0.1, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.10_mixup-0_err-0.8967_epoch-272.pt", 1),
    #     ("sm", 0, 0.2, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.20_mixup-0_err-0.8848_epoch-138.pt", 1),
    #     ("sm", 0, 0.4, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.40_mixup-0_err-0.8874_epoch-60.pt", 1),
    #     ("sm", 0, 0.5, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.50_mixup-0_err-0.8860_epoch-46.pt", 1),
    #     ("sm", 0, 0.6, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.60_mixup-0_err-0.8822_epoch-39.pt", 1),
    #     ("sm", 0, 0.7, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.70_mixup-0_err-0.8734_epoch-32.pt", 1),
    #     ("sm", 0, 0.8, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.80_mixup-0_err-0.8794_epoch-29.pt", 1),
    #     ("sm", 0, 0.9, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.80_mixup-0_err-0.8794_epoch-29.pt", 1),
    # ]

    # tasks = [
    #     ("sm", 0, 0.1, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.10_err-0.2312_epoch-276.pt", 2),
    #     ("sm", 0, 0.2, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.20_err-0.2060_epoch-136.pt", 2),
    #     ("sm", 0, 0.4, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.40_err-0.2124_epoch-69.pt", 2),
    #     ("sm", 0, 0.5, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.50_err-0.2062_epoch-56.pt", 2),
    #     ("sm", 0, 0.6, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.60_err-0.2072_epoch-46.pt", 2),
    #     ("sm", 0, 0.7, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.70_err-0.2021_epoch-39.pt", 2),
    #     ("sm", 0, 0.8, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.80_err-0.2058_epoch-35.pt", 2),
    #     ("sm", 0, 0.9, "./checkpoint/mmi/ffno/train_random/FFNO2d_data-0.90_err-0.2025_epoch-30.pt", 2),
    # ]
    tasks = {"eda13": tasks}
    with Pool(2) as p:
        p.map(task_launcher, tasks[machine])
    logger.info(f"Exp: {configs.run.experiment} Done.")
