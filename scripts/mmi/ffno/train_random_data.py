'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-17 17:13:44
'''
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
root = f"log/{dataset}/{model}/{exp_name}_data"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    ratio, id = args
    n_layer = 8
    data_list = [f"rHz_{i}" for i in range(10)]
    train_data = int(512 * 10 * 0.9 * ratio) # test set has 512 data = 10%
    n_iter = 32270 # 1843 data, 4 bs, 70 epochs
    n_epoch = int(np.round(n_iter / np.ceil(train_data / 4)))
    with open(os.path.join(root, f'rHz10_data-{ratio:.2f}_id-{id}.log'), 'w') as wfid:
        exp = [
            f"--dataset.pol_list={str(data_list)}",
            f"--dataset.processed_dir=random_size10",
            f"--dataset.test_ratio=0.1",
            f"--plot.interval=50",
            f"--plot.dir_name={model}_{exp_name}_rHz10_mixup-1_cmae_exp_nl-{n_layer}_data-{ratio:.2f}_{id}",
            f"--run.log_interval=50",
            f"--run.random_state={41+id}",
            f"--run.n_epochs={n_epoch}",
            f"--dataset.augment.prob=1",
            f"--criterion.name=cmae",
            f"--criterion.norm=True",
            f"--aux_criterion.tv_loss.weight=0",
            f"--aux_criterion.tv_loss.norm=False",
            f"--optimizer.lr=0.002",
            f"--model.pos_encoding=exp",
            f"--dataset.train_valid_split_ratio=[{ratio}, 0.1]",
            f"--model.kernel_list={[64]*n_layer}",
            f"--model.kernel_size_list={[1]*n_layer}",
            f"--model.padding_list={[1]*n_layer}",
            f"--model.mode_list={[(40, 70)]*n_layer}",
            f"--model.with_cp=False",
            f"--checkpoint.model_comment=data-{ratio:.2f}"
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    machine = os.uname()[1]
    train_ratio_list = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tasks = []
    for i, ratio in enumerate(train_ratio_list):
        tasks.append((ratio, i))
    tasks = {
        "eda05": tasks[1:2],
        "eda13": tasks[2:4],
        "eda14": tasks[4:6],
        "eda15": tasks[6:]}
    print(tasks)
    print(tasks[machine])
    # exit(0)
    with Pool(1) as p:
        p.map(task_launcher, tasks[machine])
    logger.info(f"Exp: {configs.run.experiment} Done.")
