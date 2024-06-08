'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-02 02:35:21
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "mmi"
model = "msfno"
exp_name = "train_random"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    mixup, id = args
    with open(os.path.join(root, f'rHz5_mixup-{mixup}_id-{id}.log'), 'w') as wfid:
        exp = [
            f"--dataset.pol_list=['rHz_0', 'rHz_1', 'rHz_2', 'rHz_3', 'rHz_4']",
            f"--plot.interval=10",
            f"--plot.dir_name={model}_{exp_name}_rHz5_mixup-{mixup}",
            f"--run.log_interval=50",
            f"--run.random_state={41+id}",
            f"--dataset.augment.prob={mixup}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [[0.8, 1]]
    # tasks = [[0, 1]]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
