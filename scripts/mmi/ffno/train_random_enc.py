'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-03 23:11:01
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "mmi"
model = "ffno"
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
    mixup, loss, lr, enc, n_data, n_layer, id = args
    data_list = [f"rHz_{i}" for i in range(n_data)]
    with open(os.path.join(root, f'rHz{n_data}_mixup-{mixup}_loss-{loss}_enc-{enc}_id-{id}.log'), 'w') as wfid:
        exp = [
            f"--dataset.pol_list={str(data_list)}",
            f"--plot.interval=10",
            f"--plot.dir_name={model}_{exp_name}_rHz{n_data}_mixup-{mixup}_{loss}_{enc}_{id}",
            f"--run.log_interval=50",
            f"--run.random_state={41+id}",
            f"--dataset.augment.prob={mixup}",
            f"--criterion.name={loss}",
            f"--optimizer.lr={lr}",
            f"--model.pos_encoding={enc}",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            # f"--model.dim=64",
            f"--model.kernel_list={[64]*n_layer}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    # tasks = [[0.8, "cmae", 0.002, "exp4", 1]]
    tasks = [[1, "cmae", 0.002, "exp", 5, 6, 1]] # cat
    tasks = [[1, "cmae", 0.002, "exp", 5, 6, 2]] # +
    tasks = [[1, "cmae", 0.002, "exp", 5, 6, 3]] # cat double channel
    tasks = [[1, "cmae", 0.002, "exp", 5, 6, 4]] # cat double channel
    tasks = [[1, "cmae", 0.002, "exp", 5, 6, 5]] # residual learning
    tasks = [[1, "cmae", 0.002, "exp", 5, 6, 6]] # hrvit stem
    tasks = [[1, "cmae", 0.002, "exp", 5, 6, 7]] # bsconv stem, augpath
    tasks = [[1, "cmae", 0.002, "exp", 5, 8, 8]] # deep, bsconv stem, augpath
    # tasks = [[0, 1]]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
