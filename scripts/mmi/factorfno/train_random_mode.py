'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-12 02:02:41
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
root = f"log/{dataset}/{model}/{exp_name}_mode"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    mode_1, mode_2, id = args
    data_list = [f"rHz_{i}" for i in range(5)]
    with open(os.path.join(root, f'rHz5_mode-{mode_1}x{mode_2}_id-{id}.log'), 'w') as wfid:
        exp = [
            f"--dataset.pol_list={str(data_list)}",
            f"--dataset.processed_dir=random_size5",
            f"--dataset.test_ratio=0.1",
            f"--plot.interval=50",
            f"--plot.dir_name={model}_{exp_name}_rHz5_mixup-1_cmae_exp_nl-8_mode-{mode_1}x{mode_2}_{id}",
            f"--run.log_interval=50",
            f"--run.random_state={41+id}",
            f"--run.n_epochs=80",
            f"--dataset.augment.prob=1",
            f"--criterion.name=cmae",
            f"--criterion.norm=True",
            f"--aux_criterion.tv_loss.weight=0",
            f"--aux_criterion.tv_loss.norm=False",
            f"--optimizer.lr=0.002",
            f"--model.pos_encoding=exp",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            f"--model.kernel_list={[64]*8}",
            f"--model.kernel_size_list={[1]*8}",
            f"--model.padding_list={[1]*8}",
            f"--model.mode_list={[(mode_1, mode_2)]*8}",
            f"--model.with_cp=True",
            f"--checkpoint.model_comment=mode-{mode_1}x{mode_2}"
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    machine = os.uname()[1]
    from itertools import product
    mode_1_list = [10, 20, 30, 40]
    mode_2_list = [10, 40, 70, 100, 130, 160, 190]
    tasks = []
    for i, (mode_1, mode_2) in enumerate(product(mode_1_list, mode_2_list)):
        tasks.append((mode_1, mode_2, i))
    tasks = {
        "eda05": tasks[:7],
        "eda13": tasks[7:14],
        "eda14": tasks[14:21],
        "eda15": tasks[21:]}
    print(tasks)
    print(tasks[machine])
    # exit(0)
    with Pool(3) as p:
        p.map(task_launcher, tasks[machine])
    logger.info(f"Exp: {configs.run.experiment} Done.")
