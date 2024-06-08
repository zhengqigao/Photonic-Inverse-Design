'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-30 17:14:41
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-30 17:17:26
'''
'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-03-29 15:26:53
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-29 15:32:28
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
root = f"log/{dataset}/{model}/{exp_name}_feat"
script = "test.py"
config_file = f"configs/{dataset}/{model}/{exp_name}/train.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    dataset, test_mode, mixup, ckpt, id = args
    n_layer = 12
    n_data = 5
    data_list = [f"{dataset}_{int(i)}" for i in range(n_data)]
    with open(os.path.join(root, f"{dataset}_{test_mode}_mixup-{mixup}_id-{id}.log"), "w") as wfid:
        exp = [
            f"--dataset.pol_list={str(data_list)}",
            f"--dataset.processed_dir=random_size5",
            f"--dataset.test_ratio=0.965",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            f"--plot.interval=50",
            f"--plot.dir_name=test_{model}_{exp_name}_{dataset}_mixup-{mixup}_cmae_exp_nl-{n_layer}_{test_mode}_{id}",
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
            f"--run.test_split=test",
            f"--run.test_random_state=0",
            f"--model.aug_path=False",
        ]
        logger.info(f"running command {' '.join(pres + exp)}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    machine = os.uname()[1]

    tasks = [
        (
            f"rHz",
            "feat",
            1,
            "./checkpoint/mmi/ffno/train_random/FFNO2d__err-0.1207_epoch-195.pt",
            1,
        )
    ]

    tasks = {"eda13": tasks}
    with Pool(1) as p:
        p.map(task_launcher, tasks[machine])
    logger.info(f"Exp: {configs.run.experiment} Done.")
