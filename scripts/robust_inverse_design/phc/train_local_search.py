'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-12-24 04:04:09
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

device = "phc"
model = "local_search"
exp_name = "train_local_search"
root = f"log/{device}/{model}/{exp_name}"
script = 'train_phc.py'
config_file = f'configs/{device}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)
checkpoint_dir = f'{device}/{model}/{exp_name}'


def task_launcher(args):

    model_name, num_rows_perside, num_cols, purturbation, gpu_id, epochs = args
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    pres = ['python3',
            script,
            config_file
            ]

    with open(os.path.join(root, f'model-{model_name}_ptb-{purturbation}_row-{num_rows_perside}_col-{num_cols}.log'), 'w') as wfid:
        exp = [
            f"--plot.train=True",
            f"--plot.valid=True",
            f"--plot.test=True",

            f"--run.gpu_id={gpu_id}",
            f"--run.n_epochs={epochs}",
            f"--run.random_state={59}",
            f"--run.fp16={False}",
            f"--run.plot_pics={False}",
            
            f"--model.name={model_name}",
            f"--model.num_rows_perside={num_rows_perside}",
            f"--model.num_cols={num_cols}",
            f"--model.purturbation={purturbation}",

            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        ["Repara_PhC_1x1", 11, 30, False, 0, 100],
        ]
    # tasks = [[0, 1]]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
