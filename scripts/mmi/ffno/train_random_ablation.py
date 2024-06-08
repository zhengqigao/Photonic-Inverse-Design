'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-07-29 03:37:32
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
root = f"log/{dataset}/{model}/{exp_name}_ablation"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    mixup, enc, ffn, ffnc, drop, aug_path, stem, n_layer, id = args
    n_data = 5
    loss = "cmae"
    loss_norm = True
    lr = 0.001
    data_list = [f"rHz_{i}" for i in range(n_data)]
    with open(os.path.join(root, f'rHz{n_data}_mixup-{mixup}_loss-{loss}_enc-{enc}_nl-{n_layer}_drop-{drop:.1f}_ffn-{ffn}_ffnc-{ffnc}_aug-{aug_path}_stem-{stem}_id-{id}.log'), 'w') as wfid:
        exp = [
            f"--dataset.pol_list={str(data_list)}",
            f"--dataset.processed_dir=random_size{n_data}",
            f"--dataset.test_ratio=0.1",
            f"--plot.interval=50",
            f"--plot.dir_name={model}_{exp_name}_rHz{n_data}_mixup-{mixup}_{loss}_{enc}_nl-{n_layer}_drop-{drop:.1f}_ffn-{ffn}_ffnc-{ffnc}_aug-{aug_path}_stem-{stem}_{id}",
            f"--run.log_interval=50",
            f"--run.random_state={41+id}",
            f"--dataset.augment.prob={mixup}",
            f"--criterion.name={loss}",
            f"--criterion.norm={loss_norm}",
            f"--aux_criterion.tv_loss.weight=0.005",
            f"--aux_criterion.tv_loss.norm=True",
            f"--optimizer.lr={lr}",
            f"--model.pos_encoding={enc}",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            f"--model.kernel_list={[64]*n_layer}",
            f"--model.kernel_size_list={[1]*n_layer}",
            f"--model.padding_list={[1]*n_layer}",
            f"--model.drop_path_rate={drop}",
            f"--model.mode_list={[(40, 70)]*n_layer}",
            f"--model.with_cp=False",
            f"--model.ffn={ffn}",
            f"--model.ffn_dwconv={ffnc}",
            f"--model.aug_path={aug_path}",
            f"--model.conv_stem={stem}",

            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    machine = os.uname()[1]

    tasks = [
        [1, "exp", 0, 0.1, 0, 1, 12, 2],  # no ffn
        # [1, "exp", 1, 0.1, 0, 1, 12, 1], # no aug_path done, wrong
        [1, "exp", 1, 0.1, 0, 1, 12, 2], # no aug_path done
        [1, "exp", 1, 0.1, 1, 0, 12, 1], # no conv stem
        [1, "exp", 1, 0.1, 1, 1, 12, 1], # no conv stem
        ]
    tasks = [
        # [1, "exp_full", 1, 1, 0.1, 0, 1, 12, 1],
        [1, "exp", 1, 1, 0.1, 0, 1, 12, 3], # for gelu before ffn id 3
        [1, "exp", 1, 0, 0.1, 0, 1, 12, 1], # for no dwconv in ffn
        # [1, "raw", 1, 1, 1, 0.1, 0, 1, 12, 1],
        # [1, "exp_noeps", 1, 1, 0.1, 0, 1, 12, 1],
        # [1, "exp_full_r", 1, 1, 0.1, 0, 1, 12, 1],
        ]
    # mixup, enc, ffn, ffnc, drop, aug_path, stem, n_layer, id
    tasks = [
        [1, "exp", 1, 1, 0.1, 0, 1, 12, 3], # no aug_path done, tv_loss=0
        [1, "exp", 1, 1, 0.1, 0, 1, 12, 2], # no aug_path done, tv_loss=0.005
        ] # for error bar
    tasks = [
        [1, "exp", 1, 1, 0.1, 0, 1, 12, 42], # no aug_path done, tv_loss=0.005
        [1, "exp", 1, 1, 0.1, 0, 1, 12, 0], # no aug_path done, tv_loss=0.005
        ] # for error bar
    tasks = {
        "eda05": tasks[0:1],
        "eda13": tasks[1:2],
        # "eda05": tasks[2:3],
        # "eda14": tasks[3:4],
    }
    print(tasks)
    print(tasks[machine])

    with Pool(1) as p:
        p.map(task_launcher, tasks[machine])
    logger.info(f"Exp: {configs.run.experiment} Done.")
