'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-30 14:41:56
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "mmi"
model = "ffno"
exp_name = "tune"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'tune.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    dataset, dataset_ids, n_layer, n_lp_epochs, n_ft_epochs, bs, ft_lr, ckpt, id = args
    n_data = len(dataset_ids)
    loss = "cmae"
    loss_norm = True
    lr = 0.001
    data_list = [f"{dataset}_{i}" for i in dataset_ids]
    with open(os.path.join(root, f'{dataset}-{n_data}_nl-{n_layer}_lp-{n_lp_epochs}_ft-{n_ft_epochs}_id-{id}.log'), 'w') as wfid:
        exp = [
            f"--dataset.pol_list={str(data_list)}",
            f"--dataset.processed_dir={dataset}-{n_data}",
            f"--dataset.test_ratio={0.1 if n_data == 10 else 0.2}",
            f"--run.batch_size={bs}",
            f"--plot.interval=50",
            f"--plot.dir_name=tune_{model}_{exp_name}_{dataset}-{n_data}_nl-{n_layer}_{id}",
            f"--run.log_interval=50",
            f"--run.random_state={41+id}",
            f"--dataset.augment.prob=1",
            f"--criterion.name={loss}",
            f"--criterion.norm={loss_norm}",
            f"--aux_criterion.tv_loss.weight=0.005",
            f"--aux_criterion.tv_loss.norm=True",
            f"--model.pos_encoding=exp",
            f"--dataset.train_valid_split_ratio=[0.9, 0.1]",
            f"--model.kernel_list={[64]*n_layer}",
            f"--model.kernel_size_list={[1]*n_layer}",
            f"--model.padding_list={[1]*n_layer}",
            f"--model.drop_path_rate=0.1",
            f"--model.mode_list={[(40, 70)]*n_layer}",
            f"--model.with_cp=False",
            f"--model.aug_path=False",
            f"--run.n_lp_epochs={n_lp_epochs}",
            f"--run.n_ft_epochs={n_ft_epochs}",
            f"--lp_optimizer.lr=0.002",
            f"--ft_optimizer.lr={ft_lr}",
            f"--checkpoint.resume=1",
            f"--checkpoint.restore_checkpoint={ckpt}",
            f"--checkpoint.no_linear=0",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    machine = os.uname()[1]
    # dataset, dataset_ids, n_layer, n_lp_epochs, n_ft_epochs, ckpt, id
    tasks = [
        # ["rHz_mmi4x4", [0,1,2,3,4], 12, 20, 20, 3, "./checkpoint/mmi/ffno/train_random/FFNO2d__err-0.1207_epoch-195.pt", 1],
        ["rHz_mmi2x2", [0,1,2,3,4], 12, 30, 100, 6, 0.0005, "./checkpoint/mmi/ffno/train_random/FFNO2d__err-0.1207_epoch-195.pt", 2],
        ["rHz_mmi4x4", [0,1,2,3,4], 12, 20, 30, 3, 0.0005, "./checkpoint/mmi/ffno/train_random/FFNO2d__err-0.1207_epoch-195.pt", 2],
        ["rHz_mmi5x5", [0,1,2,3,4], 12, 20, 30, 2, 0.0005, "./checkpoint/mmi/ffno/train_random/FFNO2d__err-0.1207_epoch-195.pt", 2],
        ["slot_rHz", [0,1,2,3,4,5,6,7,8,9], 12, 20, 30, 4, 0.0005, "./checkpoint/mmi/ffno/train_random/FFNO2d__err-0.1207_epoch-195.pt", 2],
        ]
    tasks = {
        # "eda05": tasks[0:1],
        "eda13": tasks[1:2],
        "eda14": tasks[2:3],
        "eda05": tasks[3:4],
    }
    print(tasks)
    print(tasks[machine])

    with Pool(1) as p:
        p.map(task_launcher, tasks[machine])
    logger.info(f"Exp: {configs.run.experiment} Done.")
