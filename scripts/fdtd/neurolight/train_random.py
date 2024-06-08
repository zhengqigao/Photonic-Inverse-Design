'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-12-26 22:21:27
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "fdtd"
model = "neurolight"
exp_name = "train_random"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)
checkpoint_dir = f'{dataset}/{model}/{exp_name}'


def task_launcher(args):
    pres = ['python3',
            script,
            config_file
            ]
    mixup, dim, n_layer, in_frames, out_channels, out_frames, offset_frames, description, epoch, gpu_id, id, task, chekpt = args
    with open(os.path.join(root, f'mix-{mixup}_nl-{n_layer}_d-{dim}_task-{task}_des-{description}_id-{id}.log'), 'w') as wfid:
        if 'mrr' in description.lower():
            dataset_dir = 'processed_small_mrr_160'
            device_list = ['mrr_random']
        elif 'mmi' in description.lower():
            dataset_dir = 'processed_small_mmi_160'
            device_list = ['mmi_3x3_L_random']
        elif 'meta' in description.lower():
            dataset_dir = 'processed_small_metaline_160'
            device_list = ['metaline_3x3']
        else:
            raise ValueError(f"dataset {description} not recognized")
        exp = [
            # f"--dataset.device_list=['mmi_3x3_L_random', 'mrr_random', 'dc_N', 'metaline_3x3']",
            f"--dataset.device_list={device_list}",
            f"--dataset.processed_dir={dataset_dir}",
            f"--dataset.in_frames={in_frames}",
            f"--dataset.offset_frames={offset_frames}",
            f"--dataset.out_frames={out_frames}",
            f"--dataset.out_channels={out_channels}",

            f"--run.{task}={True}",

            f"--plot.train=True",
            f"--plot.valid=True",
            f"--plot.test=True",
            f"--plot.interval=1",
            f"--plot.dir_name={model}_des-{description}_{exp_name}_mixup-{mixup}_id-{id}",
            f"--run.log_interval=100",
            f"--run.n_epochs={epoch}",
            f"--run.batch_size={1}",
            f"--run.gpu_id={gpu_id}",
            f"--run.random_state={59}",
            f"--run.plot_pics={False}",
            f"--dataset.augment.prob={mixup}",
            
            f"--model.out_channels={out_channels}",
            f"--model.in_channels={in_frames+out_channels+1}",

            f"--model.kernel_list={[72]*n_layer}",
            f"--model.kernel_size_list={[1]*n_layer}",
            f"--model.padding_list={[1]*n_layer}",
            f"--model.mode_list={[(84, 85)]*n_layer}" if "meta" in description.lower() else f"--model.mode_list={[(128, 129)]*n_layer}",
            f"--model.pos_encoding={'none'}",
            f"--model.num_iters={1}",
            f"--model.dim={dim}",

            f"--checkpoint.model_comment={description}",
            f"--checkpoint.restore_checkpoint={chekpt}",

            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [ 
        # [0.0, 96, 'rtv_loss', 0.005, True, 4, 1, "none", 14], # 1/eps, 70,70 mode, seperate kernels, use better enc, correct posenc, 
        # [0.0, 128, 'rtv_loss', 0, True, 10, 1, "none", 14], # 1/eps, 70,70 mode, seperate kernels, use better enc, correct posenc, 
        # [0.0, 72, 15, "NOL_outC160_inC10_MMI", 100, 1, 0],
        [0.0, 72, 6, 10, 160, 160, 10, "NOL_FINAL_outC160_inC10_MMI_TEST_SPEED", 100, 0, 0, "test_speed", './checkpoint/fdtd/neurolight2d/train_random/NeurOLight2d_NOL_FINAL_outC160_inC10_MMI_err-0.1618_epoch-100.pt'],
        # [0.0, 72, 6, 10, 160, 160, 10, "NOL_FINAL_outC160_inC10_MRR_TEST_SPEED", 100, 0, 0, "test_speed", "./checkpoint/fdtd/neurolight2d/train_random/NeurOLight2d_NOL_FINAL_outC160_inC10_MRR_err-0.1480_epoch-97.pt"],
        # [0.0, 72, 6, 10, 160, 160, 10, "NOL_FINAL_outC160_inC10_META_REALIGN_RES_TEST_SPEED", 100, 0, 0, "test_speed", "./checkpoint/fdtd/neurolight2d/train_random/NeurOLight2d_NOL_FINAL_outC160_inC10_META_REALIGN_RES_err-0.1856_epoch-100.pt"],
        # [0.0, 72, 15, "NOL_outC160_inC10_MRR", 100, 2, 0],
        # [0.0, 72, 15, "NOL_outC160_inC10_METALINE", 100, 3, 0],
        ]
    # tasks = [[0, 1]]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
