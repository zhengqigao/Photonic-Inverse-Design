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
model = "kno"
# model = "cnn"
exp_name = "train_random"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)
checkpoint_dir = f'{dataset}/{model}/{exp_name}'


def task_launcher(args):
    mixup, dim, alg, in_frames, out_channels, out_frames, offset_frames, trans, r, n_layer, iter, pos, T1, id, field_norm_mode, norm, description, epochs, bs, gpu_id, task, checkpt, lr = args
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    pres = ['python3',
            script,
            config_file
            ]
    suffix = f'alg-{alg}_tr-{trans}_r-{r}_nl-{n_layer}_d-{dim}_T1-{int(T1)}_inC-{in_frames}_outC_{out_channels}_id-{id}_des-{description}'
    with open(os.path.join(root, suffix+'.log'), 'w') as wfid:
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
            f"--dataset.device_list={device_list}",
            f"--dataset.img_height={168}" if "metaline" in device_list[0] else f"--dataset.img_height={256}",
            f"--dataset.processed_dir={dataset_dir}",
            f"--dataset.in_frames={in_frames}",
            f"--dataset.offset_frames={offset_frames}",
            f"--dataset.out_frames={out_frames}",
            f"--dataset.out_channels={out_channels}",
            f"--dataset.augment.prob={mixup}",
            f"--run.n_epochs={epochs}",
            f"--run.batch_size={bs}",
            f"--run.gpu_id={gpu_id}",
            f"--run.{task}={True}",
            f"--run.test_mode={'whole_video'}",
            f"--run.log_interval=200",
            f"--run.random_state={59}",
            f"--run.fp16={False}",
            f"--run.plot_pics={False}",

            f"--criterion.name={'nmse'}",
            f"--criterion.weighted_frames={0}",
            f"--criterion.weight={1}",
            f"--test_criterion.name={'nmse'}",
            f"--test_criterion.weighted_frames={0}",
            f"--test_criterion.weight={1}",
            f"--scheduler.lr_min={lr*5e-3}",
            f"--plot.train=True",
            f"--plot.valid=True",
            f"--plot.test=True",
            f"--plot.interval=1",
            f"--plot.dir_name={model}_{exp_name}_mixup-{mixup}_id-{id}",
            f"--plot.autoreg={True}",

            f"--optimizer.lr={lr}",

            f"--model.out_channels={out_channels}",
            f"--model.in_channels={in_frames+ 1 + out_channels}",
            f"--model.in_frames={in_frames}",
            f"--model.kno_alg={alg}",
            f"--model.kno_r={r}",
            f"--model.mode_list=[[84, 85]]" if "metaline" in device_list[0] else f"--model.mode_list=[[128, 129]]",
            f"--model.pos_encoding={pos}",
            f"--model.num_iters={iter}",
            f"--model.dim={dim}",
            f"--model.T1={T1}",
            f"--model.transform={trans}",
            f"--model.field_norm_mode={field_norm_mode}",
            f"--model.norm={norm}",

            f"--checkpoint.model_comment={suffix}",
            f"--checkpoint.resume={False}" if checkpt == "none" else f"--checkpoint.resume={True}",
            f"--checkpoint.restore_checkpoint={checkpt}",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        # [0.0, 96, "fourier_kno", "dft", 32, 2, 1, "none", True, 14, "max", "ln", "FourierCNN_max_ln_test"]
        # [0.0, 96, "fourier_kno", "dft", 32, 2, 1, "none", True, 14, "max", "ln", "FourierCNN_max_ln_feat_map_plot"]

        # [0.0, 72, "kno", 10, 160, 160, 10, "dft", 32, 2, 1, "none", False, 14, "max", "ln", "KNO_BASELINE_MMI_TEST_SPEED", 100, 1, 0, "test_speed", "./checkpoint/fdtd/kno/train_random/KNO2d_alg-kno_tr-dft_r-6_nl-2_d-72_T1-0_inC-10_outC_160_id-14_des-KNO_BASELINE_MMI_err-0.2100_epoch-100.pt", 0.002],
        # [0.0, 72, "kno", 10, 160, 160, 10, "dft", 6, 2, 1, "none", False, 14, "max", "ln", "KNO_BASELINE_MRR_TEST_SPEED", 100, 1, 0, "test_speed", "./checkpoint/fdtd/kno/train_random/KNO2d_alg-kno_tr-dft_r-6_nl-2_d-72_T1-0_inC-10_outC_160_id-14_des-KNO_BASELINE_MRR_err-0.1811_epoch-96.pt", 0.002],
        [0.0, 72, "kno", 10, 160, 160, 10, "dft", 6, 2, 1, "none", False, 14, "max", "ln", "KNO_BASELINE_META_TEST_SPEED", 100, 1, 0, "test_speed", "./checkpoint/fdtd/kno/train_random/KNO2d_alg-kno_tr-dft_r-6_nl-2_d-72_T1-0_inC-10_outC_160_id-14_des-KNO_BASELINE_META_err-0.2657_epoch-100.pt", 0.002],
        ]
    # tasks = [[0, 1]]

    with Pool() as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
