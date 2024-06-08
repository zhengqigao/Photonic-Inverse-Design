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

dataset = "fdtd"
model = "ffno"
exp_name = "train_random"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)
checkpoint_dir = f'{dataset}/{model}/{exp_name}'


def task_launcher(args):

    mixup, in_frames, out_channels, out_frames, offset_frames, r, description, task, checkpt, epochs, id, gpu_id, device_type, processed_dir = args
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    pres = ['python3',
            script,
            config_file
            ]

    with open(os.path.join(root, f'inC_{in_frames}_outC_{out_channels}_r-{r}_des-{description}_task-{task}_epochs-{epochs}.log'), 'w') as wfid:
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
            # f"--dataset.device_list=['mmi_3x3_L_random', 'mrr_random', 'metaline_3x3']",
            f"--dataset.device_list={device_list}",
            f"--dataset.processed_dir={dataset_dir}",
            f"--dataset.in_frames={in_frames}",
            f"--dataset.offset_frames={offset_frames}",
            f"--dataset.out_frames={out_frames}",
            f"--dataset.out_channels={out_channels}",
            f"--dataset.augment.prob={mixup}",

            f"--plot.train=True",
            f"--plot.valid=True",
            f"--plot.test=True",
            f"--plot.interval=1",
            f"--plot.dir_name={model}_{description}_{exp_name}_mixup-{mixup}_id-{id}",
            f"--plot.autoreg={False}",
            f"--plot.root={'plot'}",

            f"--run.log_interval=100",
            f"--run.batch_size=1",
            f"--run.gpu_id={gpu_id}",
            f"--run.{task}={True}",
            f"--run.n_epochs={epochs}",
            f"--run.random_state={59}",
            f"--run.fp16={False}",
            f"--run.plot_pics={False}",

            f"--model.out_channels={out_channels}",
            f"--model.in_channels={in_frames+ 1 + out_channels}",
            f"--model.in_frames={in_frames}",
            f"--model.dim={72}",
            f"--model.kernel_list={[72]*r}",
            f"--model.kernel_size_list={[1]*r}",
            f"--model.padding_list={[0]*r}",
            f"--model.hidden_list=[512]",
            f"--model.mode_list={[(84, 85)]*r}" if 'meta' in description.lower() else f"--model.mode_list={[(128, 129)]*r}",
            f"--model.act_func={'GELU'}",
            f"--model.domain_size=[20, 100]",
            f"--model.grid_step={1.550/20}",
            f"--model.pml_width={0}",
            f"--model.pml_permittivity={0}",
            f"--model.buffer_width={0.5}",
            f"--model.buffer_permittivity={-1e-10}",
            f"--model.dropout_rate={0.0}",
            f"--model.drop_path_rate={0.0}",
            f"--model.eps_min={2.085136}",
            f"--model.eps_max={12.3}",
            f"--model.aux_head={False}",
            f"--model.aux_head_idx={1}",
            f"--model.pos_encoding={'none'}",
            f"--model.with_cp={False}",
            f"--model.conv_stem={False}",
            f"--model.aug_path={True}",
            f"--model.ffn={True}",
            f"--model.ffn_dwconv={True}",

            f"--criterion.name={'nmse'}",
            f"--criterion.weighted_frames={0}",
            f"--criterion.weight={1}",
            f"--test_criterion.name={'nmse'}",
            f"--test_criterion.weighted_frames={0}",
            f"--test_criterion.weight={1}",

            f"--checkpoint.resume={False}" if checkpt == "none" else f"--checkpoint.resume={True}",
            f"--checkpoint.restore_checkpoint={checkpt}",
            f"--checkpoint.model_comment={description}",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        # [0.0, 10, 160, 160, 10, 12, "FFNO_outC160_inC10_MMI_TEST_SPEED", "test_speed", "./checkpoint/fdtd/ffno/train_random/FFNO2d_FFNO_outC160_inC10_MMI_err-0.0685_epoch-100.pt", 100, 0, 0, "mmi_3x3_L_random", "processed_small_mmi_160"],
        # [0.0, 10, 160, 160, 10, 12, "FFNO_outC160_inC10_MRR_TEST_SPEED", "test_speed", "./checkpoint/fdtd/ffno/train_random/FFNO2d_FFNO_outC160_inC10_MRR_err-0.1404_epoch-99.pt", 100, 0, 0, "mrr_random", "processed_small_mrr_160"],
        # [0.0, 10, 160, 160, 10, 12, "FFNO_outC160_inC10_METALINE_REALIGN_RES_TEST_SPEED", "test_speed", "./checkpoint/fdtd/ffno/train_random/FFNO2d_FFNO_outC160_inC10_METALINE_REALIGN_RES_err-0.0941_epoch-95.pt", 100, 0, 0, 'metaline_3x3', "processed_small_metaline_160"],

        # [0.0, 10, 160, 160, 10, 12, "FFNO_outC160_inC10_MMI_PLOT_VIS", "prediction_visualization", "./checkpoint/fdtd/ffno/train_random/FFNO2d_FFNO_outC160_inC10_MMI_err-0.0685_epoch-100.pt", 100, 0, 0, "mmi_3x3_L_random", "processed_small_mmi_160"],
        # [0.0, 10, 160, 160, 10, 12, "FFNO_outC160_inC10_MRR_PLOT_VIS", "prediction_visualization", "./checkpoint/fdtd/ffno/train_random/FFNO2d_FFNO_outC160_inC10_MRR_err-0.1404_epoch-99.pt", 100, 0, 0, "mrr_random", "processed_small_mrr_160"],
        [0.0, 10, 160, 160, 10, 12, "FFNO_outC160_inC10_METALINE_REALIGN_RES_PLOT_VIS", "prediction_visualization", "./checkpoint/fdtd/ffno/train_random/FFNO2d_FFNO_outC160_inC10_METALINE_REALIGN_RES_err-0.0941_epoch-95.pt", 100, 0, 0, 'metaline_3x3', "processed_small_metaline_160"],
        ]
    # tasks = [[0, 1]]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
