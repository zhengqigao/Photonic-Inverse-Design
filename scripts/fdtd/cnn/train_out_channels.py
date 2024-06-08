'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-13 15:54:53
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "fdtd"
model = "cnn"
exp_name = "train_out_channels"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
checkpoint_dir = f'{dataset}/{model}/{exp_name}'
configs.load(config_file, recursive=True)


def task_launcher(args):
    mixup, dim, alg, pac, input_mode, stem, se, out_frames, out_channels, kernel_size, r, id, description, gpu_id, epochs, lr, criterion, weighted_frames, criterion_weight, task, checkpt, bs, device_name, dataset_dir, roll_out_frames = args

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    pres = [
            'python3',
            script,
            config_file
            ]
    
    with open(os.path.join(root, f'dev-{device_name}_alg-{alg}_pac-{pac}_input_mode-{input_mode}_out_frames-{out_frames}_des-{description}_task-{task}_batch_size-{bs}_lr-{lr}.log'), 'w') as wfid:
        if stem == "FNO_lifting":
            kernel_size_list = [1]
            kernel_list = [72]
            stride_list = [1]
            padding_list = [0]
            dilation_list = [1]
            groups_list = [1]
            residual = [False]
            norm_list = [False]
            act_list = [False]
            fuse_laplacian = [False]
            encoder_se = [False]
        elif stem == "NO2":
            kernel_size_list = [1, 3, 1, 3]
            kernel_list = [dim, dim, dim, dim]
            stride_list = [1, 1, 1, 1]
            padding_list = [0, 1, 0, 1]
            dilation_list = [1, 1, 1, 1]
            groups_list = [1, dim, 1, dim]
            residual = [False, True, False, True]
            norm_list = [False, True, False, True]
            act_list = [False, True, False, True]
            fuse_laplacian = [False, False, False, False]
            if se:
                encoder_se = [True, True, True, True]
            else:
                encoder_se = [False, False, False, False]
        else:
            raise ValueError(f"stem {stem} not recognized")

        exp = [
            # f"--dataset.device_list=['mmi_3x3_L_random', 'mrr_random', 'dc_N', 'etchedmmi_3x3_L_random', 'metaline_3x3]",
            f"--dataset.device_list=[{'mmi_3x3_L_random'}]",
            # f"--dataset.processed_dir=processed_small_more_frames",
            f"--dataset.processed_dir={dataset_dir}",
            # f"--dataset.processed_dir=processed_small_mmi",
            # f"--dataset.processed_dir=processed_small",
            # f"--dataset.processed_dir=processed_small_dc",
            f"--dataset.in_frames={10}",
            f"--dataset.offset_frames={10}",
            f"--dataset.out_frames={roll_out_frames}",
            f"--dataset.out_channels={roll_out_frames}",
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

            f"--criterion.name={criterion}",
            f"--criterion.weighted_frames={weighted_frames}",
            f"--criterion.weight={criterion_weight}",
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

            f"--model.dim={dim}",
            f"--model.field_norm_mode={'max'}",
            f"--model.input_cfg.input_mode={input_mode}",
            f"--model.input_cfg.include_src={True}",
            f"--model.input_cfg.eps_lap={False}",
            f"--model.out_channels={out_channels}",
            f"--model.num_iters={out_frames//out_channels}",
            f"--model.max_propagating_filter={0}",
            f"--model.out_frames={out_frames}",

            ## encoder cfg, this is fixed to skip + NeurOLight style stem
            f"--model.encoder_cfg.kernel_size_list={kernel_size_list}",
            f"--model.encoder_cfg.kernel_list={kernel_list}",
            f"--model.encoder_cfg.stride_list={stride_list}",
            f"--model.encoder_cfg.padding_list={padding_list}",
            f"--model.encoder_cfg.dilation_list={dilation_list}",
            f"--model.encoder_cfg.groups_list={groups_list}",
            f"--model.encoder_cfg.residual={residual}",
            f"--model.encoder_cfg.norm_list={norm_list}",
            f"--model.encoder_cfg.act_list={act_list}",
            f"--model.encoder_cfg.fuse_laplacian={fuse_laplacian}",
            f"--model.encoder_cfg.se={encoder_se}",
            f"--model.encoder_cfg.pac={False}",

            ## backbone cfg
            f"--model.backbone_cfg.conv_cfg.type={alg}", 
            f"--model.backbone_cfg.kernel_size_list={[kernel_size]*r}",
            f"--model.backbone_cfg.kernel_list={[dim]*r}" if "2d" in alg else f"--model.backbone_cfg.conv_cfg.kernel_list={[1]*r}",
            f"--model.backbone_cfg.stride_list={[1]*r}",
            f"--model.backbone_cfg.padding_list={[kernel_size//2]*r}",
            f"--model.backbone_cfg.dilation_list={[1]*r}",
            f"--model.backbone_cfg.groups_list={[1]*r}",
            f"--model.backbone_cfg.norm_list={[True]*r}",
            f"--model.backbone_cfg.act_list={[True]*r}",
            f"--model.backbone_cfg.residual={[True]*r}",
            f"--model.backbone_cfg.conv_cfg.r={r}",
            f"--model.backbone_cfg.share_weight={False}",
            f"--model.backbone_cfg.num_shared_layers={1}",
            f"--model.backbone_cfg.fuse_laplacian={[False]*r}",
            f"--model.backbone_cfg.se={[se]*r}",
            f"--model.backbone_cfg.pac={pac}",

            ## decoder cfg
            f"--model.decoder_cfg.conv_cfg.type={'Conv2d'}",
            f"--model.decoder_cfg.residual={[False, False]}",
            f"--model.decoder_cfg.fuse_laplacian={[False, False]}",
            f"--model.decoder_cfg.act_cfg.type={'GELU'}",
            f"--model.decoder_cfg.kernel_list={[512, out_channels]}",
            f"--model.decoder_cfg.se={[se]*2}",
            f"--model.decoder_cfg.pac={False}",

            f"--checkpoint.model_comment={description}",
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
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "NO2", False, 5, 5, 3, 8, 0, "outC5_KS3", 1, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1, "mmi_3x3_L_random", "processed_small_mmi_100"],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "NO2", False, 10, 10, 3, 8, 0, "outC10_KS3", 1, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1, "mmi_3x3_L_random", "processed_small_mmi_100"],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "NO2", False, 20, 20, 7, 8, 0, "outC20_KS7", 2, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1, "mmi_3x3_L_random", "processed_small_mmi_100"],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "NO2", False, 40, 40, 11, 8, 0, "outC40_KS11", 3, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1, "mmi_3x3_L_random", "processed_small_mmi_100"],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "NO2", False, 80, 80, 21, 8, 0, "outC80_KS21", 2, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1, "mmi_3x3_L_random", "processed_small_mmi_100"],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "FNO_lifting", False, 80, 80, 17, 8, 0, "Exp5_outC80_KS17", 3, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1, "mmi_3x3_L_random", "processed_small_mmi_160"],

        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "FNO_lifting", False, 80, 80, 17, 8, 0, "Exp5_outC80_KS17_TEST_SPEED", 0, 100, 0.002, "nmse", 0, 0.1, "test_speed", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_Exp5_outC80_KS17_err-0.0315_epoch-94.pt", 1, "mmi_3x3_L_random", "processed_small_mmi_160", 80],
        [0.0, 72, "Conv2d", False, "eps_E0_Ji", "FNO_lifting", False, 80, 80, 17, 8, 0, "TEST_SPEED_FOR_DIFF_KS", 0, 100, 0.002, "nmse", 0, 0.1, "test_speed_for_diff_ks", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_Exp5_outC80_KS17_err-0.0315_epoch-94.pt", 1, "mmi_3x3_L_random", "processed_small_mmi_160", 80],

        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "NO2", False, 5, 5, 3, 8, 0, "outC5_KS3_TEST_ROLL_OUT", 0, 100, 0.002, "nmse", 0, 0.1, "test_roll_out", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_outC5_KS3_err-0.0028_epoch-97.pt", 1, "mmi_3x3_L_random", "processed_small_mmi_160", 160],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "NO2", False, 10, 10, 3, 8, 0, "outC10_KS3_TEST_ROLL_OUT", 1, 100, 0.002, "nmse", 1, 0.1, "test_roll_out", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_outC10_KS3_err-0.0032_epoch-97.pt", 1, "mmi_3x3_L_random", "processed_small_mmi_160", 160],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "NO2", False, 20, 20, 7, 8, 0, "outC20_KS7_TEST_ROLL_OUT", 2, 100, 0.002, "nmse", 2, 0.1, "test_roll_out", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_outC20_KS7_err-0.0046_epoch-97.pt", 1, "mmi_3x3_L_random", "processed_small_mmi_160", 160],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "NO2", False, 40, 40, 11, 8, 0, "outC40_KS11_TEST_ROLL_OUT", 3, 100, 0.002, "nmse", 3, 0.1, "test_roll_out", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_outC40_KS11_err-0.0080_epoch-97.pt", 1, "mmi_3x3_L_random", "processed_small_mmi_160", 160],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", "NO2", False, 80, 80, 21, 8, 0, "outC80_KS21_TEST_ROLL_OUT", 0, 100, 0.002, "nmse", 0, 0.1, "test_roll_out", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_outC80_KS21_err-0.0195_epoch-97.pt", 1, "mmi_3x3_L_random", "processed_small_mmi_160", 160],
        ]
    # tasks = [[0, 1]]

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
