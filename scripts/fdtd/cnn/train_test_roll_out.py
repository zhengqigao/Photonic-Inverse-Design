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
exp_name = "train_test_roll_out"
root = f"log/{dataset}/{model}/{exp_name}"
# script = 'train_push_forward.py'
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    mixup, dim, alg, pac, input_mode, eps_lap, field_norm_mode, stem, include_src, fuse_lap, se, dec_act, in_frames, out_frames, out_channels, offset_frames, kernel_size, r, id, share_weight, num_shared_layers, description, gpu_id, epochs, lr, criterion, weighted_frames, criterion_weight, task, checkpt, bs, propagating_filter = args

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    pres = [
            'python3',
            script,
            config_file
            ]
    
    with open(os.path.join(root, f'alg-{alg}_pac-{pac}_input_mode-{input_mode}_field_norm-{field_norm_mode}_in_frames-{in_frames}_out_frames-{out_frames}_offset_frames-{offset_frames}_des-{description}_pf-{propagating_filter}_task-{task}_batch_size-{bs}_lr-{lr}.log'), 'w') as wfid:
        if stem == "FNO_lifting":
            kernel_size_list = [1]
            kernel_list = [96]
            stride_list = [1]
            padding_list = [0]
            dilation_list = [1]
            groups_list = [1]
            residual = [False]
            norm_list = [False]
            act_list = [False]
            fuse_laplacian = [False]
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
            if_pre_dwconv = [False, False, False, False]
            if fuse_lap:
                fuse_laplacian = [False, True, False, False]
            else:
                fuse_laplacian = [False, False, False, False]
            if se:
                encoder_se = [True, True, True, True]
            else:
                encoder_se = [False, False, False, False]
        else:
            raise ValueError(f"stem {stem} not recognized")

        exp = [
            # f"--dataset.device_list=['mmi_3x3_L_random', 'mrr_random', 'dc_N', 'etchedmmi_3x3_L_random']",
            f"--dataset.device_list=['mmi_3x3_L_random']",
            # f"--dataset.processed_dir=processed_small_more_frames",
            f"--dataset.processed_dir=processed_small_mmi",
            # f"--dataset.processed_dir=processed_small_mmi",
            # f"--dataset.processed_dir=processed_small",
            # f"--dataset.processed_dir=processed_small_dc",
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
            f"--run.random_state={41+id}",
            f"--run.fp16={True}",

            f"--criterion.name={criterion}",
            f"--criterion.weighted_frames={weighted_frames}",
            f"--criterion.weight={criterion_weight}",
            # f"--test_criterion.name={'unitymaskmse'}",
            f"--test_criterion.name={'unitymaskmse'}" if criterion == "maskmse" else f"--test_criterion.name={'nmse'}",
            f"--test_criterion.weighted_frames={0}",
            f"--test_criterion.weight={1}",
            f"--scheduler.lr_min={lr*5e-3}",
            f"--plot.train=True",
            f"--plot.valid=True",
            f"--plot.test=True",
            f"--plot.interval=1",
            f"--plot.dir_name={model}_{exp_name}_mixup-{mixup}_id-{id}_gating",
            f"--plot.autoreg={False}",
            f"--optimizer.lr={lr}",

            f"--model.dim={dim}",
            f"--model.field_norm_mode={field_norm_mode}",
            f"--model.input_cfg.input_mode={input_mode}",
            f"--model.input_cfg.include_src={include_src}",
            f"--model.input_cfg.eps_lap={eps_lap}",
            f"--model.out_channels={out_channels}",
            f"--model.num_iters={out_frames//out_channels}",
            f"--model.max_propagating_filter={propagating_filter}",

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
            f"--model.encoder_cfg.if_pre_dwconv={if_pre_dwconv}",

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
            f"--model.backbone_cfg.share_weight={share_weight}",
            f"--model.backbone_cfg.num_shared_layers={num_shared_layers}",
            f"--model.backbone_cfg.fuse_laplacian={[False]*r}",
            f"--model.backbone_cfg.se={[se]*r}",
            f"--model.backbone_cfg.pac={pac}",
            f"--model.backbone_cfg.if_pre_dwconv={[False]*r}",

            ## decoder cfg
            f"--model.decoder_cfg.conv_cfg.type={'Conv2d'}",
            f"--model.decoder_cfg.residual={[False, False]}",
            f"--model.decoder_cfg.fuse_laplacian={[False, False]}",
            f"--model.decoder_cfg.act_cfg.type={dec_act}",
            f"--model.decoder_cfg.kernel_list={[512, out_channels]}",
            f"--model.decoder_cfg.se={[se]*2}",
            f"--model.decoder_cfg.pac={False}",
            f"--model.decoder_cfg.if_pre_dwconv={[False, False]}",

            f"--checkpoint.model_comment={description}",
            f"--checkpoint.resume={False}" if checkpt == "none" else f"--checkpoint.resume={True}",
            f"--checkpoint.restore_checkpoint={checkpt}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        # test the roll out error
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "avg_pool_max", "NO2", True, False, False, "GELU", 10, 20, 20, 10, 7, 8, 0, False, 1, "outC20_avg_max_MMI", 3, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_field_norm_mode/FourierCNN_alg-Conv2d_norm-avg_pool_max_of-20_oc-20_ks-7_pf-0_di-1_r-8_id-1_err-0.0036_epoch-97.pt", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "avg_pool_max99", "NO2", True, False, False, "GELU", 10, 20, 20, 10, 7, 8, 0, False, 1, "outC20_avg_max99_MMI", 3, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_field_norm_mode/FourierCNN_alg-Conv2d_norm-avg_pool_max99_of-20_oc-20_ks-7_pf-0_di-1_r-8_id-1_err-0.0040_epoch-99.pt", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "rigional_std", "NO2", True, False, False, "GELU", 10, 20, 20, 10, 7, 8, 0, False, 1, "outC20_rigional_std_MMI", 3, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_field_norm_mode/FourierCNN_alg-Conv2d_norm-rigional_std_of-20_oc-20_ks-7_pf-0_di-1_r-8_id-1_err-0.0035_epoch-97.pt", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max99", "NO2", True, False, False, "GELU", 10, 20, 20, 10, 7, 8, 0, False, 1, "outC20_max99_MMI", 3, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_field_norm_mode/FourierCNN_alg-Conv2d_norm-max99_of-20_oc-20_ks-7_pf-0_di-1_r-8_id-1_err-0.0037_epoch-97.pt", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 20, 20, 10, 7, 8, 0, False, 1, "outC20_max_MMI", 3, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_field_norm_mode/FourierCNN_alg-Conv2d_norm-max_of-20_oc-20_ks-7_pf-0_di-1_r-8_id-1_err-0.0039_epoch-99.pt", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "std", "NO2", True, False, False, "GELU", 10, 20, 20, 10, 7, 8, 0, False, 1, "outC20_std_MMI", 3, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_field_norm_mode/FourierCNN_alg-Conv2d_norm-std_of-20_oc-20_ks-7_pf-0_di-1_r-8_id-1_err-0.0040_epoch-99.pt", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 20, 20, 10, 7, 8, 0, False, 1, "outC20_max_MMI_0p1gating_test_time", 3, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_field_norm_mode/FourierCNN_alg-Conv2d_norm-max_of-20_oc-20_ks-7_pf-0_di-1_r-8_id-1_err-0.0039_epoch-99.pt", 1, 61],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 5, 5, 10, 3, 8, 0, False, 1, "outC5_MMI_test_outC_rollout", 0, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_outC5_KS3_err-0.0028_epoch-97.pt", 1, 0],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 10, 10, 10, 3, 8, 0, False, 1, "outC10_MMI_test_outC_rollout", 1, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_outC10_KS3_err-0.0032_epoch-97.pt", 1, 0],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 20, 20, 10, 7, 8, 0, False, 1, "outC20_MMI_test_outC_rollout", 2, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_outC20_KS7_err-0.0046_epoch-97.pt", 1, 0],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 40, 40, 10, 11, 8, 0, False, 1, "outC40_MMI_test_outC_rollout", 3, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_outC40_KS11_err-0.0080_epoch-97.pt", 1, 0],
        [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 21, 8, 0, False, 1, "outC80_MMI_test_outC_rollout", 0, 100, 0.002, "nmse", 0, 0.1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_out_channels/FourierCNN_outC80_KS21_err-0.0195_epoch-97.pt", 1, 0],

        ]

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
