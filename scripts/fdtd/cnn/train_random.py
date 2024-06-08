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
exp_name = "train_random"
root = f"log/{dataset}/{model}/{exp_name}"
# script = 'train_push_forward.py'
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)


def task_launcher(args):
    mixup, dim, alg, pac, input_mode, eps_lap, field_norm_mode, stem, include_src, fuse_lap, se, dec_act, in_frames, out_frames, out_channels, offset_frames, kernel_size, r, id, share_weight, num_shared_layers, description, gpu_id, epochs, lr, criterion, weighted_frames, criterion_weight, task, checkpt, bs = args

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    pres = [
            'python3',
            script,
            config_file
            ]
    
    with open(os.path.join(root, f'alg-{alg}_pac-{pac}_input_mode-{input_mode}_field_norm-{field_norm_mode}_in_frames-{in_frames}_out_frames-{out_frames}_offset_frames-{offset_frames}_des-{description}_task-{task}_batch_size-{bs}_lr-{lr}.log'), 'w') as wfid:
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
            kernel_list = [dim-24, dim-24, dim, dim]
            stride_list = [1, 1, 1, 1]
            padding_list = [0, 1, 0, 1]
            dilation_list = [1, 1, 1, 1]
            groups_list = [1, dim-24, 1, dim]
            residual = [False, True, False, True]
            norm_list = [False, True, False, True]
            act_list = [False, True, False, True]
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
            f"--dataset.device_list=['mrr_random']",
            # f"--dataset.processed_dir=processed_small_more_frames",
            f"--dataset.processed_dir=processed_small_mrr",
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
            f"--run.test_mode={'whole_video'}" if field_norm_mode == "none" else f"--run.test_mode={'pattern'}",
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
            f"--plot.dir_name={model}_{exp_name}_mixup-{mixup}_id-{id}",
            f"--plot.autoreg={True}",
            f"--optimizer.lr={lr}",

            f"--model.dim={dim}",
            f"--model.field_norm_mode={field_norm_mode}",
            f"--model.input_cfg.input_mode={input_mode}",
            f"--model.input_cfg.include_src={include_src}",
            f"--model.input_cfg.eps_lap={eps_lap}",
            f"--model.out_channels={out_channels}",
            f"--model.num_iters={out_frames//out_channels}",

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
            f"--model.backbone_cfg.share_weight={share_weight}",
            f"--model.backbone_cfg.num_shared_layers={num_shared_layers}",
            f"--model.backbone_cfg.fuse_laplacian={[False]*r}",
            f"--model.backbone_cfg.se={[se]*r}",
            f"--model.backbone_cfg.pac={pac}",

            ## decoder cfg
            f"--model.decoder_cfg.conv_cfg.type={'Conv2d'}",
            f"--model.decoder_cfg.residual={[False, False]}",
            f"--model.decoder_cfg.fuse_laplacian={[False, False]}",
            f"--model.decoder_cfg.act_cfg.type={dec_act}",
            f"--model.decoder_cfg.kernel_list={[dim+32, out_channels]}",
            f"--model.decoder_cfg.se={[se]*2}",
            f"--model.decoder_cfg.pac={False}",

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
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 17, 5, "none", 18, False, 1, "CNN_outC50_origion", 0, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_lap_Ji", False, True, 8, 50, 17, 5, "none", 18, False, 1, "CNN_outC50_lap_input", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", True, True, 8, 50, 17, 5, "none", 18, False, 1, "CNN_outC50_fuse_laplacian_0p3", 2, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, True, 8, 1, 3, 1, "none", 18, False, 1, "CNN_outC1_origion", 0, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_lap_Ji", False, True, 8, 1, 3, 1, "none", 18, False, 1, "CNN_outC1_lap_input", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", True, True, 8, 1, 3, 1, "none", 18, False, 1, "CNN_outC1_fuse_laplacian_0p3", 3, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 1, 17, 5, "none", 18, False, 1, "CNN_outC1_RF85", 0, 30, "masknmse"], # run this later
        # --------------- sweep the RF for different devices and different output frames
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 3, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks3", 0, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 5, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks5", 0, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 7, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks7", 0, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 9, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks9", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 11, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks11", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 13, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks13", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 15, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks15", 2, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 17, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks17", 2, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 19, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks19", 2, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 21, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks21", 2, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 23, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks23", 0, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 25, 5, "none", 18, False, 1, "CNN_outC50_MMI_ks25", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 17, 5, "none", 18, False, 1, "CNN_outC25_MMI_ks15", 2, 30, "masknmse"],
        # --------------- mrr
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", False, "none", "NO2", True, False, "GELU", 10, 1, 1, 10, 17, 5, 18, False, 1, "CNN_outC1_MMI_inC10_no_norm", 0, 100, 0.002, "maskmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", False, "none", "NO2", True, False, "GELU", 10, 10, 10, 10, 17, 5, 18, False, 1, "CNN_outC10_MMI_inC10_no_norm", 1, 100, 0.002, "maskmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", False, "none", "NO2", True, False, "GELU", 10, 20, 20, 10, 17, 5, 18, False, 1, "CNN_outC20_MMI_inC10_no_norm", 2, 100, 0.002, "maskmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", False, "none", "NO2", True, False, "GELU", 10, 30, 30, 10, 17, 5, 18, False, 1, "CNN_outC30_MMI_inC10_no_norm", 3, 100, 0.002, "maskmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", False, "none", "NO2", True, False, "GELU", 10, 40, 40, 10, 17, 5, 18, False, 1, "CNN_outC40_MMI_inC10_no_norm", 1, 100, 0.002, "maskmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", False, "eps_E0_lap_Ji", False, "none", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 17, 5, 18, False, 1, "CNN_outC50_MMI_inC10_test_gradient_shrink_output", 0, 100, 0.0002, "masknmse", 0, 0.1, "train", "none", 2],
        # [0.0, 96, "Conv2d", False, "eps_E0_lap_Ji", False, "none", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 17, 5, 18, False, 1, "CNN_outC50_MMI_inC10_test_gradient", 0, 100, 0.00004, "masknmse", 0, 0.1, "train", "none", 2],
        # [0.0, 96, "Conv2d", False, "eps_E0_lap_Ji", False, "none", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 17, 5, 18, False, 1, "CNN_outC50_MMI_inC10_test_gradient", 2, 100, 0.0001, "masknmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", False, "eps_E0_lap_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 17, 5, 18, False, 1, "CNN_outC50_MMI_inC10_test_gradient", 1, 100, 0.002, "masknmse", 0, 0.1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", True, "none", "NO2", True, False, "GELU", 10, 1, 1, 10, 17, 5, 18, False, 1, "CNN_outC1_MMI_inC10_no_norm_eps_lap", 2, 100, 0.002, "maskmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", True, "none", "NO2", True, False, "GELU", 10, 1, 1, 10, 17, 5, 18, False, 1, "CNN_outC1_MMI_inC10_no_norm_eps_lap", 2, 100, 0.002, "maskmse", 0, 1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC1_MMI_inC10_no_norm_eps_lap_err-0.0000_epoch-99.pt", 2],
        # [0.0, 256, "Conv2d", "eps_E0_lap_Ji", False, "none", "NO2", True, False, False, "GELU", 10, 200, 200, 10, 17, 5, 18, False, 1, "CNN_outC200_MMI_inC10", 0, 100, 0.002, "maskmse", 0, 1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC200_MMI_inC10_err-0.0000_epoch-100.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", False, "none", "NO2", True, False, False, "GELU", 10, 10, 10, 50, 17, 5, 18, False, 1, "CNN_outC10_MMI_inC10_offset_50", 0, 100, 0.002, "maskmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", False, "eps_E0_lap_Ji", False, "none", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 17, 5, 18, False, 1, "CNN_outC50_MMI_inC10_test", 0, 100, 0.001, "masknmse", 0, 1, "train", "none", 1],
        # [0.0, 96, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 17, 5, 18, False, 1, "outC50_inC10_MMI", 3, 50, 0.002, "masknmse", 0, 0.1, "train", "none", 1],
        # [0.0, 96, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 8, 50, 50, 8, 17, 5, 18, False, 1, "outC50_inC8_MMI", 1, 50, 0.002, "masknmse", 0, 0.1, "train", "none", 1],
        # [0.0, 96, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 12, 50, 50, 12, 17, 5, 18, False, 1, "outC50_inC12_MMI", 2, 50, 0.002, "masknmse", 0, 0.1, "train", "none", 1],
    
        # [0.0, 72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 17, 5, 18, False, 1, "outC50_inC10_MRR", 0, 50, 0.002, "masknmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 8, 50, 50, 8, 17, 5, 18, False, 1, "outC50_inC8_MRR", 1, 50, 0.002, "masknmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 12, 50, 50, 12, 17, 5, 18, False, 1, "outC50_inC12_MRR", 2, 50, 0.002, "masknmse", 0, 0.1, "train", "none", 1],
        # [0.0, 96, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, True, "GELU", 10, 50, 50, 10, 17, 5, 18, False, 1, "outC50_inC10_MMI_with_SE", 3, 50, 0.002, "masknmse", 0, 0.1, "train", "none", 1],



        # Choose NO2 as the stem
        # choose the batch size 2
        # choose the lr for epoch 50 2e-3
        # chhose the input mode "eps_E0_lap_Ji" include src
        # the best input frames is 10
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "NO2", True, False, "GELU", 10, 10, 10, 17, 5, 18, False, 1, "CNN_outC10_ALL_DEV_inC10", 1, 100, 0.002, "masknmse", False, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC10_ALL_DEV_inC10_err-0.0000_epoch-86.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "NO2", True, False, "GELU", 10, 20, 10, 17, 5, 18, False, 1, "CNN_outC20_ALL_DEV_inC10", 2, 100, 0.002, "masknmse", False, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC20_ALL_DEV_inC10_err-0.0001_epoch-100.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "NO2", True, False, "GELU", 10, 30, 10, 17, 5, 18, False, 1, "CNN_outC30_ALL_DEV_inC10", 3, 100, 0.002, "masknmse", False, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC30_ALL_DEV_inC10_err-0.0002_epoch-100.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "NO2", True, False, "GELU", 10, 40, 10, 17, 5, 18, False, 1, "CNN_outC40_ALL_DEV_inC10", 1, 100, 0.002, "masknmse", False, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC40_ALL_DEV_inC10_err-0.0004_epoch-100.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "NO2", True, False, "GELU", 10, 50, 10, 17, 5, 18, False, 1, "CNN_outC50_ALL_DEV_inC10", 0, 100, 0.002, "masknmse", False, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC50_ALL_DEV_inC10_err-0.0006_epoch-100.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "NO2", True, False, "GELU", 10, 20, 10, 17, 5, 18, False, 1, "CNN_outC20_ALL_DEV_inC10_wf", 0, 100, 0.002, "masknmse", 10, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC20_ALL_DEV_inC10_wf_err-0.0001_epoch-100.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "NO2", True, False, "GELU", 10, 30, 10, 17, 5, 18, False, 1, "CNN_outC30_ALL_DEV_inC10_wf", 1, 100, 0.002, "masknmse", 10, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC30_ALL_DEV_inC10_wf_err-0.0002_epoch-100.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "NO2", True, False, "GELU", 10, 40, 10, 17, 5, 18, False, 1, "CNN_outC40_ALL_DEV_inC10_wf", 2, 100, 0.002, "masknmse", 10, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC40_ALL_DEV_inC10_wf_err-0.0004_epoch-100.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "NO2", True, False, "GELU", 10, 50, 10, 17, 5, 18, False, 1, "CNN_outC50_ALL_DEV_inC10_wf", 3, 100, 0.002, "masknmse", 10, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC50_ALL_DEV_inC10_wf_err-0.0007_epoch-100.pt", 2],

        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "max_w_src", "NO2", True, False, "GELU", 10, 10, 10, 17, 5, 18, False, 1, "CNN_outC10_ALL_DEV_inC10_max_w_src", 1, 50, 0.002, "masknmse", 0, 1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC10_ALL_DEV_inC10_max_w_src_err-0.0000_epoch-98.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "none", "NO2", True, False, "GELU", 10, 50, 10, 10, 17, 5, 18, False, 1, "CNN_outC10_inC10_outF50_MRR", 3, 50, 0.002, "maskmse", 0, 1, "train", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC10_MRR_inC10_no_norm_err-0.0000_epoch-50.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "none", "NO2", True, False, "GELU", 10, 100, 50, 10, 17, 5, 18, False, 1, "CNN_outC50_inC10_outF100_MRR", 2, 50, 0.002, "maskmse", 50, 1, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC50_inC10_outF100_MRR_err-0.0001_epoch-46.pt", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "none", "NO2", True, False, "GELU", 10, 50, 50, 10, 17, 5, 18, False, 1, "CNN_outC50_MRR_inC10_no_norm", 2, 100, 0.002, "maskmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "max_w_src", "NO2", True, False, "GELU", 10, 1, 10, 17, 5, 18, False, 1, "CNN_outC1_ALL_DEV_inC10_max_w_src", 0, 100, 0.002, "masknmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "none", "NO2", True, False, "GELU", 10, 10, 10, 17, 5, 18, False, 1, "CNN_outC10_MRR_inC10_no_norm", 1, 50, 0.002, "maskmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "max_w_src", "NO2", True, False, "GELU", 10, 20, 10, 17, 5, 18, False, 1, "CNN_outC20_ALL_DEV_inC10_max_w_src", 2, 100, 0.002, "masknmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "max_w_src", "NO2", True, False, "GELU", 10, 30, 10, 17, 5, 18, False, 1, "CNN_outC30_ALL_DEV_inC10_max_w_src", 3, 100, 0.002, "masknmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "max_w_src", "NO2", True, False, "GELU", 10, 40, 10, 17, 5, 18, False, 1, "CNN_outC40_ALL_DEV_inC10_max_w_src", 0, 100, 0.002, "masknmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "max_w_src", "NO2", True, False, "GELU", 10, 50, 10, 17, 5, 18, False, 1, "CNN_outC50_ALL_DEV_inC10_max_w_src", 3, 100, 0.002, "masknmse", 0, 1, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "max_w_src", "NO2", True, False, "GELU", 10, 10, 10, 17, 5, 18, False, 1, "CNN_outC10_ALL_DEV_inC10_wf_max_w_src", 2, 100, 0.002, "masknmse", 10, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "max_w_src", "NO2", True, False, "GELU", 10, 20, 10, 17, 5, 18, False, 1, "CNN_outC20_ALL_DEV_inC10_wf_max_w_src", 0, 100, 0.002, "masknmse", 10, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "max_w_src", "NO2", True, False, "GELU", 10, 30, 10, 17, 5, 18, False, 1, "CNN_outC30_ALL_DEV_inC10_wf_max_w_src", 1, 100, 0.002, "masknmse", 10, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "max_w_src", "NO2", True, False, "GELU", 10, 40, 10, 17, 5, 18, False, 1, "CNN_outC40_ALL_DEV_inC10_wf_max_w_src", 2, 100, 0.002, "masknmse", 10, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_lap_Ji", "max_w_src", "NO2", True, False, "GELU", 10, 50, 10, 17, 5, 18, False, 1, "CNN_outC50_ALL_DEV_inC10_wf_max_w_src", 3, 100, 0.002, "masknmse", 10, "train", "none", 2],



        # [0.0, 96, "Conv2d", "eps_E0_Ji", "NO2", False, 8, 10, 8, 17, 5, 18, False, 1, "CNN_outC10_MRR_NO2", 0, 50, 0.002, "masknmse", False, "train", "none", 1],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", "NO2", False, 8, 10, 8, 17, 5, 18, False, 1, "CNN_outC10_MRR_lr10", 0, 50, 0.01, "masknmse", False, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", "NO2", False, 8, 10, 8, 17, 5, 18, False, 1, "CNN_outC10_MRR_lr12", 1, 50, 0.012, "masknmse", False, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", "NO2", False, 8, 10, 8, 17, 5, 18, False, 1, "CNN_outC10_MRR_lr1", 2, 50, 0.001, "masknmse", False, "train", "none", 2],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", "NO2", False, 8, 10, 8, 17, 5, 18, False, 1, "CNN_outC10_MRR_NO2", 1, 50, 0.008, "masknmse", False, "train", "none", 4],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", "NO2", False, 8, 10, 8, 17, 5, 18, False, 1, "CNN_outC10_MRR_NO2", 3, 50, 0.016, "masknmse", False, "train", "none", 8],


        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 3, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks3", 1, 30, "masknmse", "train", "dummy"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 5, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks5", 2, 30, "masknmse", "train", "dummy"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 7, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks7", 3, 30, "masknmse", "train", "dummy"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 9, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks9", 1, 30, "masknmse", "train", "dummy"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 11, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks11", 1, 30, "masknmse", "train", "dummy"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 13, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks13", 3, 30, "masknmse", "train", "dummy"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 15, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks15", 1, 30, "masknmse", "train", "dummy"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 17, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks17", 2, 30, "masknmse", "train", "dummy"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 19, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks19", 1, 30, "masknmse", "train", "dummy"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 21, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks21", 1, 30, "masknmse", "train", "dummy"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 23, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks23", 2, 30, "masknmse", "train", "dummy"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 8, 50, 25, 5, "none", 18, False, 1, "CNN_outC8_OS50_MRR_ks25", 3, 30, "masknmse", "train", "dummy"],
        # test the output channel:
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 1, 8, 17, 5, 18, False, 1, "CNN_outC1_MRR_weighted_frames_F", 0, 100, 0.002, "masknmse", False, "dummy", "none"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 10, 8, 17, 5, 18, False, 1, "CNN_outC10_MRR_weighted_frames_F", 1, 100, 0.002, "masknmse", False, "dummy", "none"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 20, 8, 17, 5, 18, False, 1, "CNN_outC20_MRR_weighted_frames_F", 2, 100, 0.002, "masknmse", False, "dummy", "none"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 30, 8, 17, 5, 18, False, 1, "CNN_outC30_MRR_weighted_frames_F", 3, 100, 0.002, "masknmse", False, "dummy", "none"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 40, 8, 17, 5, 18, False, 1, "CNN_outC40_MRR_weighted_frames_F", 0, 100, 0.002, "masknmse", False, "dummy", "none"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 50, 8, 17, 5, 18, False, 1, "CNN_outC50_MRR_weighted_frames_F", 1, 100, 0.002, "masknmse", False, "dummy", "none"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 10, 8, 17, 5, 18, False, 1, "CNN_outC10_MRR_weighted_frames_T", 2, 100, 0.002, "masknmse", True, "dummy", "none"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 20, 8, 17, 5, 18, False, 1, "CNN_outC20_MRR_weighted_frames_T", 3, 100, 0.002, "masknmse", True, "dummy", "none"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 30, 8, 17, 5, 18, False, 1, "CNN_outC30_MRR_weighted_frames_T", 1, 100, 0.002, "masknmse", True, "dummy", "none"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 40, 8, 17, 5, 18, False, 1, "CNN_outC40_MRR_weighted_frames_T", 2, 100, 0.002, "masknmse", True, "dummy", "none"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 50, 8, 17, 5, 18, False, 1, "CNN_outC50_MRR_weighted_frames_T", 3, 100, 0.002, "masknmse", True, "dummy", "none"],
        # test the roll out error
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 1, 8, 17, 5, 18, False, 1, "CNN_outC1_MRR_weighted_frames_F", 0, 100, 0.002, "masknmse", False, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC1_MRR_weighted_frames_F_err-0.0000_epoch-92.pt"],

        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 10, 8, 17, 5, 18, False, 1, "CNN_outC10_MRR_weighted_frames_F", 1, 100, 0.002, "masknmse", False, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC10_MRR_weighted_frames_F_err-0.0000_epoch-92.pt"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 20, 8, 17, 5, 18, False, 1, "CNN_outC20_MRR_weighted_frames_F", 2, 100, 0.002, "masknmse", False, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC20_MRR_weighted_frames_F_err-0.0001_epoch-92.pt"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 30, 8, 17, 5, 18, False, 1, "CNN_outC30_MRR_weighted_frames_F", 3, 100, 0.002, "masknmse", False, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC30_MRR_weighted_frames_F_err-0.0001_epoch-92.pt"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 40, 8, 17, 5, 18, False, 1, "CNN_outC40_MRR_weighted_frames_F", 0, 100, 0.002, "masknmse", False, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC40_MRR_weighted_frames_F_err-0.0002_epoch-95.pt"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 50, 8, 17, 5, 18, False, 1, "CNN_outC50_MRR_weighted_frames_F", 1, 100, 0.002, "masknmse", False, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC50_MRR_weighted_frames_F_err-0.0004_epoch-99.pt"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 10, 8, 17, 5, 18, False, 1, "CNN_outC10_MRR_weighted_frames_T", 2, 100, 0.002, "masknmse", True, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC10_MRR_weighted_frames_T_err-0.0000_epoch-92.pt"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 20, 8, 17, 5, 18, False, 1, "CNN_outC20_MRR_weighted_frames_T", 3, 100, 0.002, "masknmse", True, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC20_MRR_weighted_frames_T_err-0.0001_epoch-92.pt"],

        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 30, 8, 17, 5, 18, False, 1, "CNN_outC30_MRR_weighted_frames_T", 1, 100, 0.002, "masknmse", True, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC30_MRR_weighted_frames_T_err-0.0001_epoch-99.pt"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 40, 8, 17, 5, 18, False, 1, "CNN_outC40_MRR_weighted_frames_T", 2, 100, 0.002, "masknmse", True, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC40_MRR_weighted_frames_T_err-0.0002_epoch-92.pt"],
        # [0.0, 96, "Conv2d", "eps_E0_Ji", False, 8, 50, 8, 17, 5, 18, False, 1, "CNN_outC50_MRR_weighted_frames_T", 3, 100, 0.002, "masknmse", True, "test_autoregressive", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC50_MRR_weighted_frames_T_err-0.0004_epoch-99.pt"],

        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 3, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks3", 1, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks3_err-0.0004_epoch-30.pt"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 5, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks5", 2, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks5_err-0.0004_epoch-30.pt"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 7, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks7", 3, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks7_err-0.0004_epoch-30.pt"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 9, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks9", 1, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks9_err-0.0004_epoch-30.pt"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 11, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks11", 2, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks11_err-0.0004_epoch-30.pt"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 13, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks13", 3, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks13_err-0.0004_epoch-30.pt"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 15, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks15", 1, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks15_err-0.0006_epoch-30.pt"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 17, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks17", 2, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks17_err-0.0007_epoch-30.pt"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 19, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks19", 3, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks19_err-0.0005_epoch-30.pt"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 21, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks21", 1, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks21_err-0.0007_epoch-30.pt"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 23, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks23", 2, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks23_err-0.0007_epoch-30.pt"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 25, 5, "none", 18, False, 1, "CNN_outC25_MRR_ks25", 3, 30, "masknmse", "test_first_last_frames", "./checkpoint/fdtd/cnn/train_random/FourierCNN_CNN_outC25_MRR_ks25_err-0.0007_epoch-30.pt"],
        # --------------- dc
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 3, 5, "none", 18, False, 1, "CNN_outC50_DC_ks3", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 5, 5, "none", 18, False, 1, "CNN_outC50_DC_ks5", 2, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 7, 5, "none", 18, False, 1, "CNN_outC50_DC_ks7", 3, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 9, 5, "none", 18, False, 1, "CNN_outC50_DC_ks9", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 11, 5, "none", 18, False, 1, "CNN_outC50_DC_ks11", 2, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 13, 5, "none", 18, False, 1, "CNN_outC50_DC_ks13", 3, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 15, 5, "none", 18, False, 1, "CNN_outC50_DC_ks15", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 17, 5, "none", 18, False, 1, "CNN_outC50_DC_ks17", 2, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 19, 5, "none", 18, False, 1, "CNN_outC50_DC_ks19", 3, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 21, 5, "none", 18, False, 1, "CNN_outC50_DC_ks21", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 23, 5, "none", 18, False, 1, "CNN_outC50_DC_ks23", 2, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 25, 5, "none", 18, False, 1, "CNN_outC50_DC_ks25", 3, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 27, 5, "none", 18, False, 1, "CNN_outC50_DC_ks27", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 29, 5, "none", 18, False, 1, "CNN_outC50_DC_ks29", 2, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 31, 5, "none", 18, False, 1, "CNN_outC50_DC_ks31", 3, 30, "masknmse"],


        # ---------------
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 25, 17, 5, "none", 18, False, 1, "CNN_outC25_RF85", 1, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 50, 17, 5, "none", 18, False, 1, "CNN_outC50_RF85", 2, 30, "masknmse"],
        # [0.0, 96, "conv2d", "eps_E0_Ji", False, False, 8, 75, 17, 5, "none", 18, False, 1, "CNN_outC75_RF85", 3, 30, "masknmse"],
        # [0.0, 96, "conv2d", "dft", 12, "none", 18, "max", "ln", "CNN_max_ln_feat_map_plot"]
        # [0.0, 96, "conv2d", "dft", 12, "none", 18, "max", "ln", "CNN_max_ln_auto_reg_test"] # used for Full CNN test long range iteration
        # [0.0, 96, "Fourier", "dft", 12, "none", 18, "max", "ln", "FourierCNN_max_ln_test_if_work", 1] # E0/eps
        # [0.0, 96, "Fourier", "dft", 11, "none", 18, "max", "ln", "FourierCNN_max_ln_test_if_work"] # 
        # [0.0, 96, "Fourier", "dft", 1, "none", 15, "max", "ln", "FourierCNN_max_ln_test_if_work"] # 77x77x77 Fconv3d
        # [0.0, 96, "Fourier", "dft", 1, "none", 16, "max", "ln", "FourierCNN_max_ln_test_if_work"] # 77x77x77 Fconv3d
        # [0.0, 96, "Fourier", "dft", 1, "none", 17, "max", "ln", "FourierCNN_max_ln_test_if_work"] # 77x77x77 Fconv3d
        # [0.0, 96, "Fourier", 8, 50, 5, "none", 18, False, 1, "FourierCNN_new", 0, 50] # 77x77x77 Fconv3d
        # [0.0, 96, "conv2d", "dft", 12, "none", 18, "max", "ln", "FourierCNN_max_ln_feat_map_plot"]
        ]
    # tasks = [[0, 1]]

    with Pool(4) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
