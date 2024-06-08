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
exp_name = "train_plot_pics"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
checkpoint_dir = f'{dataset}/{model}/{exp_name}'
configs.load(config_file, recursive=True)


def task_launcher(args):
    mixup, dim, alg, pac, input_mode, eps_lap, field_norm_mode, stem, include_src, fuse_lap, se, dec_act, in_frames, out_frames, out_channels, offset_frames, kernel_size, backbone_dilation, r, id, share_weight, num_shared_layers, description, gpu_id, epochs, lr, criterion, weighted_frames, criterion_weight, task, checkpt, bs = args
    assert pac == True if alg == "PacConv2d" else pac == False
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    pres = [
            'python3',
            script,
            config_file
            ]
    suffix = f"alg-{alg}_of-{out_frames}_oc-{out_channels}_ks-{kernel_size}_di-{backbone_dilation}_r-{r}_des-{description}_id-{id}"
    with open(os.path.join(root, f'{suffix}.log'), 'w') as wfid:
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
            # f"--dataset.device_list=['mmi_3x3_L_random', 'mrr_random', 'dc_N', 'etchedmmi_3x3_L_random', 'metaline_3x3']",
            f"--dataset.device_list=['mmi_3x3_L_random']",
            # f"--dataset.processed_dir=processed_small_more_frames",
            f"--dataset.processed_dir=processed_small_mmi_160",
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
            f"--run.random_state={39}",
            f"--run.fp16={False}",

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
            f"--model.field_norm_mode={field_norm_mode}",
            f"--model.input_cfg.input_mode={input_mode}",
            f"--model.input_cfg.include_src={include_src}",
            f"--model.input_cfg.eps_lap={eps_lap}",
            f"--model.out_channels={out_channels}",
            f"--model.num_iters={out_frames//out_channels}",

            f"--model.guidance_generator_cfg.if_pre_dwconv={False}",

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
            f"--model.encoder_cfg.conv_cfg.padding_mode={'zeros'}" if pac else f"--model.encoder_cfg.conv_cfg.padding_mode={'replicate'}",
            f"--model.encoder_cfg.if_pre_dwconv={False}",

            ## backbone cfg
            f"--model.backbone_cfg.conv_cfg.type={alg}", 
            f"--model.backbone_cfg.kernel_size_list={[kernel_size]*r}",
            f"--model.backbone_cfg.kernel_list={[dim]*r}" if "2d" in alg else f"--model.backbone_cfg.conv_cfg.kernel_list={[1]*r}",
            f"--model.backbone_cfg.stride_list={[1]*r}",
            f"--model.backbone_cfg.padding_list={[kernel_size//2]*r}",
            f"--model.backbone_cfg.dilation_list={[backbone_dilation]*r}",
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
            f"--model.backbone_cfg.conv_cfg.padding_mode={'zeros'}" if pac else f"--model.backbone_cfg.conv_cfg.padding_mode={'replicate'}",
            f"--model.backbone_cfg.if_pre_dwconv={True}" if backbone_dilation > 1 else f"--model.backbone_cfg.if_pre_dwconv={False}",

            ## decoder cfg
            f"--model.decoder_cfg.conv_cfg.type={'Conv2d'}",
            f"--model.decoder_cfg.residual={[False, False]}",
            f"--model.decoder_cfg.fuse_laplacian={[False, False]}",
            f"--model.decoder_cfg.act_cfg.type={dec_act}",
            f"--model.decoder_cfg.kernel_list={[512, out_channels]}",
            f"--model.decoder_cfg.se={[se]*2}",
            f"--model.decoder_cfg.pac={False}",
            f"--model.decoder_cfg.conv_cfg.padding_mode={'zeros'}" if pac else f"--model.decoder_cfg.conv_cfg.padding_mode={'replicate'}",
            f"--model.decoder_cfg.if_pre_dwconv={False}",

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
        # [0.0, 72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 100, 100, 10, 17, 2, 8, 18, False, 1, "outC100_KS17_DI2_MRR", 0, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 100, 100, 10, 21, 1, 8, 18, False, 1, "outC80_KS17_DI1_MRR", 0, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],

        # [0.0, 72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 21, 1, 16, 18, False, 1, "outC160_KS17_LAY6_MRR", 0, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 17, 4, 16, 18, False, 1, "PAC_FINAL_outC160_KS17_LAY16_MMI", 2, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 17, 4, 16, 18, False, 1, "PAC_FINAL_outC160_KS17_LAY16_MRR", 1, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        [0.0, 72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 17, 4, 16, 18, False, 1, "PAC_FINAL_outC160_KS17_LAY16_MMI_JUST_PLOT_FIGS", 0, 100, 0.002, "nmse", 0, 0.1, "plot_pics", "none", 1],
        ]
    # tasks = [[0, 1]]

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
