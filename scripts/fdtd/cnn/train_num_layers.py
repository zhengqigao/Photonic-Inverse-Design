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
exp_name = "train_num_layers"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
checkpoint_dir = f'{dataset}/{model}/{exp_name}'
configs.load(config_file, recursive=True)


def task_launcher(args):
    mixup, dim, alg, pac, input_mode, eps_lap, field_norm_mode, stem, include_src, fuse_lap, se, dec_act, in_frames, out_frames, out_channels, offset_frames, kernel_size, r, id, share_weight, num_shared_layers, description, gpu_id, epochs, lr, criterion, if_spatial_mask, weighted_frames, criterion_weight, task, checkpt, bs, roll_out_frames = args

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    pres = [
            'python3',
            script,
            config_file
            ]
    
    with open(os.path.join(root, f'alg-{alg}_pac-{pac}_input_mode-{input_mode}_field_norm-{field_norm_mode}_in_frames-{in_frames}_out_frames-{out_frames}_offset_frames-{offset_frames}_des-{description}_task-{task}_batch_size-{bs}_lr-{lr}.log'), 'w') as wfid:
        if 'mrr' in description.lower():
            if roll_out_frames > 160:
                dataset_dir = 'processed_small_mrr_320'
            else:
                dataset_dir = 'processed_small_mrr_160'
            device_list = ['mrr_random']
            guidance_kernel_size_list = [5,5,5,5]
            guidance_padding_list = [2,2,2,2]
        elif 'mmi' in description.lower():
            if roll_out_frames > 160:
                dataset_dir = 'processed_small_mmi_320'
            else:
                dataset_dir = 'processed_small_mmi_160'
            device_list = ['mmi_3x3_L_random']
            guidance_kernel_size_list = [3,3,5,5]
            guidance_padding_list = [1,1,2,2]
        elif 'meta' in description.lower():
            if roll_out_frames > 160:
                dataset_dir = 'processed_small_meta_320'
            else:
                dataset_dir = 'processed_small_metaline_160'
            device_list = ['metaline_3x3']
            guidance_kernel_size_list = [3,3,5,5]
            guidance_padding_list = [1,1,2,2]
        else:
            raise ValueError(f"dataset {description} not recognized")
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
            # f"--dataset.device_list=['mmi_3x3_L_random', 'mrr_random', 'dc_N', 'etchedmmi_3x3_L_random']",
            f"--dataset.device_list={device_list}",
            # f"--dataset.processed_dir=processed_small_more_frames",
            f"--dataset.processed_dir={dataset_dir}",
            # f"--dataset.processed_dir=processed_small_mmi",
            # f"--dataset.processed_dir=processed_small",
            # f"--dataset.processed_dir=processed_small_dc",
            f"--dataset.in_frames={in_frames}",
            f"--dataset.offset_frames={offset_frames}",
            # f"--dataset.out_frames={out_frames}",
            # f"--dataset.out_channels={out_channels}",
            f"--dataset.out_frames={roll_out_frames}",
            f"--dataset.out_channels={roll_out_frames}",

            f"--dataset.augment.prob={mixup}",
            f"--run.n_epochs={epochs}",
            f"--run.batch_size={bs}",
            f"--run.gpu_id={gpu_id}",
            f"--run.{task}={True}",
            f"--run.test_mode={'whole_video'}",
            f"--run.log_interval=200",
            f"--run.random_state={41+id}",
            f"--run.fp16={False}",
            f"--run.plot_pics={False}",

            f"--criterion.name={criterion}",
            f"--criterion.weighted_frames={weighted_frames}",
            f"--criterion.weight={criterion_weight}",
            f"--criterion.if_spatial_mask={if_spatial_mask}",
            f"--test_criterion.name={'nmse'}",
            f"--test_criterion.weighted_frames={0}",
            f"--test_criterion.weight={1}",
            f"--scheduler.lr_min={lr*5e-3}",
            f"--plot.train=True",
            f"--plot.valid=True",
            f"--plot.test=True",
            f"--plot.interval=1",
            f"--plot.dir_name={model}_{exp_name}_des-{description}_id-{id}",
            f"--plot.autoreg={True}",
            f"--optimizer.lr={lr}",

            f"--model.dim={dim}",
            f"--model.field_norm_mode={field_norm_mode}",
            f"--model.input_cfg.input_mode={input_mode}",
            f"--model.input_cfg.include_src={include_src}",
            f"--model.input_cfg.eps_lap={eps_lap}",
            f"--model.out_channels={out_channels}",
            f"--model.num_iters={out_frames//out_channels}",
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
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 5, 20, 18, False, 1, "outC50_KS5_LAY20_MMI", 0, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 11, 8, 18, False, 1, "outC50_KS11_LAY8_MMI", 1, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 13, 7, 18, False, 1, "outC50_KS13_LAY7_MMI", 2, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 15, 6, 18, False, 1, "outC50_KS15_LAY6_MMI", 3, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 21, 4, 18, False, 1, "outC50_KS21_LAY4_MMI", 2, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 50, 50, 10, 41, 2, 18, False, 1, "outC50_KS41_LAY2_MMI", 3, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],

        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 29, 8, 18, False, 1, "outC160_KS29_LAY8_MMI", 2, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_MRR", 2, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 21, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS21_LAY16_MRR", 2, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_META", 1, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_META_REALIGN_RES", 2, 100, 0.002, "nmse", False, 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 29, 8, 18, False, 1, "outC160_KS29_LAY8_MMI_TEST_ROLL_OUT", 2, 100, 0.002, "nmse", False, 0, 0.1, "test_roll_out", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_outC160_KS29_LAY8_MMI_err-0.0633_epoch-94.pt", 1, 320],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "outC160_KS15_LAY16_MMI_TEST_ROLL_OUT", 3, 100, 0.002, "nmse", False, 0, 0.1, "test_roll_out", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_outC160_KS15_LAY16_MMI_err-0.0609_epoch-100.pt", 1, 320],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 29, 8, 18, False, 1, "outC160_KS29_LAY8_MMI_TEST_SPEED", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_speed", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_outC160_KS29_LAY8_MMI_err-0.0633_epoch-94.pt", 1, 160],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "outC160_KS15_LAY16_MMI_TEST_SPEED", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_speed", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_outC160_KS15_LAY16_MMI_err-0.0609_epoch-100.pt", 1, 160],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "outC160_KS15_LAY16_MMI_FIND_AVG_SIZE", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_speed", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_outC160_KS15_LAY16_MMI_err-0.0609_epoch-100.pt", 1, 160],

        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 17, 8, 18, False, 1, "outC80_KS17_LAY8_MMI", 2, 100, 0.002, "nmse", 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 17, 8, 18, False, 1, "Exp_5_up_outC80_KS17_LAY8_MMI_timemask", 3, 100, 0.002, "masknmse", False, 10, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 17, 8, 18, False, 1, "Exp_5_up_outC80_KS17_LAY8_MMI_spatialmask", 2, 100, 0.002, "masknmse", True, 0, 0.1, "train", "none", 1],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 17, 8, 18, False, 1, "Exp_5_up_outC80_KS17_LAY8_MMI_bothmask", 1, 100, 0.002, "masknmse", True, 10, 0.1, "train", "none", 1],

        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 17, 8, 18, False, 1, "Exp_5_up_outC80_KS17_LAY8_MMI_timemask_TEST_ROLL_OUT", 3, 100, 0.002, "nmse", False, 10, 0.1, "test_roll_out", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_Exp_5_up_outC80_KS17_LAY8_MMI_timemask_err-0.0195_epoch-94.pt", 1, 160],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 17, 8, 18, False, 1, "Exp_5_up_outC80_KS17_LAY8_MMI_spatialmask_TEST_ROLL_OUT", 2, 100, 0.002, "nmse", True, 0, 0.1, "test_roll_out", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_Exp_5_up_outC80_KS17_LAY8_MMI_spatialmask_err-0.0205_epoch-94.pt", 1, 160],
        # [0.0, 72, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 17, 8, 18, False, 1, "Exp_5_up_outC80_KS17_LAY8_MMI_bothmask_TEST_ROLL_OUT", 1, 100, 0.002, "nmse", True, 10, 0.1, "test_roll_out", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_Exp_5_up_outC80_KS17_LAY8_MMI_bothmask_err-0.0200_epoch-94.pt", 1, 160],

        # [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_MMI_REDUCE_PARA", 3, 100, 0.002, "nmse", False, 0, 0.1, "train", "none", 1],
        # [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 21, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS21_LAY16_MRR_REDUCE_PARA", 2, 100, 0.002, "nmse", False, 0, 0.1, "train", "none", 1],
        # [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_META_REDUCE_PARA", 1, 100, 0.002, "nmse", False, 0, 0.1, "train", "none", 1],

        # [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_MMI_REDUCE_PARA_TEST_SPEED", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_speed", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_MMI_REDUCE_PARA_err-0.0747_epoch-94.pt", 1, 160],
        # [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 21, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS21_LAY16_MRR_REDUCE_PARA_TEST_SPEED", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_speed", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_FINAL_RESULT_SIMPLECNN_outC160_KS21_LAY16_MRR_REDUCE_PARA_err-0.0898_epoch-99.pt", 1, 160],
        # [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_META_REDUCE_PARA_TEST_SPEED", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_speed", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_META_REDUCE_PARA_err-0.1253_epoch-100.pt", 1, 160],

        # [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_MMI_REDUCE_PARA_TEST_SPEED", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_speed", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_MMI_REDUCE_PARA_err-0.0747_epoch-94.pt", 1, 160],
        # [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 21, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS21_LAY16_MRR_REDUCE_PARA_FIND_AVG_SIZE", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_speed", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_FINAL_RESULT_SIMPLECNN_outC160_KS21_LAY16_MRR_REDUCE_PARA_err-0.0898_epoch-99.pt", 1, 160],
        # [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_META_REDUCE_PARA_FIND_AVG_SIZE", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_speed", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_META_REDUCE_PARA_err-0.1253_epoch-100.pt", 1, 160],

        # [0.0, 64, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 3, 8, 18, False, 1, "KS3_LAY8_MMI_TEST_RUNTIME_MEMORY", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_runtime_memory", "none", 1, 80],
        # [0.0, 64, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 5, 8, 18, False, 1, "KS5_LAY8_MMI_TEST_RUNTIME_MEMORY", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_runtime_memory", "none", 1, 80],
        # [0.0, 64, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 7, 8, 18, False, 1, "KS7_LAY8_MMI_TEST_RUNTIME_MEMORY", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_runtime_memory", "none", 1, 80],
        # [0.0, 64, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 9, 8, 18, False, 1, "KS9_LAY8_MMI_TEST_RUNTIME_MEMORY", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_runtime_memory", "none", 1, 80],
        # [0.0, 64, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 11, 8, 18, False, 1, "KS11_LAY8_MMI_TEST_RUNTIME_MEMORY", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_runtime_memory", "none", 1, 80],
        # [0.0, 64, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 13, 8, 18, False, 1, "KS13_LAY8_MMI_TEST_RUNTIME_MEMORY", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_runtime_memory", "none", 1, 80],
        # [0.0, 64, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 15, 8, 18, False, 1, "KS15_LAY8_MMI_TEST_RUNTIME_MEMORY", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_runtime_memory", "none", 1, 80],
        # [0.0, 64, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 17, 8, 18, False, 1, "KS17_LAY8_MMI_TEST_RUNTIME_MEMORY", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_runtime_memory", "none", 1, 80],
        # [0.0, 64, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 19, 8, 18, False, 1, "KS19_LAY8_MMI_TEST_RUNTIME_MEMORY", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_runtime_memory", "none", 1, 80],
        # [0.0, 64, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 80, 80, 10, 21, 8, 18, False, 1, "KS21_LAY8_MMI_TEST_RUNTIME_MEMORY", 0, 100, 0.002, "nmse", False, 0, 0.1, "test_runtime_memory", "none", 1, 80],
        
        # [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_MMI_REDUCE_PARA_POL_VIS", 0, 100, 0.002, "nmse", False, 0, 0.1, "prediction_visualization", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_MMI_REDUCE_PARA_err-0.0747_epoch-94.pt", 1, 160],
        # [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 21, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS21_LAY16_MRR_REDUCE_PARA_POL_VIS", 0, 100, 0.002, "nmse", False, 0, 0.1, "prediction_visualization", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_FINAL_RESULT_SIMPLECNN_outC160_KS21_LAY16_MRR_REDUCE_PARA_err-0.0898_epoch-99.pt", 1, 160],
        [0.0, 32, "Conv2d", False, "eps_E0_Ji", False, "max", "NO2", True, False, False, "GELU", 10, 160, 160, 10, 15, 16, 18, False, 1, "FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_META_REDUCE_PARA_POL_VIS", 0, 100, 0.002, "nmse", False, 0, 0.1, "prediction_visualization", "./checkpoint/fdtd/cnn/train_num_layers/FourierCNN_FINAL_RESULT_SIMPLECNN_outC160_KS15_LAY16_META_REDUCE_PARA_err-0.1253_epoch-100.pt", 1, 160],
        ]
    # tasks = [[0, 1]]

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
