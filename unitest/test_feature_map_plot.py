import argparse
import os
from typing import Callable, Dict, Iterable, List, Tuple
import torch.cuda.amp as amp
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter, logger as lg
from pyutils.loss import KLLossMixed
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler
import torch.fft
from core import builder
from core.datasets.mixup import Mixup, MixupAll
from core.utils import DeterministicCtx, normalize, make_axes_locatable, plot_compare, print_stat
import matplotlib.pyplot as plt
from torch import Tensor
import h5py
import yaml
import random

def fft2d_and_shift(tensor): # utils function to calculate the fft2d and shift the zero frequency to the center
    """
    Apply 2D FFT on the last two dimensions of a 4D tensor, shift the zero frequency to the center,
    and compute the magnitude of the complex numbers.

    Args:
    tensor (torch.Tensor): A 4D tensor.

    Returns:
    torch.Tensor: The magnitude of the FFT2D with zero frequency shifted to the center.
    """
    # Apply 2D FFT on the last two dimensions
    fft_result = torch.fft.fft2(tensor, dim=(-2, -1))
    
    # Shift the zero frequency component to the center
    fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
    # Calculate the magnitude
    magnitude = torch.abs(fft_shifted)
    
    return magnitude

# a dict to store the activations.
# uitl function to get the activation map
activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

def plot_activation(tensor, filename, number_of_slices=96, is_freq_domain=False):
    # Ensure the tensor is on CPU and converted to numpy for plotting
    tensor = tensor.squeeze(0).transpose(-2, -1)
    tensor_np = tensor.cpu().numpy() if torch.is_tensor(tensor) else tensor
    if is_freq_domain:
        tensor_np = fft2d_and_shift(tensor).cpu().numpy()

    # Calculate the number of rows and columns to display the tensor slices
    if number_of_slices == 96:
        nrows = 8  # For example, 8 rows
        ncols = 12  # and 12 columns for 96 subplots
    elif number_of_slices == 50:
        nrows = 5
        ncols = 10
    elif number_of_slices == 128:
        nrows = 8
        ncols = 16
    else:
        raise ValueError(f"Unsupported number of slices: {number_of_slices}")

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2, nrows * 2))
    
    # Iterate over all subplots and fill them with tensor slices
    for i, ax in enumerate(axes.flat):
        if i < tensor_np.shape[0]:
            ax.imshow(tensor_np[i], cmap='RdBu_r')
            ax.axis('off')  # Hide axes ticks
        else:
            ax.axis('off')  # Ensure off for unused subplots

    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust space between plots
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the figure
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

def visualize_tensors(tensors, file_name):

    assert tensors.ndim == 4, f"Expected a 4D tensor, got {tensors.ndim}"

    tensors = [tensors[i, ...] for i in range(tensors.shape[0])]

    # Calculate grid size for subplots
    grid_size = int(np.ceil(np.sqrt(len(tensors))))
    
    fig = plt.figure(figsize=(20, 20))
    
    for i, tensor in enumerate(tensors):
        ax = fig.add_subplot(grid_size, grid_size, i + 1, projection='3d')
        # Flatten the tensor for plotting
        x, y, z = np.indices(tensor.shape)
        ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=tensor.flatten(), cmap="RdBu_r")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f"Cube {i+1}")

    # Adjust layout
    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    # Save the figure
    plt.savefig(file_name)
    plt.close()

def feature_map_plot(model, plot_path, test_loader, device):
    model.eval()
    # register forward hooks on the layers of choice
    # this is for KNO model
    # uncomment this if you want to visualize the feature maps of the KNO model
    # h1 = model.encoder[0].act_func.register_forward_hook(getActivation('encoder_0'))
    # h2 = model.encoder[1].norm.register_forward_hook(getActivation('encoder_1'))
    # h3 = model.kernel[0].k_conv.register_forward_hook(getActivation('kernel_0_k_conv'))
    # h4 = model.kernel[0].conv.register_forward_hook(getActivation('kernel_0_conv'))
    # h5 = model.kernel[0].norm.register_forward_hook(getActivation('kernel_0_norm'))
    # h6 = model.kernel[0].act_func.register_forward_hook(getActivation('kernel_0_act_func'))
    # h7 = model.kernel[1].k_conv.register_forward_hook(getActivation('kernel_1_k_conv'))
    # h8 = model.kernel[1].conv.register_forward_hook(getActivation('kernel_1_conv'))
    # h9 = model.kernel[1].norm.register_forward_hook(getActivation('kernel_1_norm'))
    # h10 = model.decoder[0].act_func.register_forward_hook(getActivation('decoder_0'))
    # h11 = model.decoder[1].conv.register_forward_hook(getActivation('decoder_1'))

    ## this is for conv model
    ## uncomment this if you want to visualize the feature maps of the conv model
    # h1 = model.conv_stem[0].act_func.register_forward_hook(getActivation('conv_stem_0'))
    # h2 = model.conv_stem[1].norm.register_forward_hook(getActivation('conv_stem_1'))
    # h3 = model.conv_stem[2].act_func.register_forward_hook(getActivation('conv_stem_2'))
    # h4 = model.conv_stem[3].act_func.register_forward_hook(getActivation('conv_stem_3'))
    # h5 = model.conv_stem[4].act_func.register_forward_hook(getActivation('conv_stem_4'))
    # h6 = model.conv_stem[5].act_func.register_forward_hook(getActivation('conv_stem_5'))
    # h7 = model.predictor[0].act_func.register_forward_hook(getActivation('predictor_0'))
    # h8 = model.predictor[1].conv.register_forward_hook(getActivation('predictor_1'))

    ## this is for the FourierCNN model
    ## uncomment this if you want to visualize the feature maps of the FourierCNN model
    h1 = model.encoder[0].act_func.register_forward_hook(getActivation('encoder_0'))
    h2 = model.encoder[1].norm.register_forward_hook(getActivation('encoder_1'))
    h3 = model.fourier_cnn[0].norm.register_forward_hook(getActivation('fourier_cnn_0'))
    h4 = model.decoder[0].act_func.register_forward_hook(getActivation('decoder_0'))
    h5 = model.decoder[1].conv.register_forward_hook(getActivation('decoder_1'))

    total_dataset = test_loader.dataset
    device_type = ""
    # while device_type != 1:
    state = random.getstate()
    random.seed()  # Resets the seed to a value based on the current time
    picked_index = random.randint(0, len(total_dataset))
    random.setstate(state)

    raw_data, raw_target = total_dataset[picked_index]
    device_type = raw_data["device_type"]
        # print("lottery: ", picked_index, device_type)
    for key, d in raw_data.items():
        if key == "device_type":
            continue
        raw_data[key] = d.to(device, non_blocking=True).unsqueeze(0)
    for key, t in raw_target.items():
        raw_target[key] = t.to(device, non_blocking=True).unsqueeze(0)
    data = torch.cat([raw_data["eps"], raw_data["Ez"], raw_data["source"]], dim=1)
    target = raw_target["Ez"]

    output, normalization_factor = model(data, target, print_info=False, plot=False, grid_step=raw_data["grid_step"], src_mask=raw_data["src_mask"])

    # remove the hooks
    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    h5.remove()
    # # ----------
    # # comment h6~h8 if you want to visualize the feature maps of the FourierCNN model
    # h6.remove()
    # h7.remove()
    # h8.remove()
    # # ----------
    # # comment h9~h11 if you want to visualize the feature maps of the CNN model
    # h9.remove()
    # h10.remove()
    # h11.remove()

    # plot the feature maps
    # for key in activation.keys():
    #     # now the shape of the actviation map are all 1, # of frames, 256, 256
    #     plot_activation(activation[key][0], f'{plot_path}/{key}.png', number_of_slices=activation[key].shape[1])
    #     plot_activation(activation[key][0], f'{plot_path}/{key}_freq.png', number_of_slices=activation[key].shape[1], is_freq_domain=True)

    weights = {}
    ## this is for visualize the weight for CNN model
    # weights["weight0"] = model.conv_stem[0].conv.weight
    # weights["weight1"] = model.conv_stem[1].conv.weight
    # weights["weight2"] = model.conv_stem[2].conv.weight
    # weights["weight3"] = model.conv_stem[3].conv.weight
    # weights["weight4"] = model.conv_stem[4].conv.weight
    # weights["weight5"] = model.conv_stem[5].conv.weight
    # weights["weight6"] = model.predictor[0].conv.weight
    # weights["weight7"] = model.predictor[1].conv.weight

    # ## this is for visualize the weight for KNO model
    # weights["encoder_0"] = model.encoder[0].conv.weight
    # weights["encoder_1"] = model.encoder[1].conv.weight
    # temp = torch.view_as_complex(model.kernel[0].k_conv.kernel)
    # temp = torch.abs(temp)
    # weights["kernel_0_k_conv_0"] = temp[0]
    # weights["kernel_0_k_conv_1"] = temp[1]
    # weights["kernel_0_conv"] = model.kernel[0].conv.weight
    # temp = torch.view_as_complex(model.kernel[1].k_conv.kernel)
    # temp = torch.abs(temp)
    # weights["kernel_1_k_conv_0"] = temp[0]
    # weights["kernel_1_k_conv_1"] = temp[1]
    # weights["kernel_1_conv"] = model.kernel[1].conv.weight
    # weights["decoder_0"] = model.decoder[0].conv.weight
    # weights["decoder_1"] = model.decoder[1].conv.weight

    ## this is for visualize the weight for FourierCNN model
    weights["my_param"] = model.fourier_cnn[0].my_param.squeeze().unsqueeze(0)

    for weight_key in weights.keys():
        # if "k_conv" in weight_key:
        #     visualize_tensors(weights[weight_key].cpu().detach().numpy(), f'{plot_path}/{weight_key}.png')
        # else:
        #     continue
        visualize_tensors(weights[weight_key].cpu().detach().numpy(), f'{plot_path}/{weight_key}.png')
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if "backbone_cfg" in configs.model.keys():
        if configs.model.backbone_cfg.conv_cfg.type == "Conv2d" or configs.model.backbone_cfg.conv_cfg.type == "LargeKernelConv2d":
            if "r" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["r"]
            if "is_causal" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["is_causal"]
            if "mask_shape" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["mask_shape"]
            if "enable_padding" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["enable_padding"]

    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )
    lg.info(model)

    _, _, test_loader = builder.make_dataloader()

    load_model(
        model,
        configs.checkpoint.restore_checkpoint,
        ignore_size_mismatch=int(configs.checkpoint.no_linear),
    )
    print("model loaded successfully!", flush=True)
    feature_map_plot(model, f'./plot/feature_map/{configs.checkpoint.model_comment}', test_loader, device)

if __name__ == "__main__":
    main()