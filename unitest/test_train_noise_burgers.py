import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullFormatter
from pyutils.plot import set_axes_size_ratio, batch_plot, set_ms
from pyutils.torch_train import set_torch_deterministic
from sklearn.manifold import TSNE
from pyutils.general import print_stat
color_dict = {
    "black": "#000000",
    "red": "#de425b",  # red
    "blue": "#1F77B4",  # blue
    "orange": "#f58055",  # orange
    "yellow": "#f6df7f",  # yellow
    "green": "#2a9a2a",  # green
    "grey": "#979797",  # grey
    "purple": "#AF69C5",  # purple,
    "mitred": "#A31F34",  # mit red
    "pink": "#CDA2BE",
}

sys.path.append("..")

from core.datasets.burgers import (
    Burgers,
    InputDownSampleError,
    TargetDownSampleError,
    InputGaussianError,
    TargetGaussianError,
    InputQuantizationError,
    TargetQuantizationError,
)


def test_input_downsample_error():
    burgers = Burgers(
        root="data",
        train=True,
        download=True,
        encode_input=True,
        encode_output=True
    )
    print(burgers.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = burgers.data[:, :]
    target = burgers.targets[:]
    # error_fn = InputDownSampleError(size=[H2,H2], mode="nearest-exact")
    errors_mean = []
    errors_std = []
    for ratio in np.linspace(0.333, 1, 20):
        error_fn = InputDownSampleError(
            size=None,
            scale_factor=ratio,
            mode="linear",
            antialias=False,
            align_corners=True,
        )
        data_n, _ = error_fn(data, target)
        err = (
            data_n.sub(data).abs().sum(dim=[2]).div(data.abs().sum(dim=[2]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())
    # exit(0)
    name = "test_input_downsample_error_Scan_Burgers"
    fig, ax = None, None
    errors_mean = np.array(errors_mean)
    errors_std = np.array(errors_std)
    # print(len(errors_mean), len(errors_std))
    fig, ax, _ = batch_plot(
        type="errorbar",
        raw_data={
            "x": np.linspace(0.333, 1, 20),
            "y": errors_mean,
            "yerror": errors_std,
        },
        fig=fig,
        ax=ax,
        name=name,
        trace_color=color_dict["blue"],
        xlabel="Scale Factor",
        ylabel="NMAE",
        xrange=[0.3, 1.01, 0.1],
        yrange=[0.0, 0.401, 0.1],
        linestyle="-",
        trace_marker="o",
        xformat="%.1f",
        yformat="%.1f",
        linewidth=0.5,
        fontsize=10,
        legend=True,
        legend_loc="upper right",
        legend_ncol=1,
        trace_label="Data error",
    )
    set_ms()

    errors_mean = []
    errors_std = []
    for ratio in np.linspace(0.333, 1, 20):
        error_fn = TargetDownSampleError(
            size=None,
            scale_factor=ratio,
            mode="linear",
            antialias=False,
            align_corners=True,
        )
        _, target_n = error_fn(data, target)
        err = (
            target_n.sub(target)
            .abs()
            .sum(dim=[2])
            .div(target.abs().sum(dim=[2]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())

    errors_mean = np.array(errors_mean)
    errors_std = np.array(errors_std)
    # print(len(errors_mean), len(errors_std))
    fig, ax, _ = batch_plot(
        type="errorbar",
        raw_data={
            "x": np.linspace(0.333, 1, 20),
            "y": errors_mean,
            "yerror": errors_std,
        },
        fig=fig,
        ax=ax,
        name=name,
        trace_color=color_dict["mitred"],
        xlabel="Scale Factor",
        ylabel="NMAE",
        xrange=[0.3, 1.01, 0.1],
        yrange=[0.0, 0.401, 0.1],
        linestyle="--",
        trace_marker="^",
        xformat="%.1f",
        yformat="%.1f",
        linewidth=0.5,
        fontsize=10,
        legend=True,
        legend_loc="upper right",
        legend_ncol=1,
        trace_label="Target error",
    )
    set_ms()
    set_axes_size_ratio(0.5, 0.5, fig=fig, ax=ax)
    plt.savefig(f"./figs/{name}.png", dpi=300)

    # print(data, data_n)
    # print(data.shape, data_n.shape)
    # fig, axes = plt.subplots(1,2)
    # axes[0].imshow(data[0,0,0].real.cpu().numpy())
    # axes[1].imshow(data_n[0,0,0].real.cpu().numpy())
    # fig.savefig("./unitest/test_input_downsample_error.png")


def test_input_quant_error():
    burgers = Burgers(
        root="data",
        train=True,
        download=True,
        encode_input=True,
        encode_output=True
    )
    print(burgers.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = burgers.data[:, :]
    target = burgers.targets[:]
    # error_fn = InputDownSampleError(size=[H2,H2], mode="nearest-exact")
    errors_mean = []
    errors_std = []
    for prec in [
        "fp32",
        "fp16",
        "bfp16",
        "int16",
        "int8",
        "int7",
        "int6",
        "int5",
        "int4",
    ]:
        error_fn = InputQuantizationError(prec=prec)
        data_n, _ = error_fn(data, target)
        err = (
            data_n.sub(data).abs().sum(dim=[2]).div(data.abs().sum(dim=[2]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())
    name = "test_input_quant_error_Scan_Burgers"
    fig, ax = None, None
    errors_mean = np.array(errors_mean)
    errors_std = np.array(errors_std)
    # print(len(errors_mean), len(errors_std))
    fig, ax, _ = batch_plot(
        type="errorbar",
        raw_data={"x": np.arange(9), "y": errors_mean, "yerror": errors_std},
        fig=fig,
        ax=ax,
        name=name,
        trace_color=color_dict["blue"],
        xlabel="Quantization Bitwidth",
        ylabel="NMAE",
        xrange=[0, 8.01, 1],
        yrange=[0.0, 0.101, 0.02],
        xlimit=[-0.1, 8.1],
        linestyle="-",
        trace_marker="o",
        xformat="%.1f",
        yformat="%.2f",
        linewidth=0.5,
        fontsize=10,
        legend=True,
        legend_loc="upper right",
        legend_ncol=1,
        trace_label="Data error",
    )
    set_ms()

    errors_mean = []
    errors_std = []
    for prec in [
        "fp32",
        "fp16",
        "bfp16",
        "int16",
        "int8",
        "int7",
        "int6",
        "int5",
        "int4",
    ]:
        error_fn = TargetQuantizationError(prec=prec)
        _, target_n = error_fn(data, target)
        err = (
            target_n.sub(target)
            .abs()
            .sum(dim=[2])
            .div(target.abs().sum(dim=[2]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())

    errors_mean = np.array(errors_mean)
    errors_std = np.array(errors_std)
    # print(len(errors_mean), len(errors_std))
    fig, ax, _ = batch_plot(
        type="errorbar",
        raw_data={"x": np.arange(9), "y": errors_mean, "yerror": errors_std},
        fig=fig,
        ax=ax,
        name=name,
        trace_color=color_dict["mitred"],
        xlabel="Quantization Bitwidth",
        ylabel="NMAE",
        xrange=[0, 8.01, 1],
        yrange=[0.0, 0.101, 0.02],
        xlimit=[-0.1, 8.1],
        linestyle="--",
        trace_marker="^",
        xformat="%.1f",
        yformat="%.2f",
        linewidth=0.5,
        fontsize=10,
        legend=True,
        legend_loc="upper right",
        legend_ncol=1,
        trace_label="Target error",
        tick_label="fp32,fp16,bfp16,int16,int8,int7,int6,int5,int4".split(","),
    )
    set_ms()
    plt.xticks(rotation=35)
    set_axes_size_ratio(0.5, 0.5, fig=fig, ax=ax)
    plt.savefig(f"./figs/{name}.png", dpi=300)


def test_input_gaussian_error():
    burgers = Burgers(
        root="data",
        train=True,
        download=True,
        encode_input=True,
        encode_output=True
    )
    print(burgers.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = burgers.data[:, :]
    target = burgers.targets[:]

    # error_fn = InputDownSampleError(size=[H2,H2], mode="nearest-exact")
    errors_mean = []
    errors_std = []
    for std in np.linspace(0.02, 0.2, 10):
        error_fn = InputGaussianError(std=std)
        data_n, _ = error_fn(data, target)
        err = (
            data_n.sub(data).abs().sum(dim=[2]).div(data.abs().sum(dim=[2]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())
    name = "test_input_gaussian_error_Scan_Burgers"
    fig, ax = None, None
    errors_mean = np.array(errors_mean)
    errors_std = np.array(errors_std)
    # print(len(errors_mean), len(errors_std))
    fig, ax, _ = batch_plot(
        type="errorbar",
        raw_data={
            "x": np.linspace(0.02, 0.2, 10),
            "y": errors_mean,
            "yerror": errors_std,
        },
        fig=fig,
        ax=ax,
        name=name,
        trace_color=color_dict["blue"],
        xlabel="Gaussian Noise std.",
        ylabel="NMAE",
        xrange=[0, 0.21, 0.05],
        yrange=[0.0, 0.501, 0.1],
        linestyle="-",
        trace_marker="o",
        xformat="%.2f",
        yformat="%.1f",
        linewidth=0.5,
        fontsize=10,
        legend=True,
        legend_loc="upper left",
        legend_ncol=1,
        trace_label="Data error",
    )
    set_ms()

    errors_mean = []
    errors_std = []
    for std in np.linspace(0.02, 0.2, 10):
        error_fn = TargetGaussianError(std=std)
        _, target_n = error_fn(data, target)
        err = (
            target_n.sub(target)
            .abs()
            .sum(dim=[2])
            .div(target.abs().sum(dim=[2]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())

    errors_mean = np.array(errors_mean)
    errors_std = np.array(errors_std)
    # print(len(errors_mean), len(errors_std))
    fig, ax, _ = batch_plot(
        type="errorbar",
        raw_data={
            "x": np.linspace(0.02, 0.2, 10),
            "y": errors_mean,
            "yerror": errors_std,
        },
        fig=fig,
        ax=ax,
        name=name,
        trace_color=color_dict["mitred"],
        xlabel="Gaussian Noise std.",
        ylabel="NMAE",
        xrange=[0, 0.21, 0.05],
        yrange=[0.0, 0.501, 0.1],
        linestyle="--",
        trace_marker="^",
        xformat="%.2f",
        yformat="%.1f",
        linewidth=0.5,
        fontsize=10,
        legend=True,
        legend_loc="upper left",
        legend_ncol=1,
        trace_label="Target error",
    )
    set_ms()
    set_axes_size_ratio(0.5, 0.5, fig=fig, ax=ax)
    plt.savefig(f"./figs/{name}.png", dpi=300)

def plot_input_downsample():
    burgers = Burgers(
        root="data",
        train=True,
        download=True,
        encode_input=True,
        encode_output=True
    )
    print(burgers.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = burgers.data[:1, :]
    target = burgers.targets[:1]
    # error_fn = InputDownSampleError(size=[H2,H2], mode="nearest-exact")
    errors_mean = []
    errors_std = []
    
    name = "Downsample_Burgers"
    errors_mean = []
    errors_std = []
    fields = []
    for ratio in [0.333, 0.4, 0.5, 0.6, 0.7, 0.8, 1]:
        error_fn = TargetDownSampleError(
            size=None,
            scale_factor=ratio,
            mode="linear",
            antialias=False,
            align_corners=True,
        )
        _, target_n = error_fn(data, target)
        err = (
            target_n.sub(target)
            .abs()
            .sum(dim=[2])
            .div(target.abs().sum(dim=[2]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())
        fields.append(target_n)
    fig, axes = plt.subplots(1, 7, figsize=(7, 0.3))
    for i in range(7):
        axes[i].plot(np.arange(target.shape[-1]), fields[i][0, 0].cpu().numpy(), linewidth=0.5)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.savefig(f"./figs/{name}.png", dpi=300)


def plot_input_quant():
    burgers = Burgers(
        root="data",
        train=True,
        download=True,
        encode_input=True,
        encode_output=True
    )
    print(burgers.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = burgers.data[:1, :]
    target = burgers.targets[:1]
    # error_fn = InputDownSampleError(size=[H2,H2], mode="nearest-exact")
    errors_mean = []
    errors_std = []
    
    name = "Quantization_Burgers"
    errors_mean = []
    errors_std = []
    fields = []
    for prec in [
        "fp32",
        "fp16",
        "bfp16",
        "int16",
        "int8",
        "int7",
        "int6",
        "int5",
        "int4",
    ][::-1]:
        error_fn = TargetQuantizationError(prec=prec)
        _, target_n = error_fn(data, target)
        err = (
            target_n.sub(target)
            .abs()
            .sum(dim=[2])
            .div(target.abs().sum(dim=[2]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())
        fields.append(target_n)
    fig, axes = plt.subplots(1, 9, figsize=(9, 0.3))
    for i in range(9):
        axes[i].plot(np.arange(target.shape[-1]), fields[i][0, 0].cpu().numpy(), linewidth=0.5)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.savefig(f"./figs/{name}.png", dpi=300)


def plot_input_gaussian():
    burgers = Burgers(
        root="data",
        train=True,
        download=True,
        encode_input=True,
        encode_output=True
    )
    print(burgers.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = burgers.data[:1, :]
    target = burgers.targets[:1]
    # error_fn = InputDownSampleError(size=[H2,H2], mode="nearest-exact")
    errors_mean = []
    errors_std = []
    
    name = "Gaussian_Burgers"
    errors_mean = []
    errors_std = []
    fields = []
    for std in [0.2, 0.15, 0.11, 0.09, 0.07, 0.05, 0.03, 0]:
        error_fn = TargetGaussianError(std=std)
        _, target_n = error_fn(data, target)
        err = (
            target_n.sub(target)
            .abs()
            .sum(dim=[2])
            .div(target.abs().sum(dim=[2]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())
        fields.append(target_n)
    fig, axes = plt.subplots(1, 8, figsize=(8, 0.3))
    for i in range(8):
        axes[i].plot(np.arange(target.shape[-1]), fields[i][0, 0].cpu().numpy(), linewidth=0.5)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.savefig(f"./figs/{name}.png", dpi=300)


if __name__ == "__main__":
    # test_input_downsample_error()
    # test_input_quant_error()
    # test_input_gaussian_error()
    # test_burgers_input_downsample_error()
    # plot_input_downsample()
    # plot_input_quant()
    plot_input_gaussian()
