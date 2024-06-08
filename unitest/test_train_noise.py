import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullFormatter
from pyutils.plot import set_axes_size_ratio, batch_plot, set_ms
from pyutils.torch_train import set_torch_deterministic
from sklearn.manifold import TSNE
from torch import Tensor
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

from core.datasets.mmi import (
    MMI,
    InputDownSampleError,
    TargetDownSampleError,
    InputGaussianError,
    TargetGaussianError,
    InputQuantizationError,
    TargetQuantizationError,
)


def test_input_downsample_error():
    mmi = MMI(
        root="data",
        train=True,
        pol_list=["rHz_0", "rHz_1", "rHz_2", "rHz_3", "rHz_4"],
        download=True,
        processed_dir="random_size5",
    )
    print(mmi.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = mmi.data[:, :].to(torch.cfloat)
    target = mmi.targets[:].to(torch.cfloat)
    # error_fn = InputDownSampleError(size=[H2,H2], mode="nearest-exact")
    errors_mean = []
    errors_std = []
    for ratio in np.linspace(0.333, 1, 20):
        error_fn = InputDownSampleError(
            size=None,
            scale_factor=ratio,
            mode="bicubic",
            antialias=False,
            align_corners=True,
        )
        data_n, _ = error_fn(data, target)
        err = (
            data_n.sub(data).abs().sum(dim=[2, 3, 4]).div(data.abs().sum(dim=[2, 3, 4]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())
    name = "test_input_downsample_error_Scan_Maxwell"
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
        xrange=[0.25, 1.01, 0.1],
        yrange=[0.0, 1.01, 0.2],
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
            mode="bicubic",
            antialias=False,
            align_corners=True,
        )
        _, target_n = error_fn(data, target)
        err = (
            target_n.sub(target)
            .abs()
            .sum(dim=[2, 3, 4])
            .div(target.abs().sum(dim=[2, 3, 4]))
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
        xrange=[0.25, 1.01, 0.1],
        yrange=[0.0, 1.01, 0.2],
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
    mmi = MMI(
        root="data",
        train=True,
        pol_list=["rHz_0", "rHz_1", "rHz_2", "rHz_3", "rHz_4"],
        download=True,
        processed_dir="random_size5",
    )
    print(mmi.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = mmi.data[:, :].to(torch.cfloat)
    target = mmi.targets[:].to(torch.cfloat)
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
            data_n.sub(data).abs().sum(dim=[2, 3, 4]).div(data.abs().sum(dim=[2, 3, 4]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())
    name = "test_input_quant_error_Scan_Maxwell"
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
            .sum(dim=[2, 3, 4])
            .div(target.abs().sum(dim=[2, 3, 4]))
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
    mmi = MMI(
        root="data",
        train=True,
        pol_list=["rHz_0", "rHz_1", "rHz_2", "rHz_3", "rHz_4"],
        download=True,
        processed_dir="random_size5",
    )
    print(mmi.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = mmi.data[:, :].to(torch.cfloat)
    target = mmi.targets[:].to(torch.cfloat)
    # error_fn = InputDownSampleError(size=[H2,H2], mode="nearest-exact")
    errors_mean = []
    errors_std = []
    for std in np.linspace(0.2, 2, 10):
        error_fn = InputGaussianError(std=std)
        data_n, _ = error_fn(data, target)
        err = (
            data_n.sub(data).abs().sum(dim=[2, 3, 4]).div(data.abs().sum(dim=[2, 3, 4]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())
    name = "test_input_gaussian_error_Scan_Maxwell"
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
            .sum(dim=[2, 3, 4])
            .div(target.abs().sum(dim=[2, 3, 4]))
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



def normalize(x):
    if isinstance(x, np.ndarray):
        x_min, x_max = np.percentile(x, 5), np.percentile(x, 95)
        x = np.clip(x, x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    else:
        x_min, x_max = torch.quantile(x, 0.05), torch.quantile(x, 0.95)
        x = x.clamp(x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    return x



def plot_compare(
    epsilon: Tensor,
    target_fields: Tensor,
    filepath: str,
    pol: str = "Hz",
    norm: bool = True,
) -> None:
    if epsilon.is_complex():
        epsilon = epsilon.real
    # simulation = Simulation(omega, eps_r=epsilon.data.cpu().numpy(), dl=grid_step, NPML=[1,1], pol='Hz')
    target_field_val = target_fields.data.cpu().numpy()
    # eps_r = simulation.eps_r
    eps_r = epsilon.data.cpu().numpy()
    # field_val = np.abs(field_val)
    # target_field_val = np.abs(target_field_val)
    target_field_val = target_field_val.real
    outline_val = np.abs(eps_r)

    # vmax = field_val.max()
    vmin = 0.0
    b = target_field_val.shape[0]
    fig, axes = plt.subplots(1, b, constrained_layout=True, figsize=(5 * b, 3.1))
    if b == 1:
        axes = axes[..., np.newaxis]
    cmap = "RdBu"
    # print(field_val.shape, target_field_val.shape, outline_val.shape)
    for i in range(b):
        vmax = np.max(target_field_val[i])
        if norm:
            h1 = axes[i].imshow(normalize(target_field_val[i]), cmap=cmap, origin="lower")
        else:
            h1 = axes[i].imshow(target_field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")

        # fig.colorbar(h2, label=pol, ax=axes[1,i])
        # fig.colorbar(h3, label=pol, ax=axes[2,i])

    # Do black and white so we can see on both magma and RdBu
    for ax in axes.flatten():
        ax.contour(outline_val[0], levels=2, linewidths=1.0, colors="w")
        ax.contour(outline_val[0], levels=2, linewidths=0.5, colors="k")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filepath, dpi=150)
    plt.close()

def plot_input_downsample():
    mmi = MMI(
        root="data",
        train=True,
        pol_list=["rHz_0", "rHz_1", "rHz_2", "rHz_3", "rHz_4"],
        download=True,
        processed_dir="random_size5",
    )
    print(mmi.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = mmi.data[:1, :1].to(torch.cfloat)
    target = mmi.targets[:1, :1].to(torch.cfloat)
    # error_fn = InputDownSampleError(size=[H2,H2], mode="nearest-exact")
    errors_mean = []
    errors_std = []
    name = "Downsample_Maxwell"
    fields = []
    for ratio in [0.333, 0.4, 0.5, 0.6, 0.7, 0.8, 1]:
        error_fn = TargetDownSampleError(
            size=None,
            scale_factor=ratio,
            mode="bicubic",
            antialias=False,
            align_corners=True,
        )
        _, target_n = error_fn(data, target)
        fields.append(target_n)
        err = (
            target_n.sub(target).abs().sum(dim=[2, 3, 4]).div(target.abs().sum(dim=[2, 3, 4]))
        )
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())
    fields = torch.cat(fields, dim=0)
    plot_compare(data[:,0, 0], fields[:,0, 0], filepath=f"./figs/{name}.png")
        # plt.imshow(target_n[0, 0, 0].real.cpu().numpy(), cmap="RdBu")
        # plt.savefig(f"./figs/{name}_{ratio}.png", dpi=300)

    # print(data, data_n)
    # print(data.shape, data_n.shape)
    # fig, axes = plt.subplots(1,2)
    # axes[0].imshow(data[0,0,0].real.cpu().numpy())
    # axes[1].imshow(data_n[0,0,0].real.cpu().numpy())
    # fig.savefig("./unitest/test_input_downsample_error.png")


def plot_input_quant():
    mmi = MMI(
        root="data",
        train=True,
        pol_list=["rHz_0", "rHz_1", "rHz_2", "rHz_3", "rHz_4"],
        download=True,
        processed_dir="random_size5",
    )
    print(mmi.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = mmi.data[:1, :1].to(torch.cfloat)
    target = mmi.targets[:1, :1].to(torch.cfloat)
    # error_fn = InputDownSampleError(size=[H2,H2], mode="nearest-exact")
    errors_mean = []
    errors_std = []
    errors_mean = []
    errors_std = []
    fields = []
    name = "Quantization_Maxwell"
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
            .sum(dim=[2, 3, 4])
            .div(target.abs().sum(dim=[2, 3, 4]))
        )
        fields.append(target_n)
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())

    errors_mean = np.array(errors_mean)
    errors_std = np.array(errors_std)
    
    fields = torch.cat(fields, dim=0)
    plot_compare(data[:,0, 0], fields[:,0, 0], filepath=f"./figs/{name}.png")


def plot_input_gaussian():

    mmi = MMI(
        root="data",
        train=True,
        pol_list=["rHz_0", "rHz_1", "rHz_2", "rHz_3", "rHz_4"],
        download=True,
        processed_dir="random_size5",
    )
    print(mmi.targets.shape)
    # data = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    # target = torch.randn(1, 1, 1, H, H, dtype=torch.cfloat)
    data = mmi.data[:1, :1].to(torch.cfloat)
    target = mmi.targets[:1, :1].to(torch.cfloat)
    # error_fn = InputDownSampleError(size=[H2,H2], mode="nearest-exact")
    errors_mean = []
    errors_std = []
    errors_mean = []
    errors_std = []
    fields = []
    name = "Gaussian_Maxwell"
    for std in [0.2, 0.15, 0.11, 0.09, 0.07, 0.05, 0.03, 0]:
        error_fn = TargetGaussianError(std=std)
        _, target_n = error_fn(data, target)
        err = (
            target_n.sub(target)
            .abs()
            .sum(dim=[2, 3, 4])
            .div(target.abs().sum(dim=[2, 3, 4]))
        )
        fields.append(target_n)
        err_std, err_mean = torch.std_mean(err.flatten())
        print(err_std, err_mean)
        errors_mean.append(err_mean.item())
        errors_std.append(err_std.item())

    errors_mean = np.array(errors_mean)
    errors_std = np.array(errors_std)
    
    fields = torch.cat(fields, dim=0)
    plot_compare(data[:,0, 0], fields[:,0, 0], filepath=f"./figs/{name}.png")



if __name__ == "__main__":
    # test_input_downsample_error()
    # test_input_quant_error()
    # test_input_gaussian_error()
    # test_burgers_input_downsample_error()
    # plot_input_downsample()
    # plot_input_quant()
    plot_input_gaussian()
