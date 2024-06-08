from typing import Tuple
import matplotlib
import numpy as np
import os
from pyutils.plot import (
    autolabel,
    batch_plot,
    pdf_crop,
    set_axis_formatter,
    set_axes_size_ratio,
    smooth_line,
    set_ms
)
import re
import matplotlib.ticker as ticker

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
}


def parse_mlflow_trace(
    run_id: str, exp_id: str = None, exp_name: str = None, metric: str = "val_acc"
) -> Tuple[np.ndarray, np.ndarray]:
    if exp_id is None:
        if exp_name is None:
            raise ValueError(
                f"At least one of exp_name and exp_id needs to be assigned"
            )
        exps = [f"./mlruns/{i}" for i in os.listdir(f"./mlruns") if i != ".trash"]
        for exp in exps:
            meta = os.path.join(exp, "meta.yaml")
            with open(meta, "r") as f:
                lines = f.readlines()
            e_name = None
            e_id = None
            for line in lines:
                if line.startswith("experiment_id:"):
                    e_id = line[15:].strip()[1:-1]
                    continue
                elif line.startswith("name:"):
                    e_name = line[6:].strip()
                    break
                else:
                    continue
            if e_name is not None and e_name == exp_name:
                exp_id = e_id
                break
    if exp_id is None:
        raise ValueError(f"experiment {exp_name} does not exist")

    path = f"./mlruns/{exp_id}/{run_id}/metrics/{metric}"

    with open(path, "r") as f:
        lines = np.array(
            [line.strip().split(" ")[1:] for line in f.readlines()]
        ).astype(np.float32)
    data = lines[:, 0]
    step = lines[:, 1]
    return step, data


def plot_gaussian():
    stds = [0.03, 0.05, 0.07, 0.09, 0.11, 0.15, 0.2]
    runs = []
    for std in stds:
        log = f"rHz5_istd-{std}_tstd-{std}_is-1_ts-1_ip-fp32_tp-fp32_id-1.log"
        with open(f"./log/mmi/fno/train_random_noise_grad/{log}", "r") as f:
            lines = f.readlines()
            p = re.compile(r".*Run ID: \((\S+)\).*")
            for line in lines:
                match = p.search(line)
                if match is not None:
                    print(match.group(1))
                    runs.append(match.group(1))
                    break

    # runs = [
    #     "a4e16b894b524bf8a61575ad4a41e46c", # 0.03
    #     "f87a9bdb69a64ef8a3ba15288ef76ff6", # 0.05
    #     "a83fd33b2fd2465aa0d24646ace762bc", # 0.07
    #     "891c64cc16244f199e2561ed3ec276bd", # 0.09
    #     "04a7e29f5e6141c5ab06322fc8792c4c", # 0.11
    #     "7f620da9aacf40da960a8d8b27593610", # 0.15
    #     "1f05eccfa4ff40d78015f31db78f0847", # 0.2
    # ]

    metrics = {"total_avg_arg": [], "total_std_arg": []}

    for run_id in runs:
        for metric in metrics:
            x, y = parse_mlflow_trace(exp_id=615031351664796851, run_id=run_id, metric=metric)
            metrics[metric].append(
                y
            )

    
    black, blue, orange, red = (
        color_dict["black"],
        color_dict["blue"],
        color_dict["orange"],
        color_dict["mitred"],
    )

    fig, ax = None, None
    name = "TrainGradGaussian"
    scale = 1e11

    markers = ["o", "X", "^", "v", "*", "<", "s"]
 
    for i, std in enumerate([0.03, 0.05, 0.07, 0.09, 0.11, 0.15, 0.2]):
        fig, ax, _ = batch_plot(
            "line",
            raw_data={
                "x": x,
                "y": metrics["total_avg_arg"][i],
            },
            name=name,
            xlabel="Epoch",
            ylabel="Angular Similarity",
            fig=fig,
            ax=ax,
            xrange=[0, max(x)+0.1, 1],
            xlimit=[0.5, max(x) + 0.5],
            yrange=[0.75, 1.01, 0.05],
            xformat="%d",
            yformat="%.2f",
            figscale=[0.75, 0.6],
            fontsize=8,
            linewidth=0.6,
            gridwidth=0.3,
            ieee=True,
            legend=False,
            legend_loc="lower right",
            trace_color=blue,
            trace_marker=markers[i],
            alpha=1 - i / (len(runs)+2),
            trace_label=f"std={std}",
            linestyle="-",
            smoothness=0,
        )
        set_ms()
        
        # ax.plot(
        #     x,
        #     metrics["total_avg_arg"][i],
        #     color=blue,
        #     alpha=1 - i / (len(runs) - 0.4),
        #     linestyle="-",
        #     linewidth=0.6,
        #     marker=markers[1],
        # )
    for ln in ax.lines:
        ln.set_markeredgewidth(0)
        ln.set_markersize(2)
    lgd = ax.legend(bbox_to_anchor=(-0.2, 1.02, 1.2, 0.3), loc="lower left",
              mode="expand", borderaxespad=0, ncol=2, fontsize=10, prop={'size': 7}, markerscale=2)
    # lgd = ax.legend(bbox_to_anchor=(1.02, 0.02, 0.5, 1), loc="right",
    #           mode="expand", borderaxespad=0, ncol=1, fontsize=10, prop={'size': 7}, markerscale=2)
    set_axes_size_ratio(0.4, 0.4, fig, ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    # set_axes_size_ratio(0.4, 0.4, fig, ax)
    fig.savefig(f"./figs/{name}.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig(f"./figs/{name}.pdf")
    # fig.savefig(f"./figs/{name}.svg")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


def plot_quantization():
    precs = ["fp16", "bfp16", "int16", "int8", "int7", "int6", "int5", "int4"]
    runs = []
    for prec in precs:
        log = f"rHz5_istd-0_tstd-0_is-1_ts-1_ip-{prec}_tp-{prec}_id-1.log"
        with open(f"./log/mmi/fno/train_random_noise_grad/{log}", "r") as f:
            lines = f.readlines()
            p = re.compile(r".*Run ID: \((\S+)\).*")
            for line in lines:
                match = p.search(line)
                if match is not None:
                    print(match.group(1))
                    runs.append(match.group(1))
                    break
    # runs = [
    #     "d203e4fd612f4e41b9444e6ce78873ff", # fp16
    #     "a0064db19d60403198d5b07d06bb20b8", # bfp16
    #     "8bc37c52419445a9802ef3ccf624ced2", # int16
    #     "4ed1989fc2074d529be0b10d834f0466", # int8
    #     "e1d8e3bbefcc42f6848699cbca95cea3", # int7
    #     "fe31c35595d04c9396f5fdd8dc869ed9", # int6
    #     "90f99452775c40afa2fa76ecf2e53624", # int5
    #     "3802209062b54dabb378e81848f5283c", # int4
    # ]

    metrics = {"total_avg_arg": [], "total_std_arg": []}

    for run_id in runs:
        for metric in metrics:
            x, y = parse_mlflow_trace(exp_id=615031351664796851, run_id=run_id, metric=metric)
            y = np.nan_to_num(y, nan=1)
            metrics[metric].append(
                y
            )
    
    black, blue, orange, red = (
        color_dict["black"],
        color_dict["blue"],
        color_dict["orange"],
        color_dict["mitred"],
    )

    fig, ax = None, None
    name = "TrainGradQuantization"
    scale = 1e11

    markers = ["o", "X", "^", "v", "*", "<", "s", ">"]
 
    for i, prec in enumerate(["fp16", "bfp16", "int16", "int8", "int7", "int6", "int5", "int4"]):
        fig, ax, _ = batch_plot(
            "line",
            raw_data={
                "x": x,
                "y": metrics["total_avg_arg"][i],
            },
            name=name,
            xlabel="Epoch",
            ylabel="Angular Similarity",
            fig=fig,
            ax=ax,
            xrange=[0, max(x)+0.1, 1],
            xlimit=[0.5, max(x) + 0.5],
            yrange=[0.5, 1.01, 0.1],
            xformat="%d",
            yformat="%.2f",
            figscale=[0.75, 0.6],
            fontsize=8,
            linewidth=0.6,
            gridwidth=0.3,
            ieee=True,
            legend=False,
            legend_loc="lower right",
            trace_color=blue,
            trace_marker=markers[i],
            alpha=1 - i / (len(runs)+2),
            trace_label=f"{prec}",
            linestyle="-",
            smoothness=0,
        )
        set_ms()
        
        # ax.plot(
        #     x,
        #     metrics["total_avg_arg"][i],
        #     color=blue,
        #     alpha=1 - i / (len(runs) - 0.4),
        #     linestyle="-",
        #     linewidth=0.6,
        #     marker=markers[1],
        # )
    for ln in ax.lines:
        ln.set_markeredgewidth(0)
        ln.set_markersize(2)
    lgd = ax.legend(bbox_to_anchor=(-0.2, 1.02, 1.2, 0.3), loc="lower left",
              mode="expand", borderaxespad=0, ncol=2, fontsize=10, prop={'size': 7}, markerscale=2)
    # lgd = ax.legend(bbox_to_anchor=(1.02, 0.02, 0.4, 1), loc="right",
    #           mode="expand", borderaxespad=0, ncol=1, fontsize=10, prop={'size': 7}, markerscale=2)
    set_axes_size_ratio(0.4, 0.4, fig, ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    # set_axes_size_ratio(0.4, 0.4, fig, ax)
    fig.savefig(f"./figs/{name}.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig(f"./figs/{name}.pdf")
    # fig.savefig(f"./figs/{name}.svg")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


def plot_downsample():
    factors = [0.8, 0.7, 0.6, 0.5, 0.4, "0.333"]
    runs = []
    for factor in factors:
        log = f"rHz5_istd-0_tstd-0_is-{factor}_ts-{factor}_ip-fp32_tp-fp32_id-1.log"
        with open(f"./log/mmi/fno/train_random_noise_grad/{log}", "r") as f:
            lines = f.readlines()
            p = re.compile(r".*Run ID: \((\S+)\).*")
            for line in lines:
                match = p.search(line)
                if match is not None:
                    print(match.group(1))
                    runs.append(match.group(1))
                    break

    # runs = [
    #     "872355f4546745d0a412fcc2fd8f867a", # 0.8
    #     "809cdf59c09245649d073f5932546a05", # 0.7
    #     "34dc2d91293741248284fbd8698d611d", # 0.6
    #     "a291db9b3f684327b8d6ef4d39a84ad8", # 0.5
    #     "c8a71fa429b84965a0a678812bfec27d", # 0.4
    #     "bc74bd2224ea47118c3570abf9386588", # 0.333
    # ]

    metrics = {"total_avg_arg": [], "total_std_arg": []}

    for run_id in runs:
        for metric in metrics:
            x, y = parse_mlflow_trace(exp_id=615031351664796851, run_id=run_id, metric=metric)
            metrics[metric].append(
                y
            )

    
    black, blue, orange, red = (
        color_dict["black"],
        color_dict["blue"],
        color_dict["orange"],
        color_dict["mitred"],
    )

    fig, ax = None, None
    name = "TrainGradDownsample"
    scale = 1e11

    markers = ["o", "X", "^", "v", "*", "<", "s"]
 
    for i, std in enumerate([0.8, 0.7, 0.6, 0.5, 0.4, "1/3"]):
        fig, ax, _ = batch_plot(
            "line",
            raw_data={
                "x": x,
                "y": metrics["total_avg_arg"][i],
            },
            name=name,
            xlabel="Epoch",
            ylabel="Angular Similarity",
            fig=fig,
            ax=ax,
            xrange=[0, max(x)+0.1, 1],
            xlimit=[0.5, max(x) + 0.5],
            yrange=[0.5, 1.01, 0.1],
            xformat="%d",
            yformat="%.1f",
            figscale=[0.75, 0.6],
            fontsize=8,
            linewidth=0.6,
            gridwidth=0.3,
            ieee=True,
            legend=False,
            legend_loc="lower right",
            trace_color=blue,
            trace_marker=markers[i],
            alpha=1 - i / (len(runs)+2),
            trace_label=f"s={std}",
            linestyle="-",
            smoothness=0,
        )
        set_ms()
        
        # ax.plot(
        #     x,
        #     metrics["total_avg_arg"][i],
        #     color=blue,
        #     alpha=1 - i / (len(runs) - 0.4),
        #     linestyle="-",
        #     linewidth=0.6,
        #     marker=markers[1],
        # )
    for ln in ax.lines:
        ln.set_markeredgewidth(0)
        ln.set_markersize(2)
    lgd = ax.legend(bbox_to_anchor=(-0.2, 1.02, 1.2, 0.3), loc="lower left",
              mode="expand", borderaxespad=0, ncol=2, fontsize=10, prop={'size': 7}, markerscale=2)
    # lgd = ax.legend(bbox_to_anchor=(1.02, 0.02, 0.4, 1), loc="right",
    #           mode="expand", borderaxespad=0, ncol=1, fontsize=10, prop={'size': 7}, markerscale=2)
    set_axes_size_ratio(0.4, 0.4, fig, ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    # set_axes_size_ratio(0.4, 0.4, fig, ax)
    fig.savefig(f"./figs/{name}.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig(f"./figs/{name}.pdf")
    # fig.savefig(f"./figs/{name}.svg")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


# plot_gaussian()
# plot_quantization()
# plot_downsample()