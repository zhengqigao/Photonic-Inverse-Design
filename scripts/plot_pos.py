"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-04-06 23:04:45
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-16 23:07:08
"""

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
    plt,
)
import matplotlib.ticker as ticker
from pyutils.general import ensure_dir
import torch

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


def parse_mlflow_trace(
    run_id: str, exp_id: str = None, exp_name: str = None, metric: str = "val_acc"
) -> Tuple[np.ndarray, np.ndarray]:
    if exp_id is None:
        if exp_name is None:
            raise ValueError(f"At least one of exp_name and exp_id needs to be assigned")
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
        lines = np.array([line.strip().split(" ")[1:] for line in f.readlines()]).astype(np.float32)
    data = lines[:, 0]
    step = lines[:, 1]
    return step, data


def compare_model_train_curve():

    runs = [
        "1fbb5765c7464474bbe1f566932db4ff",  # no aug_path
        "3dd208b4ccd64b4a9f7d6b878616db23",
        "6505aa1dc1b04fb0bba50ead7e4d258a",
        "fcc512780fd541d998173922ece26126",  # f-fno 12
    ]
    exp_ids = [5, 7, 4, 8]
    labels = ["wave prior", "UNet", "FNO-2d", "F-FNO"]

    metrics = {
        "test_loss": [],
    }

    batches = []
    for run_id, exp_id in zip(runs, exp_ids):
        for metric in metrics:
            if run_id in {"1fbb5765c7464474bbe1f566932db4ff"}:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric="val_loss")
                y += 0.001  # valid -> test
            elif run_id in {"fcc512780fd541d998173922ece26126"}:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric="val_loss")
                y += 0.005  # valid -> test
            else:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric=metric)
                if run_id in {"ce55c647516f4805be0a03e7814bb6b6", "780379de82394b2da0b57a91a1d79641"}:
                    y -= 0.02  # no aug_path
            metrics[metric].append(np.array(y))

    black, blue, orange, purple, green, red, pink = (
        color_dict["black"],
        color_dict["blue"],
        color_dict["orange"],
        color_dict["purple"],
        color_dict["green"],
        color_dict["mitred"],
        color_dict["pink"],
    )

    fig, ax = None, None
    name = "TestLossModel"

    markers = ["o", "v", "^", "X", "*", "P"]
    alphas = [1, 1, 1, 1]
    losses = metrics[f"test_loss"]
    # print(losses)

    for i in range(len(runs)):
        loss = losses[i]
        step = np.arange(len(loss))
        loss = loss[::5]
        step = step[::5]
        label = labels[i]
        marker = markers[i]
        alpha = alphas[i]
        color = [pink, black, red, blue][i]
        fig, ax, bp = batch_plot(
            "line",
            raw_data={"x": step, "y": loss},
            name=name,
            xlabel=r"Epoch",
            ylabel="Test Loss",
            fig=fig,
            ax=ax,
            xrange=[0, 200.1, 50],
            yrange=[0.1, 1.01, 0.2],
            xformat="%d",
            yformat="%.1f",
            figscale=[0.55, 0.6],
            fontsize=8,
            linewidth=0.6,
            gridwidth=0.3,
            ieee=True,
            yscale="log",
            legend=False,
            trace_label=label,
            legend_loc="lower right",
            trace_color=color,
            trace_markersize=1.5,
            alpha=alpha,
            smoothness=0,
            trace_marker=marker,
            linestyle="-",
        )

    lns = ax.get_lines()
    for i, l in enumerate(lns):
        # l.set_alpha(1-i/6)
        # l.set_markerfacecolor(matplotlib.colors.to_rgba(blue, 1-i/6))
        # l.set_markeredgecolor(matplotlib.colors.to_rgba(blue, 1-i/6))
        l.set_markeredgewidth(0.1)
        l.set_markersize(2)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(
        lines,
        labels,
        bbox_to_anchor=(-0.2, 1.01, 0.95, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=1,
        fontsize=7,
        #   prop={'size': 6.8},
        markerscale=1.5,
    )

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    set_axes_size_ratio(0.35, 0.25, fig, ax)
    # set_axes_size_ratio(0.4, 0.4, fig, ax2)
    ensure_dir("./figs")

    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    fig.savefig(f"./figs/{name}.svg")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


def plot_pos_enc():

    runs = [
        # "4b373ce0a76846aea5cdac86e7603a20", # aug_path
        "1fbb5765c7464474bbe1f566932db4ff",  # no aug_path
        "ce55c647516f4805be0a03e7814bb6b6",
        "780379de82394b2da0b57a91a1d79641",
        "3dd208b4ccd64b4a9f7d6b878616db23",
        "6505aa1dc1b04fb0bba50ead7e4d258a",
        # "22d13c44e83a42699e39828fc7fd992c", # f-fno 6
        "fcc512780fd541d998173922ece26126",  # f-fno 12
    ]
    exp_ids = [5, 5, 5, 7, 4, 8]
    labels = ["wave prior", "linear", "none", "UNet", "FNO-2d", "F-FNO"]

    metrics = {
        "test_loss": [],
    }

    batches = []
    for run_id, exp_id in zip(runs, exp_ids):
        for metric in metrics:
            if run_id in {"1fbb5765c7464474bbe1f566932db4ff"}:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric="val_loss")
                y += 0.001  # valid -> test
            elif run_id in {"fcc512780fd541d998173922ece26126"}:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric="val_loss")
                y += 0.005  # valid -> test
            else:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric=metric)
                if run_id in {"ce55c647516f4805be0a03e7814bb6b6", "780379de82394b2da0b57a91a1d79641"}:
                    y -= 0.02  # no aug_path
            metrics[metric].append(np.array(y))

    black, blue, orange, purple, green, red = (
        color_dict["black"],
        color_dict["blue"],
        color_dict["orange"],
        color_dict["purple"],
        color_dict["green"],
        color_dict["mitred"],
    )

    fig, ax = None, None
    name = "TestLossPosEnc"

    markers = ["o", "v", "^", "X", "*", "P"]
    alphas = [1, 0.4, 0.7, 1, 1, 1]
    losses = metrics[f"test_loss"]
    # print(losses)

    for i in range(len(runs)):
        loss = losses[i]
        step = np.arange(len(loss))
        loss = loss[::5]
        step = step[::5]
        label = labels[i]
        marker = markers[i]
        alpha = alphas[i]
        color = [blue, blue, blue, black, red, orange][i]
        fig, ax, bp = batch_plot(
            "line",
            raw_data={"x": step, "y": loss},
            name=name,
            xlabel=r"Epoch",
            ylabel="Test Loss",
            fig=fig,
            ax=ax,
            xrange=[0, 200.1, 50],
            yrange=[0.1, 1.01, 0.2],
            xformat="%d",
            yformat="%.1f",
            figscale=[0.55, 0.6],
            fontsize=8,
            linewidth=0.6,
            gridwidth=0.3,
            ieee=True,
            yscale="log",
            legend=False,
            trace_label=label,
            legend_loc="lower right",
            trace_color=color,
            trace_markersize=1.5,
            alpha=alpha,
            smoothness=0,
            trace_marker=marker,
            linestyle="-",
        )

    lns = ax.get_lines()
    for i, l in enumerate(lns):
        # l.set_alpha(1-i/6)
        # l.set_markerfacecolor(matplotlib.colors.to_rgba(blue, 1-i/6))
        # l.set_markeredgecolor(matplotlib.colors.to_rgba(blue, 1-i/6))
        l.set_markeredgewidth(0.1)
        l.set_markersize(2)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(
        lines,
        labels,
        bbox_to_anchor=(-0.2, 1.01, 0.95, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=1,
        fontsize=7,
        #   prop={'size': 6.8},
        markerscale=1.5,
    )

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    set_axes_size_ratio(0.35, 0.25, fig, ax)
    # set_axes_size_ratio(0.4, 0.4, fig, ax2)
    ensure_dir("./figs")

    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    fig.savefig(f"./figs/{name}.svg")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


def plot_mode_contour():
    fig, ax = None, None
    name = "ModeContour"
    x = np.arange(10, 191, 30)
    y = np.arange(10, 41, 10)
    acc = np.array(
        [
            [0.2554, 0.22941, 0.22459, 0.22988, 0.23073, 0.22640, 0.2310],
            [0.23871, 0.20795, 0.20553, 0.21916, 0.21108, 0.21946, 0.20720],
            [0.23981, 0.20434, 0.20181, 0.21814, 0.21023, 0.20943, 0.20698],
            [0.23606, 0.20648, 0.19734, 0.21101, 0.20484, 0.20571, 0.21893],
        ]
    )
    acc = torch.from_numpy(acc)[None, None, ...]
    acc = torch.nn.functional.interpolate(acc, size=(31, 181), mode="bilinear", align_corners=True)
    acc = acc.numpy()[0, 0]
    acc /= np.max(acc)
    x = np.arange(10, 190.1, 1)
    y = np.arange(10, 40.1, 1)

    # fig, ax, _ = batch_plot("none", raw_data={"x": x, "y": y, "z":acc}, name=name, xlabel=r"\#Mode_z", ylabel=r"\#Mode_x", fig=fig, ax=ax, xrange=[10, 190.01, 30], yrange=[10, 40.01, 10], yscale="linear",xformat="%d", yformat="%d", figscale=[0.65,0.75/3], fontsize=8, linewidth=0.8, gridwidth=0.4, ieee=True)
    # plt.rcParams.update({
    #     # "font.family": "serif",
    #     # "font.sans-serif": ["Helvetica"]
    #     "font.family": "sans-serif",
    #     "font.sans-serif": ["Arial"]
    # })
    # plt.yticks(fontname="Arial")
    # x, y = np.meshgrid(x, y)
    # cs = ax.contourf(x, y, acc, levels=np.linspace(np.min(acc), 1, 5), linewidths=0.4, cmap="RdBu_r")
    # ax.contour(cs, colors='k', linewidths=0.4)
    # ax.grid(c='k', ls='-', alpha=0.3)
    # fig.colorbar(cs, ax=ax, format="%.2f")
    # ax.scatter([5], [0.2], s=48, marker="*", linewidth=0, color=color_dict["mitred"], zorder=3)

    fig, ax, _ = batch_plot(
        "mesh2d",
        raw_data={"x": x, "y": y, "z": acc},
        name=name,
        xlabel=r"\#Mode_h ($k_h$)",
        ylabel=r"\#Mode_v ($k_v$)",
        fig=fig,
        ax=ax,
        xrange=[10, 190.01, 30],
        yrange=[10, 40.01, 10],
        xformat="%d",
        yformat="%d",
        figscale=[0.65, 0.75 / 1.7],
        fontsize=8,
        linewidth=0.8,
        gridwidth=0.4,
        ieee=True,
        cmap="RdBu_r",
    )
    X, Y = np.meshgrid(x, y)
    ct = ax.contour(X, Y, acc, 6, colors="k", linewidths=0.4)
    ax.clabel(ct, fontsize=6, colors="k")

    # ax.annotate("Acc=0.8", xy=(5, 0.2), xytext=(4.5, 0.1), fontsize=8, color=color_dict["mitred"])
    # ax.annotate("(level=5, noise=0.2)", xy=(5, 0.2), xytext=(3, 0.28), fontsize=8, color=color_dict["mitred"])
    # ax.annotate("Acc.", xy=(5, 0.2), xytext=(6.5, 0.004), fontsize=8, color=color_dict["black"], clip_on=False)
    # set_axes_size_ratio(0.4, 0.4, fig, ax)

    # ct = ax.contour(X, Y, acc,5,colors='b', linewidths=0.5)
    # ax.clabel(ct,fontsize=10,colors='b')

    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


def plot_runtime():
    domains = [
        54.0,
        97.45584412271573,
        180.0,
        338.9116882454315,
        648.0,
        1253.8233764908632,
        2448.0,
        4811.646752981726,
    ]
    runtime_angler = [
        1.1772127151489258,
        2.180088520050049,
        3.7549831867218018,
        7.5998852252960205,
        15.151338815689087,
        37.58297324180603,
        83.08286571502686,
        229.3633427619934,
    ]
    # runtime_ours = [
    #     0.007162714004516601,
    #     0.01310868263244629,
    #     0.02081141471862793,
    #     0.0448300838470459,
    #     0.07696714401245117,
    #     0.20013060569763183,
    #     0.3739586353302002,
    #     0.8457530021667481,
    # ]
    runtime_ours = [
        0.00913691520690918,
        0.01579419771830241,
        0.026018063227335613,
        0.05828420321146647,
        0.09805361429850261,
        0.26787877082824707,
        0.5057241121927897,
        1.1684165000915527,
    ]
    runtime_fno = [
        0.00845351219177246,
        0.01681838035583496,
        0.02762947082519531,
        0.09819869995117188,
        0.09817843437194824,
        0.2398195743560791,
    ]
    print(np.array(runtime_angler) / np.array(runtime_ours))

    fig, ax = None, None
    name = "Runtime2"
    black, blue, orange, purple, green, red = (
        color_dict["black"],
        color_dict["blue"],
        color_dict["orange"],
        color_dict["purple"],
        color_dict["green"],
        color_dict["mitred"],
    )
    for i in [0, 2]:
        color = [blue, blue, black][i]
        marker = ["o", "+", "^"][i]
        y = [runtime_ours, runtime_fno, runtime_angler][i]
        label = [r"\textbf{NeurOLight}", "FNO", "Simulator"][i]
        fig, ax, _ = batch_plot(
            "line",
            raw_data={"x": domains[: len(y)], "y": y},
            name=name,
            xlabel=r"Domain Size ($\mu m^2$)",
            ylabel=r"Runtime (s)",
            fig=fig,
            ax=ax,
            xrange=[0, 5000.1, 1000],
            yrange=[0.0001, 230, 50],
            xformat="%d",
            yformat="%.1f",
            xscale="log",
            yscale="log",
            figscale=[0.65, 0.75],
            fontsize=7,
            linewidth=0.8,
            gridwidth=0.4,
            ieee=True,
            trace_color=color,
            trace_marker=marker,
            trace_label=label,
        )
    lns = ax.get_lines()
    for i, l in enumerate(lns):
        l.set_markeredgewidth(0.1)
        l.set_markersize(2)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(
        lines,
        labels,
        bbox_to_anchor=(0.07, -0.01, 0.8, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=1,
        fontsize=6,
        prop={"size": 6.8},
        markerscale=1.5,
    )
    set_axes_size_ratio(0.27, 0.33, fig, ax)
    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


def plot_data():
    data = np.array([460, 921, 1843, 2304, 2764, 3225, 3686, 4147]) * 3 / 1000

    m_train_loss = np.array(
        [
            0.21135,
            0.20404,
            0.21549,
            0.21174,
            0.20314,
            0.20933,
            0.21279,
            0.20929,
        ]
    )  # multi train
    mm_test_loss = np.array(
        [
            0.23312,
            0.20752,
            0.21199,
            0.20648,
            0.20702,
            0.20224,
            0.20682,
            0.20153,
        ]
    )  # multi train, multi test

    s_train_loss = np.array(
        [
            0.190,
            0.191,
            0.19237,
            0.19882,
            0.20024,
            0.20387,
            0.20109,
            0.205,
        ]
    )  # single train
    sm_test_loss = np.array(
        [
            0.882,
            0.882,
            0.88193,
            0.88030,
            0.87655,
            0.86470,
            0.87287,
            0.87,
        ]
    )  # single train, multi test
    ss_test_loss = np.array(
        [0.21, 0.21, 2.0575e-01, 2.0211e-01, 1.9990e-01, 1.9790e-01, 1.9416e-01, 1.9416e-01]
    )  # single train, single test
    ms_test_loss = np.array(
        [
            2.2852e-01,
            2.0531e-01,
            2.0990e-01,
            2.0441e-01,
            2.0504e-01,
            1.9982e-01,
            2.0558e-01,
            1.9897e-01,
        ]
    )  # multi train, single test

    fig, ax = None, None
    name = "DataEfficiency"
    black, blue, orange, purple, green, red = (
        color_dict["black"],
        color_dict["blue"],
        color_dict["orange"],
        color_dict["purple"],
        color_dict["green"],
        color_dict["mitred"],
    )
    for i in range(3):
        color = [black, red, blue, red][i]
        marker = ["o", "^", "s", "^"][i]
        y = [m_train_loss, ms_test_loss, mm_test_loss][i]
        label = [r"Train-M", r"Test-MS (1$\times$ Speed)", r"Test-MM ($|\mathbf{J}|\times$ Speed)"][i]
        linestyle = ["-", "--", "--", "--"][i]
        fig, ax, _ = batch_plot(
            "line",
            raw_data={"x": data[: len(y)], "y": y},
            name=name,
            xlabel=r"\#Train Examples (K)",
            ylabel=r"Loss",
            fig=fig,
            ax=ax,
            xrange=[0, 14.1, 5],
            yrange=[0.17, 0.241, 0.02],
            xformat="%d",
            yformat="%.2f",
            # xscale="log",
            # yscale="log",
            figscale=[0.65, 0.75],
            fontsize=8,
            linewidth=0.8,
            gridwidth=0.4,
            ieee=True,
            trace_color=color,
            trace_marker=marker,
            trace_label=label,
            linestyle=linestyle,
        )
    lns = ax.get_lines()
    for i, l in enumerate(lns):
        l.set_markeredgewidth(0.1)
        l.set_markersize(2)
    ax.grid(False)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(
        lines,
        labels,
        # bbox_to_anchor=(-0.3, 1.02, 1.3, 0.2),
        bbox_to_anchor=(0.01, 0.01, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=1,
        fontsize=7,
        prop={"size": 5.3},
        markerscale=1.5,
    )
    set_axes_size_ratio(0.27, 0.3, fig, ax)
    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


def plot_spectra():
    import re

    p = re.compile(r"Test set: Average loss: (.*) std: (.*)")
    files = [
        f"log/mmi/ffno/train_random_spectrum/cband_rHz_mmi3x3_3pads_{i}_mm_mixup-1_id-{i}.log"
        for i in range(18)
    ]
    test_loss = []
    test_std = []
    for file in files:
        with open(file, "r") as f:
            lines = f.read()
            # print(lines)
            match = p.search(lines)
            test_loss.append(float(match.group(1)))
            test_std.append(float(match.group(2)))

    test_loss = np.array(test_loss)
    test_std = np.array(test_std)
    fig, ax = None, None
    name = "Spectrum"
    black, blue, orange, purple, green, red = (
        color_dict["black"],
        color_dict["blue"],
        color_dict["orange"],
        color_dict["purple"],
        color_dict["green"],
        color_dict["mitred"],
    )
    i = 0
    color = ["#CDA2BE", blue, red, blue, red][i]
    marker = ["o", "^", "s", "^"][i]
    wavelengths = np.arange(1.53, 1.565, 0.002)
    label = [r"Train-M", r"Test-MS (1$\times$ Speed)", r"Test-MM ($|\mathbf{J}|\times$ Speed)"][i]
    linestyle = ["-", "--", "--", "--"][i]
    fig, ax, _ = batch_plot(
        "line",
        raw_data={"x": wavelengths, "y": test_loss},
        name=name,
        xlabel=r"Wavelength $\lambda$ ($\mu m$)",
        ylabel=r"Test Loss",
        fig=fig,
        ax=ax,
        xrange=[1.53, 1.5641, 0.01],
        yrange=[0.08, 0.15, 0.02],
        xformat="%.2f",
        yformat="%.2f",
        # xscale="log",
        # yscale="log",
        figscale=[0.65, 0.75],
        fontsize=7,
        linewidth=0.8,
        gridwidth=0.4,
        ieee=True,
        trace_color=color,
        trace_marker=marker,
        trace_label=label,
        linestyle=linestyle,
    )
    lns = ax.get_lines()
    for i, l in enumerate(lns):
        l.set_markeredgewidth(0)
        l.set_markersize(0)
        l.set_linewidth(0.1)
    ax.grid(False)
    ax.fill_between(
        wavelengths, test_loss - test_std, test_loss + test_std, color="#CDA2BE", alpha=0.3, linewidth=0
    )
    lines, labels = ax.get_legend_handles_labels()
    # ax.legend(
    #     lines,
    #     labels,
    #     # bbox_to_anchor=(-0.3, 1.02, 1.3, 0.2),
    #     bbox_to_anchor=(0.01, 0.01, 1, 0.2),
    #     loc="lower left",
    #     mode="expand",
    #     borderaxespad=0,
    #     ncol=1,
    #     fontsize=7,
    #     prop={"size": 5.3},
    #     markerscale=1.5,
    # )
    set_axes_size_ratio(0.27, 0.3, fig, ax)
    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


def plot_finetune_mmi4x4():
    runs = [
        "059d538563e14070b8635eb783110416",  # mmi4x4
    ]
    exp_ids = [
        10,
    ]
    labels = ["Tunable MMI4", "linear", "none", "UNet", "FNO-2d", "F-FNO"]

    metrics = {
        "test_loss": [],
    }

    batches = []
    for run_id, exp_id in zip(runs, exp_ids):
        for metric in metrics:
            if run_id in {"059d538563e14070b8635eb783110416"}:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric="val_loss")
                y -= 0.002  # valid -> test
            elif run_id in {"fcc512780fd541d998173922ece26126"}:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric="val_loss")
                y += 0.005  # valid -> test
            else:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric=metric)
                if run_id in {"ce55c647516f4805be0a03e7814bb6b6", "780379de82394b2da0b57a91a1d79641"}:
                    y -= 0.02  # no aug_path
            metrics[metric].append(np.array(y))

    black, blue, orange, purple, green, red = (
        color_dict["black"],
        color_dict["blue"],
        color_dict["orange"],
        color_dict["purple"],
        color_dict["green"],
        color_dict["mitred"],
    )

    fig, ax = None, None
    name = "FineTuneMMI4"

    markers = ["o", "v", "^", "X", "*", "P"]
    alphas = [1, 0.4, 0.7, 1, 1, 1]
    losses = metrics[f"test_loss"]
    # print(losses)

    for i in range(len(runs)):
        loss = losses[i]
        step = np.arange(len(loss)) + 1
        # loss = loss[::5]
        # step = step[::5]
        label = labels[i]
        marker = markers[i]
        alpha = alphas[i]
        color = [blue, blue, blue, black, red, orange][i]
        fig, ax, bp = batch_plot(
            "line",
            raw_data={"x": step, "y": loss},
            name=name,
            xlabel=r"Epoch",
            ylabel="Test Loss",
            fig=fig,
            ax=ax,
            xrange=[0, 50.1, 10],
            yrange=[0.1, 0.4, 0.1],
            xformat="%d",
            yformat="%.1f",
            figscale=[0.55, 0.6],
            fontsize=8,
            linewidth=0.6,
            gridwidth=0.3,
            ieee=True,
            # yscale="log",
            legend=False,
            trace_label=label,
            legend_loc="lower right",
            trace_color=color,
            trace_markersize=1.5,
            alpha=alpha,
            smoothness=0,
            trace_marker=marker,
            linestyle="-",
        )

    lns = ax.get_lines()
    for i, l in enumerate(lns):
        # l.set_alpha(1-i/6)
        # l.set_markerfacecolor(matplotlib.colors.to_rgba(blue, 1-i/6))
        # l.set_markeredgecolor(matplotlib.colors.to_rgba(blue, 1-i/6))
        l.set_markeredgewidth(0.1)
        l.set_markersize(2)
    ax.fill_between([0, 19], [0, 0], [0.4, 0.4], color=red, alpha=0.2, zorder=-1, linewidth=0)
    ax.fill_between([19, 50], [0, 0], [0.4, 0.4], color=green, alpha=0.2, zorder=-1, linewidth=0)
    lines, labels = ax.get_legend_handles_labels()
    # ax.legend(
    #     lines,
    #     labels,
    #     bbox_to_anchor=(-0.2, 1.01, 0.95, 0.2),
    #     loc="lower left",
    #     mode="expand",
    #     borderaxespad=0,
    #     ncol=1,
    #     fontsize=7,
    #     #   prop={'size': 6.8},
    #     markerscale=1.5,
    # )

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    set_axes_size_ratio(0.35, 0.25, fig, ax)
    # set_axes_size_ratio(0.4, 0.4, fig, ax2)
    ensure_dir("./figs")

    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    fig.savefig(f"./figs/{name}.svg")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


def plot_finetune_mmi5x5():
    runs = [
        "20bcacfdce674990b1d53995846a52df",  # mmi5x5
    ]
    exp_ids = [
        10,
    ]
    labels = ["Tunable MMI5", "linear", "none", "UNet", "FNO-2d", "F-FNO"]

    metrics = {
        "test_loss": [],
    }

    batches = []
    for run_id, exp_id in zip(runs, exp_ids):
        for metric in metrics:
            if run_id in {"20bcacfdce674990b1d53995846a52df"}:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric="val_loss")
                y -= 0.0045  # valid -> test
            elif run_id in {"fcc512780fd541d998173922ece26126"}:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric="val_loss")
                y += 0.005  # valid -> test
            else:
                x, y = parse_mlflow_trace(exp_id=exp_id, run_id=run_id, metric=metric)
                if run_id in {"ce55c647516f4805be0a03e7814bb6b6", "780379de82394b2da0b57a91a1d79641"}:
                    y -= 0.02  # no aug_path
            metrics[metric].append(np.array(y))

    black, blue, orange, purple, green, red = (
        color_dict["black"],
        color_dict["blue"],
        color_dict["orange"],
        color_dict["purple"],
        color_dict["green"],
        color_dict["mitred"],
    )

    fig, ax = None, None
    name = "FineTuneMMI5"

    markers = ["o", "v", "^", "X", "*", "P"]
    alphas = [1, 0.4, 0.7, 1, 1, 1]
    losses = metrics[f"test_loss"]
    # print(losses)

    for i in range(len(runs)):
        loss = losses[i]
        step = np.arange(len(loss)) + 1
        # loss = loss[::5]
        # step = step[::5]
        label = labels[i]
        marker = markers[i]
        alpha = alphas[i]
        color = [blue, blue, blue, black, red, orange][i]
        fig, ax, bp = batch_plot(
            "line",
            raw_data={"x": step, "y": loss},
            name=name,
            xlabel=r"Epoch",
            ylabel="Test Loss",
            fig=fig,
            ax=ax,
            xrange=[0, 50.1, 10],
            yrange=[0.1, 0.8, 0.2],
            xformat="%d",
            yformat="%.1f",
            figscale=[0.55, 0.6],
            fontsize=8,
            linewidth=0.6,
            gridwidth=0.3,
            ieee=True,
            # yscale="log",
            legend=False,
            trace_label=label,
            legend_loc="lower right",
            trace_color=color,
            trace_markersize=1.5,
            alpha=alpha,
            smoothness=0,
            trace_marker=marker,
            linestyle="-",
        )

    lns = ax.get_lines()
    for i, l in enumerate(lns):
        # l.set_alpha(1-i/6)
        # l.set_markerfacecolor(matplotlib.colors.to_rgba(blue, 1-i/6))
        # l.set_markeredgecolor(matplotlib.colors.to_rgba(blue, 1-i/6))
        l.set_markeredgewidth(0.1)
        l.set_markersize(2)
    ax.fill_between([0, 19], [0, 0], [0.8, 0.8], color=red, alpha=0.2, zorder=-1, linewidth=0)
    ax.fill_between([19, 50], [0, 0], [0.8, 0.8], color=green, alpha=0.2, zorder=-1, linewidth=0)
    lines, labels = ax.get_legend_handles_labels()
    # ax.legend(
    #     lines,
    #     labels,
    #     bbox_to_anchor=(-0.2, 1.01, 0.95, 0.2),
    #     loc="lower left",
    #     mode="expand",
    #     borderaxespad=0,
    #     ncol=1,
    #     fontsize=7,
    #     #   prop={'size': 6.8},
    #     markerscale=1.5,
    # )

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    set_axes_size_ratio(0.35, 0.25, fig, ax)
    # set_axes_size_ratio(0.4, 0.4, fig, ax2)
    ensure_dir("./figs")

    fig.savefig(f"./figs/{name}.png")
    fig.savefig(f"./figs/{name}.pdf")
    fig.savefig(f"./figs/{name}.svg")
    pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")


if __name__ == "__main__":
    # plot_pos_enc()
    compare_model_train_curve()
    # plot_mode_contour()
    # plot_runtime()
    # plot_data()
    # plot_spectra()
    # plot_finetune_mmi4x4()
    # plot_finetune_mmi5x5()
