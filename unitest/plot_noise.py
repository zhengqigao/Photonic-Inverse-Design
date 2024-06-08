'''
Author: JeremieMelo jqgu@utexas.edu
Date: 2023-09-16 13:56:19
LastEditors: JeremieMelo jqgu@utexas.edu
LastEditTime: 2023-09-16 14:41:34
FilePath: /NeurOLight_Local/unitest/plot_noise.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from sklearn.manifold import TSNE
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from pyutils.plot import set_axes_size_ratio
from pyutils.torch_train import set_torch_deterministic
from core.datasets.mmi import MMIDataset
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

def plot_error():
    set_torch_deterministic(0)
    train_data_path = "./data"
    train_noise=dict(
            input_gaussian={"mean": 0.0, "std": 0},
            target_gaussian={"mean": 0.0, "std": 0},
            input_downsample={"scale_factor": 1.0},
            target_downsample={"scale_factor": 1.0},
            input_quant={"prec": "fp32", "mode": "per_channel"},
            target_quant={"prec": "fp32", "mode": "per_channel"},
        )
    train_data = MMIDataset(
                root="./data",
                split="train",
                test_ratio=0.1,
                train_valid_split_ratio=[0.8, 0.2],
                pol_list=[f"rHz_{i}" for i in range(5)],
                processed_dir="random_size5",
                train_noise_cfg=train_noise,
            )
    
    train_data.data.data = train_data.data.data.to(torch.complex64)
    train_data.data.targets = train_data.data.targets.to(torch.complex64)

    
    
    fig, ax = plt.subplots(1, 1)

    size = np.ones([target_embedded.shape[1]])
    plt.scatter(
        target_embedded[0, :num_matrices],
        target_embedded[1, :num_matrices],
        c=c,
        cmap=plt.cm.rainbow,
        s=size,
        alpha=0.5,
    )
    size = np.zeros([target_embedded.shape[1]])
    size[[0, 7, 63, 511, 4095]] = 30
    plt.scatter(
        target_embedded[0, :num_matrices],
        target_embedded[1, :num_matrices],
        c=c,
        cmap=plt.cm.rainbow,
        s=size,
        alpha=1,
        marker="*",
        linewidths=0.5,
        edgecolors="black",
    )
    plt.colorbar()

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis("tight")
    set_axes_size_ratio(1, 1, fig, ax)

    plt.savefig("./figs/MMI_port_4_res_8_matrices_det2d.png", dpi=300)
