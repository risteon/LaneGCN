# ---------------------------------------------------------------------------
# Learning Lane Graph Representations for Motion Forecasting
#
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Ming Liang, Yun Chen
# ---------------------------------------------------------------------------

import argparse
import os
import numpy as np

import pickle
import sys
from importlib import import_module

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import ArgoTestDataset
from data_synthetic import SyntheticDataset
from utils import Logger, load_pretrain
from preprocess_data import preprocess, to_long, gpu

from test import forward_pass
import matplotlib.pyplot as plt

import collections
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as pe
from scipy.ndimage import rotate
import seaborn as sns


line_colors = ['#375397', '#F05F78', '#80CBE5', '#ABCB51', '#C8B0B0']

cars = [plt.imread('icons/Car TOP_VIEW 375397.png'),
        plt.imread('icons/Car TOP_VIEW F05F78.png'),
        plt.imread('icons/Car TOP_VIEW 80CBE5.png'),
        plt.imread('icons/Car TOP_VIEW ABCB51.png'),
        plt.imread('icons/Car TOP_VIEW C8B0B0.png')]

robot = plt.imread('icons/Car TOP_VIEW ROBOT.png')


# define parser
parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="angle90", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="val", help='data split, "val" or "test"'
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument("--synthetic", default=False, action="store_true")


def make_figure(box_coords, figsize):
    x_min, y_min, x_max, y_max = box_coords

    fig = plt.figure(figsize=figsize)

    local_width = x_max - x_min
    local_height = y_max - y_min
    assert local_height > 0, "Error: Map patch has 0 height!"
    local_aspect_ratio = local_width / local_height

    ax = fig.add_axes([0, 0, 1, 1 / local_aspect_ratio])

    x_margin = np.minimum(local_width / 4, 50)
    y_margin = np.minimum(local_height / 4, 10)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    return fig, ax


def plot(outputs, gts, hist):
    left, bottom = 0.0, 0.0
    delta = 50.0

    my_patch = (
        left,
        bottom,
        left + delta,
        bottom + delta,
    )
    x_min = -50.0
    y_min = -25.0
    ph = 30

    fig, ax = make_figure(my_patch, figsize=(10, 10))

    plot_vehicle_nice(
        ax,
        outputs[0],
        outputs[1],
        gts,
        hist,
        0.1,
        max_hl=10,
        ph=ph,
        map=None,
        x_min=x_min,
        y_min=y_min,
        scale=delta,
    )

    leg = ax.legend(loc="upper right", fontsize=20, frameon=True)
    ax.axis("off")
    for lh in leg.legendHandles:
        lh.set_alpha(0.5)
    ax.get_legend().remove()

    fig.show()
    fig.savefig(
        f"/tmp/lanegcn_visu_qual_synthetic.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def plot_vehicle_nice(ax, predictions, latent, gt_future, hist, dt, max_hl=10, ph=6, map=None, x_min=0, y_min=0, scale=50.0):
    # prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(predictions,
    #                                                                                   dt,
    #                                                                                   max_hl,
    #                                                                                   ph,
    #                                                                                   map=map)
    # assert (len(prediction_dict.keys()) <= 1)
    # if len(prediction_dict.keys()) == 0:
    #     return
    # ts_key = list(prediction_dict.keys())[0]
    #
    # prediction_dict = prediction_dict[ts_key]
    # histories_dict = histories_dict[ts_key]
    # futures_dict = futures_dict[ts_key]

    # if latent is not None:
    #     latent_dict = latent[ts_key]
    # else:
    #     latent_dict = collections.defaultdict(lambda: 0.0)

    if map is not None:
        ax.imshow(map.fdata, origin='lower', alpha=0.5)

    edge_width = 2
    circle_edge_width = 0.5
    node_circle_size = 0.3
    a = []
    i = 0
    # node_list = sorted(histories_dict.keys(), key=lambda x: x.id)
    for node_idx in range(len(predictions)):

        # history = histories_dict[node] + np.array([x_min, y_min])
        # future = futures_dict[node] + np.array([x_min, y_min])
        # predictions = prediction_dict[node] + np.array([x_min, y_min])
        # latent_prob = latent_dict[node]

        history = hist[node_idx]
        future = gt_future[node_idx]

        if True: # node.type.name == 'VEHICLE':
            # ax.plot(history[:, 0], history[:, 1], 'ko-', linewidth=1)

            ax.plot(future[4::5, 0],
                    future[4::5, 1],
                    'w--o',
                    linewidth=7,
                    markersize=4,
                    zorder=720,
                    alpha=0.8,
                    color="green",
                    path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])

            # for t in range(predictions.shape[2]):
            #     sns.kdeplot(predictions[0, :, t, 0], predictions[0, :, t, 1],
            #                 ax=ax, shade=True, shade_lowest=False,
            #                 color=line_colors[i % len(line_colors)], zorder=600, alpha=0.6)

            # vel = node.get(np.array([ts_key]), {'velocity': ['x', 'y']})
            # h = np.arctan2(vel[0, 1], vel[0, 0])
            # todo:
            heading = 0.0
            r_img = rotate(cars[i % len(cars)], heading * 180 / np.pi,
                           reshape=True)
            oi = OffsetImage(r_img, zoom=0.025 * 50.0 / scale, zorder=700)
            veh_box = AnnotationBbox(oi, (history[-1, 0], history[-1, 1]), frameon=False)
            veh_box.zorder = 700
            ax.add_artist(veh_box)
            # ax.text(history[-1, 0], history[-1, 1], f'{latent_prob:.3f}', zorder=720)


            pred = predictions[node_idx]
            for i, p in enumerate(pred):
                ax.plot(p[4::5, 0],
                        p[4::5, 1], 'ko-',
                        zorder=750,
                        markersize=3,
                        linewidth=2, alpha=0.5)

            i += 1
        else:
            # ax.plot(history[:, 0], history[:, 1], 'k--')

            for t in range(predictions.shape[2]):
                sns.kdeplot(predictions[0, :, t, 0], predictions[0, :, t, 1],
                            ax=ax, shade=True, shade_lowest=False,
                            color='b', zorder=600, alpha=0.8)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--',
                    zorder=650,
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])
            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

            ax.text(history[-1, 0], history[-1, 1], f'{latent_prob:.3f}', zorder=720)


def main():
    args = parser.parse_args()
    outputs, gts, hist, cities = forward_pass(args)

    for k in outputs:
        plot(outputs[k], gts[k], hist[k])


if __name__ == "__main__":
    main()
