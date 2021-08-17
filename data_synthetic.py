
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
import copy
from argoverse.data_loading.argoverse_forecasting_loader import (
    ArgoverseForecastingLoader,
)
from argoverse.map_representation.map_api import ArgoverseMap
from skimage.transform import rotate
from data import ArgoDataset, dilated_nbrs, dilated_nbrs2

from data_scene import Scene


class SyntheticDataset(ArgoDataset):
    def __init__(self, split, config, train=True):
        self.config = config
        self.train = train

        # only option for synthetic data
        self.online_preprocess = True

        self.examples = [
            Scene.create_simple,
            Scene.create_follow,
        ]

    def __len__(self):
        # Return number of scenarios
        return len(self.examples)

    def read_argo_data(self, idx):

        scene = self.examples[idx]()

        data = scene.to_lanegcn()
        data["argo_id"] = f"synthetic_{idx:03d}"
        return data

    def get_lane_graph(self, data):

        scene_lanes = data["scene"].lanes
        lane_ids = list(scene_lanes.keys())

        # graph = {
        #     "ctrs": None,   # ndarray [num_nodes, 2], xy (?)
        #     "num_nodes": num_nodes,
        #     "feats": None,  # ndarray [num_nodes, 2], delta between ctrs (?)
        #     "ctrs": None,
        #
        # }

        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = self.config["pred_range"]
        # radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))

        # lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
        # lane_ids = copy.deepcopy(lane_ids)

        lanes = dict()
        for lane_id in lane_ids:
            # lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = scene_lanes[lane_id]

            lane = copy.deepcopy(lane)
            centerline = np.matmul(
                data["rot"], (lane.centerline - data["orig"].reshape(-1, 2)).T
            ).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                # polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                # polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                # lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == "LEFT":
                x[:, 0] = 1
            elif lane.turn_direction == "RIGHT":
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count

        pre, suc = dict(), dict()
        for key in ["u", "v"]:
            pre[key], suc[key] = [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]
            idcs = node_idcs[i]

            pre["u"] += idcs[1:]
            pre["v"] += idcs[:-1]
            if lane.predecessors is not None:
                for nbr_id in lane.predecessors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre["u"].append(idcs[0])
                        pre["v"].append(node_idcs[j][-1])

            suc["u"] += idcs[:-1]
            suc["v"] += idcs[1:]
            if lane.successors is not None:
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc["u"].append(idcs[-1])
                        suc["v"].append(node_idcs[j][0])

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.l_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.r_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int64)
        if len(pre_pairs) == 0:
            pre_pairs = np.zeros(shape=(0, 2), dtype=np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        if len(suc_pairs) == 0:
            suc_pairs = np.zeros(shape=(0, 2), dtype=np.int64)

        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)

        graph = dict()
        graph["ctrs"] = np.concatenate(ctrs, 0)
        graph["num_nodes"] = num_nodes
        graph["feats"] = np.concatenate(feats, 0)
        graph["turn"] = np.concatenate(turn, 0)
        graph["control"] = np.concatenate(control, 0)
        graph["intersect"] = np.concatenate(intersect, 0)
        graph["pre"] = [pre]
        graph["suc"] = [suc]
        graph["lane_idcs"] = lane_idcs
        graph["pre_pairs"] = pre_pairs
        graph["suc_pairs"] = suc_pairs
        graph["left_pairs"] = left_pairs
        graph["right_pairs"] = right_pairs

        for k1 in ["pre", "suc"]:
            for k2 in ["u", "v"]:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)

        for key in ["pre", "suc"]:
            if "scales" in self.config and self.config["scales"]:
                # TODO: delete here
                graph[key] += dilated_nbrs2(
                    graph[key][0], graph["num_nodes"], self.config["scales"]
                )
            else:
                graph[key] += dilated_nbrs(
                    graph[key][0], graph["num_nodes"], self.config["num_scales"]
                )
        return graph
