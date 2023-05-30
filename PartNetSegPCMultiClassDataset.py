import h5py
import trimesh

import os
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
import h5py
import json
# from utils.provider import rotate_point_cloud_SO3, rotate_point_cloud_y
import pickle as pkl
import trimesh
import matplotlib

import os
import os.path as osp
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from main.partnet_dataset import PartNetObject, get_rgb_colors


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


def pc_scale_max(pc):
    dimension = np.abs(np.max(pc, axis=0) - np.min(pc, axis=0))
    # print("dimension:", dimension)
    long_side = np.max(dimension)
    # Scales all dimensions equally.
    pc = pc / long_side
    return pc


def semi_points_transform(points):
    spatialExtent = np.max(points, axis=0) - np.min(points, axis=0)
    eps = 2e-3*spatialExtent[np.newaxis, :]
    jitter = eps*np.random.randn(points.shape[0], points.shape[1])
    points_ = points + jitter
    return points_


class PartNetSegPCMultiClassDataset(Dataset):
    def __init__(self, data_dir, partial=False, rotate='None',
                 keep_object_classes=None, ndf_scale=False, num_pts=1500):
        super().__init__()

        self.data_dir = data_dir

        self.partial = partial
        self.rotate = rotate
        self.ndf_scale = ndf_scale
        self.num_pts = num_pts

        self.object_classes = []
        self.all_data = []
        for obj_cls in os.listdir(self.data_dir):
            if keep_object_classes and obj_cls not in keep_object_classes:
                continue
            self.object_classes.append(obj_cls)
            cls_dir = os.path.join(self.data_dir, obj_cls)
            for filename in os.listdir(cls_dir):
                self.all_data.append(os.path.join(cls_dir, filename))

        print("{} object classes: {}".format(len(self.object_classes), self.object_classes))
        print("{} data points".format(len(self.all_data)))

        self.part_path_to_idx = {}
        self.obj_cls_to_part_paths = {}
        self.obj_cls_to_idx = {}
        self.build_part_path_idxs()
        print("{} part paths: {}".format(len(self.part_path_to_idx), self.part_path_to_idx))

        # self.single_affordance = single_affordance
        # if self.single_affordance:
        #     self.single_affordance_idx = self.affordance.index(self.single_affordance)
        return

    def get_object_classes(self):
        return self.object_classes

    def get_parts(self):
        return sorted(list(self.part_path_to_idx.keys()))

    def get_num_class(self):
        return len(self.part_path_to_idx)

    def build_part_path_idxs(self):
        for obj_cls in sorted(self.object_classes):
            self.obj_cls_to_idx[obj_cls] = len(self.obj_cls_to_idx)

        unique_part_paths = set()
        obj_cls_to_part_paths = defaultdict(set)
        for h5_filename in tqdm(self.all_data):
            data_dict = PartNetObject.load_partobj_pc_and_segmentation_from_h5(h5_filename)
            point_to_part_path = data_dict["point_to_part_path"]
            obj_cls = data_dict["obj_cls"]
            unique_part_paths.update(set(point_to_part_path))
            obj_cls_to_part_paths[obj_cls].update(set(point_to_part_path))

        for part_path in sorted(unique_part_paths):
            self.part_path_to_idx[part_path] = len(self.part_path_to_idx)

        self.obj_cls_to_part_paths = {obj_cls: sorted(list(obj_cls_to_part_paths[obj_cls])) for obj_cls in obj_cls_to_part_paths}

    def __getitem__(self, index):

        h5_filename = self.all_data[index]
        data_dict = PartNetObject.load_partobj_pc_and_segmentation_from_h5(h5_filename)

        pc = data_dict["xyzs"].astype(np.float32)
        point_to_part_path = data_dict["point_to_part_path"]
        label = np.array([self.part_path_to_idx[pp] for pp in point_to_part_path], dtype=np.int64)
        # label = np.expand_dims(label, 1)  # label has dimension: num_pts, 1

        # important: use ndf scale
        if self.ndf_scale:
            pc = pc_scale_max(pc)
            pc = pc * 0.3
        else:
            pc, _, _ = pc_normalize(pc)
        rand_idx = np.random.permutation(len(pc))[:self.num_pts]
        pc = pc[rand_idx]
        label = label[rand_idx]
        obj_cls = np.array([self.obj_cls_to_idx[data_dict["obj_cls"]]]).astype(np.int32)

        # if self.rotate:
        # pc = trimesh.transform_points(pc, matrix=tra.random_rotation_matrix()).astype(pc.dtype)

        # datum = {"coords": pc, "point_cloud": pc, "shapenet_id": data_dict["shapenet_id"], "object_class": data_dict["obj_cls"]}
        # return pc, label,
        # point_set, cls, seg
        return pc, obj_cls, label

    def __len__(self):
        return len(self.all_data)


def visualize_part_segmentation(pc, seg_labels, all_labels, return_vis_pc=False):

    vis_pc = trimesh.PointCloud(pc[:, :3], colors=[0, 0, 0, 100])

    colors = get_rgb_colors()

    print(colors)

    vis_pcs = []
    if seg_labels.ndim == 1:
        unique_labels = all_labels
        for li, l in enumerate(unique_labels):
            color = colors[li][1]
            color = [int(c) * 255 for c in color] + [255]
            vis_pc.colors[seg_labels == li] = color
            if np.sum(seg_labels == li) > 0:
                print("color {} for part {}".format(colors[li][0], all_labels[li]))
        if return_vis_pc:
            vis_pcs.append(vis_pc)
        else:
            vis_pc.show()

    if return_vis_pc:
        return vis_pcs


if __name__ == "__main__":

    dataset = PartNetSegPCMultiClassDataset("/home/weiyu/data_drive/shapenet/partnet/pc_seg", partial=False, rotate='None',
                                            keep_object_classes=["Bottle"], ndf_scale=True)

    DataLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for pc, cls, seg in DataLoader:
        print(pc.shape)
        print(cls.shape)
        print(seg.shape)
        input("next?")

    # for i in np.random.permutation(len(dataset)):
    #     datum, labels = dataset[i]
    #
    #     # datum = {"coords": pc, "point_cloud": pc, "shapenet_id": data_dict["shapenet_id"],
    #     #          "object_class": data_dict["obj_cls"]}
    #     # return datum, label
    #
    #     pc = datum["point_cloud"]
    #     coords = datum["coords"]
    #     shapenet_id = datum["shapenet_id"]
    #     obj_cls = datum["object_class"]
    #
    #     print(labels)
    #     print(np.unique(labels))
    #
    #     print(pc.shape)
    #     print(labels.shape)
    #     print(shapenet_id)
    #     print(obj_cls)
    #
    #     visualize_part_segmentation(pc, labels, all_labels=np.unique(labels))
    #
    #     input("next?")

