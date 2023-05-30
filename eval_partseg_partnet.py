"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil

import trimesh

import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset
import hydra
import omegaconf
from PartNetSegPCMultiClassDataset import PartNetSegPCMultiClassDataset, visualize_part_segmentation


# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
# seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
# for cat in seg_classes.keys():
#     for label in seg_classes[cat]:
#         seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

@hydra.main(config_path='config', config_name='partseg_partnet')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    # print(args.pretty())
    print(args)

    full_dataset = PartNetSegPCMultiClassDataset(args.data_dir, partial=False, rotate='None', keep_object_classes=args.keep_object_classes, ndf_scale=args.ndf_scale, num_pts=args.num_point)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * 0.7), len(full_dataset) - int(len(full_dataset) * 0.7)], generator=torch.Generator().manual_seed(args.random_seed))
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    args.input_dim = 3 + full_dataset.get_num_class()
    args.num_class = len(full_dataset.get_parts())
    num_category = full_dataset.get_num_class()
    num_part = args.num_class

    seg_classes = {}
    # seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
    #                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
    #                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
    #                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
    for obj_cls in full_dataset.obj_cls_to_part_paths:
        for part_path in full_dataset.obj_cls_to_part_paths[obj_cls]:
            if obj_cls not in seg_classes:
                seg_classes[obj_cls] = []
            seg_classes[obj_cls].append(full_dataset.part_path_to_idx[part_path])
    seg_label_to_cat = {}
    # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            assert label not in seg_label_to_cat, "Part labels cannot be shared between object classes"
            seg_label_to_cat[label] = cat

    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()

        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            seg_pred = classifier(torch.cat([points, to_categorical(label, num_category).repeat(1, points.shape[1], 1)], -1))  # B, N, num_class
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)  # B, N

            target = target.cpu().data.numpy()
            points = points.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]  # get the category of this object
                logits = cur_pred_val_logits[i, :, :]  # B, num_class
                # Important: logits[: seg_classes[cat]] will ignore seg labels that this object class does not support
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]  # get seg labels for each point for this object

                # visualization
                gt_seg_pc = visualize_part_segmentation(points[i], target[i], all_labels=full_dataset.get_parts(), return_vis_pc=True)
                pred_seg_pc = visualize_part_segmentation(points[i], cur_pred_val[i], all_labels=full_dataset.get_parts(), return_vis_pc=True)
                for vis_pc in gt_seg_pc:
                    vis_pc.apply_translation([-2, 0, 0])
                if len(gt_seg_pc + pred_seg_pc):
                    vis_scene = trimesh.Scene()
                    vis_scene.add_geometry(gt_seg_pc + pred_seg_pc)
                    vis_scene.show()

            correct = np.sum(cur_pred_val == target)  # the number of points that are correctly labeled
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]  # predicted seg labels for this object: (N,)
                segl = target[i, :]  # gt seg labels for this object: (N,)
                cat = seg_label_to_cat[segl[0]]  # object class
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]  # len is equal to the number of parts this object class supports
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            logger.info('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)


if __name__ == '__main__':
    main()