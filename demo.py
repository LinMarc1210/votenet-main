# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset: sunrgbd or scannet [default: sunrgbd]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply
from ap_helper import parse_predictions

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

if __name__=='__main__':
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files') 
    if FLAGS.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        from sunrgbd_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
        pc_path = os.path.join(demo_dir, 'sunrgbd_results', 'pointcloud_20250604_175004.ply')
    elif FLAGS.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_scannet.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_scannet.ply')
    else:
        print('Unkown dataset %s. Exiting.'%(DATASET))
        exit(-1)

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5, 'dataset_config': DC}

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet') # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MODEL.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
        sampling='seed_fps', num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr).to(device)
    print('Constructed model.')

    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))

    # Load and preprocess input point cloud 
    net.eval() # set model to eval mode (for bn and dp)
    point_cloud = read_ply(pc_path)
    pc = preprocess_point_cloud(point_cloud)
    print('Loaded point cloud data: %s'%(pc_path))

    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs)
    toc = time.time()
    print('Inference time: %f'%(toc-tic))
    end_points['point_clouds'] = inputs['point_clouds']
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))

    dump_dir = os.path.join(demo_dir, '%s_results'%(FLAGS.dataset))
    if not os.path.exists(dump_dir): os.mkdir(dump_dir) 
    MODEL.dump_results(end_points, dump_dir, DC, True)
    print('Dumped detection results to folder %s'%(dump_dir))

    # 額外輸出一個 .txt，包含中心點、尺寸、heading、類別、分數
    box_centers = end_points['center'].cpu().numpy()[0]  # (num_boxes, 3)
    box_sizes = end_points['size_residuals'].cpu().numpy()[0]  # (num_boxes, num_size_cluster, 3)
    box_headings = end_points['heading_residuals'].cpu().numpy()[0]  # (num_boxes, num_heading_bin)
    sem_cls_scores = end_points['sem_cls_scores'].softmax(-1).cpu().numpy()[0]  # (num_boxes, num_classes)
    sem_cls_preds = np.argmax(sem_cls_scores, axis=1)
    sem_cls_confs = np.max(sem_cls_scores, axis=1)

    CLASS_NAMES = [
        'bed', 'bookshelf', 'chair', 'desk', 'dresser',
        'night_stand', 'sofa', 'table', 'tv_stand', 'toilet'
    ]

    output_path = os.path.join(dump_dir, 'detection_summary.txt')
    with open(output_path, 'w') as f:
        f.write("x\ty\tz\tlength\twidth\theight\tclass_name\tscore\n")
        pred_mask = end_points['pred_mask'][0]
        for i in range(box_centers.shape[0]):
            if pred_mask[i] == 0:
                continue  # 跳過非物體的框
            x, y, z = box_centers[i]
            cls = sem_cls_preds[i]
            score = sem_cls_confs[i]
            class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else 'unknown'
            lx, ly, lz = 0.8, 0.8, 0.8
            f.write(f"{x:.3f}\t{y:.3f}\t{z:.3f}\t{lx:.3f}\t{ly:.3f}\t{lz:.3f}\t{class_name}\t{score:.3f}\n")

    print(f"✅ 類別與尺寸資訊已輸出至: {output_path}")
