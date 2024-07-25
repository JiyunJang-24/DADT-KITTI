def eval_per_distance(eval_class, gt_annos_dict=None):
        """
        Args:
            eval_class:
            points: (N, 3), numpy 
            gt_boxes: (N, 7), numpy (lidar cordinates)
            gt_scores: (N,), numpy
            gt_labels: (N,), list
            
            or
            
            eval_class:
            points: (B, N, 3),
            gt_boxes: (B, N, 7)
            gt_scores: (B, N,),
            gt_labels: (B, N,)
        Returns:
        """
        import copy
        if gt_annos_dict is not None:
            copy_gt = copy.deepcopy(gt_annos_dict)
            if eval_class==None:
                return gt_annos_dict
            elif eval_class == 1:
                mask = (gt_annos_dict[:, 0] > 0) & (gt_annos_dict[:, 0] < 10) & (gt_annos_dict[:, 1] > -10) & (gt_annos_dict[:, 1] < 10)
            elif eval_class == 2:
                mask = (gt_annos_dict[:, 0] > 10) & (gt_annos_dict[:, 0] < 20) & (gt_annos_dict[:, 1] > -20) & (gt_annos_dict[:, 1] < 20)
            
            elif eval_class == 3:
                mask = (gt_annos_dict[:, 0] > 20) & (gt_annos_dict[:, 0] < 30) & (gt_annos_dict[:, 1] > -30) & (gt_annos_dict[:, 1] < 30)
            elif eval_class == 4:
                mask = (gt_annos_dict[:, 0] > 30) & (gt_annos_dict[:, 0] < 40) & (gt_annos_dict[:, 1] > -40) & (gt_annos_dict[:, 1] < 40)
            elif eval_class == 5:
                mask = (gt_annos_dict[:, 0] > 40) & (gt_annos_dict[:, 0] < 70) & (gt_annos_dict[:, 1] > -40) & (gt_annos_dict[:, 1] < 40)
            #gt_annos_dict['gt_boxes_lidar']=gt_annos_dict['gt_boxes_lidar'][mask]
            copy_gt = gt_annos_dict[mask]
            return copy_gt
        

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import torch
import copy
from scipy.spatial import Delaunay
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils, calibration_kitti

def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]

    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)


def remove_points_in_boxes3d(points, boxes3d):
    """
    Args:
        points: (num_points, 3 + C)
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps

    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) == 1]
    
    return points.numpy() if is_numpy else points



# 데이터 준비 및 병합
dataset = './custom/HMCData/VAN'
#dataset = './kitti'
imagesets_folder = dataset + '/ImageSets'
label_folder = dataset + '/training/label_2'
train_txt_file = os.path.join(imagesets_folder, 'train.txt')
val_txt_file = os.path.join(imagesets_folder, 'val.txt')
velodyne_folder = dataset + '/training/velodyne/'
calib_folder = dataset + '/training/calib/'
with open(train_txt_file, 'r') as file:
    train_filenames = [line.strip() for line in file.readlines()]
    
with open(val_txt_file, 'r') as file:
    val_filenames = [line.strip() for line in file.readlines()]
    
filenames  = train_filenames+val_filenames
filenames.sort()
print(len(filenames))


#구간별 object들의 x, y, z 평균 뽑아보기

statistic = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] #statistic = [# rotation_y, # of class]
cnt = 0
for filename in filenames:
    if cnt % 1000 == 0:
        print(cnt)
    cnt = cnt+1
    pointcloud1 = np.fromfile(velodyne_folder + filename + '.bin', dtype=np.float32).reshape([-1, 4])[:, :3]
    label_file = os.path.join(label_folder, filename + '.txt')
    with open(label_file, 'r') as file:
        calib_file = calib_folder + filename + '.txt'
        calib = calibration_kitti.Calibration(calib_file)
        boxes3d_camera_list = np.array([0, 0, 0, 0, 0, 0, 0])
        boolean = False
        for line in file:
            elements = line.strip().split()
            if elements[0] != 'Car':
                continue
            category = elements[8:15]
            height, width, length, cen_x, cen_y, cen_z, rotation = map(float, category)
            boxes3d_camera = np.array([cen_x, cen_y, cen_z, length, height, width, rotation]).reshape(1, 7)
            #if cen_z > 70:
            #    continue
            boxes3d_camera_list = np.vstack((boxes3d_camera_list, boxes3d_camera))
            #statistic[1] += 1
            boolean = True
        boxes3d_camera_list = boxes3d_camera_list[1:]
        if boolean:
            boxes3d_lidar = boxes3d_kitti_camera_to_lidar(boxes3d_camera_list, calib)
            boxes3d_lidar = boxes3d_lidar[boxes3d_lidar[:, 0] <= 70]
            boxes3d_lidar = boxes3d_lidar[boxes3d_lidar[:, 0] >= -70]
            #TODO boxes3d_lidar, pointcloud1을 을 input으로 넣어서 박스별 계산
            for idx in range(len(statistic)):
                boxes3d_lidar_region = eval_per_distance(gt_annos_dict=boxes3d_lidar, eval_class=idx+1)
                statistic[idx][1] += boxes3d_lidar_region.shape[0]
                x = np.sum(boxes3d_lidar_region, axis=0)
                statistic[idx][0] += x[6]

print(statistic)
result = []
for i in statistic:
    list1 = []
    list1.append(float(i[0]) / float(i[1]))
    result.append(list1)
print(result)