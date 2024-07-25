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




# 데이터 준비 및 병합
#dataset = './custom/HMCData/VAN'
dataset = './kitti'
imagesets_folder = dataset + '/ImageSets'
label_folder = dataset + '/training/label_2'
train_txt_file = os.path.join(imagesets_folder, 'train.txt')
val_txt_file = os.path.join(imagesets_folder, 'val.txt')
calib_folder = dataset + '/training/calib/'
with open(train_txt_file, 'r') as file:
    train_filenames = [line.strip() for line in file.readlines()]
    
with open(val_txt_file, 'r') as file:
    val_filenames = [line.strip() for line in file.readlines()]
filenames  = train_filenames+val_filenames
print(len(filenames))




statistic = {} #statistic[class] = [[x, y, z], # of class]
result = {}
cnt = 0
for filename in filenames:
    if cnt % 1000 == 0:
        print(cnt)
    cnt = cnt+1
    label_file = os.path.join(label_folder, filename + '.txt')
    with open(label_file, 'r') as file:
        calib_file = calib_folder + filename + '.txt'
        calib = calibration_kitti.Calibration(calib_file)
        for line in file:
            elements = line.strip().split()
            if elements[0] != 'Car' and elements[0] != 'Pedestrian' and elements[0] != 'Cyclist':
                continue
            if elements[0] not in statistic:
                statistic[elements[0]] = [[0, 0, 0], 0]
            statistic[elements[0]][1] += 1
            category = elements[8:15]
            height, width, length, cen_x, cen_y, cen_z, rotation = map(float, category)
            boxes3d_camera = np.array([cen_x, cen_y, cen_z, length, height, width, rotation]).reshape(1, 7)
            boxes3d_lidar = boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib)
            statistic[elements[0]][0][0] += boxes3d_lidar[0][0]
            statistic[elements[0]][0][1] += boxes3d_lidar[0][1]
            statistic[elements[0]][0][2] += boxes3d_lidar[0][2]
            
            # {'Van': 437228, 'Car': 634993, 'Truck': 508389, 'Traffic_Cone': 1518, 'Guardrail': 780792, 'Bus': 1025338, 'Pedestrian': 25173, 'Cyclist': 54149, 'UV': 59847, 'Curb': 2459, 'Unknown_Object': 65, 'Rubber_Cone': 144, 'Ptw': 2059}
print(statistic)
for i in statistic.keys():
    statistic[i][0][0] /= statistic[i][1]
    statistic[i][0][1] /= statistic[i][1]
    statistic[i][0][2] /= statistic[i][1]
print(statistic)

#statistic[class] = [[x_mean, y_mean, z_mean], # of class]
#statistic2[class] = [x_var, y_var, z_var]

statistic2 = {}

for filename in filenames:
    if cnt % 1000 == 0:
        print(cnt)
    cnt = cnt+1
    label_file = os.path.join(label_folder, filename + '.txt')
    with open(label_file, 'r') as file:
        calib_file = calib_folder + filename + '.txt'
        calib = calibration_kitti.Calibration(calib_file)
        for line in file:
            elements = line.strip().split()
            if elements[0] != 'Car' and elements[0] != 'Pedestrian' and elements[0] != 'Cyclist':
                continue
            if elements[0] not in statistic2:
                statistic2[elements[0]] = [0, 0, 0]
            category = elements[8:15]
            height, width, length, cen_x, cen_y, cen_z, rotation = map(float, category)
            boxes3d_camera = np.array([cen_x, cen_y, cen_z, length, height, width, rotation]).reshape(1, 7)
            boxes3d_lidar = boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib)
            statistic2[elements[0]][0] += (boxes3d_lidar[0][0] - statistic[elements[0]][0][0])**2
            statistic2[elements[0]][1] += (boxes3d_lidar[0][1] - statistic[elements[0]][0][1])**2
            statistic2[elements[0]][2] += (boxes3d_lidar[0][2] - statistic[elements[0]][0][2])**2

print(statistic2)
for i in statistic2.keys():
    statistic2[i][0] /= statistic[i][1]
    statistic2[i][1] /= statistic[i][1]
    statistic2[i][2] /= statistic[i][1]
for i in statistic2.keys():
    statistic2[i][0] = statistic2[i][0]**0.5
    statistic2[i][1] = statistic2[i][1]**0.5
    statistic2[i][2] = statistic2[i][2]**0.5
    
print(statistic2)

