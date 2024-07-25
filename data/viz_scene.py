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
    if boxes3d.shape == (7,):
        boxes3d = np.expand_dims(boxes3d, axis=0)
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) == 1]
    
    return points.numpy() if is_numpy else points

#car인데 x값이 0부터 60까지 10 단위로 끊어서 그리기.

# 데이터 준비 및 병합
dataset = './custom/HMCData/SUV'
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

max_x = 80
pts = np.array([0, 0, 0])
check=[]
for i in range(max_x//10):
    check.append(0)
max_ = 0
cnt = 0
num=0
for filename in filenames:
    if cnt % 1000 == 0:
        print(cnt)
    
    cnt = cnt+1
    pointcloud1 = np.fromfile(velodyne_folder + filename + '.bin', dtype=np.float32).reshape([-1, 4])[:, :3]
    pointcloud1 = pointcloud1[pointcloud1[:, 0] <= 80]
    pointcloud1 = pointcloud1[pointcloud1[:, 0] >= 0]
    boxes3d_lidar_condition= np.array([0, 0, 0, 0, 0, 0, 0])
    label_file = os.path.join(label_folder, filename + '.txt')
    with open(label_file, 'r') as file:
        calib_file = calib_folder + filename + '.txt'
        calib = calibration_kitti.Calibration(calib_file)
        boxes3d_camera_list = np.array([0, 0, 0, 0, 0, 0, 0])
        boolean = False
        for line in file:
            elements = line.strip().split()
            if elements[0] != 'Cyclist':
                continue
            category = elements[8:15]
            height, width, length, cen_x, cen_y, cen_z, rotation = map(float, category)
            boxes3d_camera = np.array([cen_x, cen_y, cen_z, length, height, width, rotation]).reshape(1, 7)
            boxes3d_camera_list = np.vstack((boxes3d_camera_list, boxes3d_camera))
            #statistic[1] += 1
            boolean = True
        boxes3d_camera_list = boxes3d_camera_list[1:]
        if boolean:
            boxes3d_lidar = boxes3d_kitti_camera_to_lidar(boxes3d_camera_list, calib)
            boxes3d_lidar = boxes3d_lidar[boxes3d_lidar[:, 0] <= 10]
            boxes3d_lidar = boxes3d_lidar[boxes3d_lidar[:, 0] >= 0]
            num += boxes3d_lidar.shape[0]
            pts = np.vstack((pts, remove_points_in_boxes3d(pointcloud1, boxes3d_lidar)))
            if num > 10:
                np.save('local_send/points_SUV.npy', pts)
                print(num)
                print(pts.shape)
                print(Asdf)
    #pts = pts[1:]
np.save('local_send/points_VAN.npy', pts)

            
            
            


