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



# 데이터 준비 및 병합
#dataset = './custom/HMCData/VAN'
dataset = './kitti'
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




statistic = [0, 0, 0, 0] #statistic[class] = [# points in object, # of class]
cnt = 0
for filename in filenames:
    if cnt % 1000 == 0:
        print(cnt)
    cnt = cnt+1
    pointcloud1 = np.fromfile(velodyne_folder + filename + '.bin', dtype=np.float32).reshape([-1, 4])[:, :3]
    statistic[0] += np.max(pointcloud1[:, 0])
    statistic[1] += np.max(pointcloud1[:, 1])
    statistic[2] += np.min(pointcloud1[:, 2])
    statistic[3] += 1
print(statistic)
for i in range(3):
    print(statistic[i] / statistic[3])


