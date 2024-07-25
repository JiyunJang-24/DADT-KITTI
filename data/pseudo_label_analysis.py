def eval_per_distance(eval_class, gt_annos_dict=None):
        """
        Args:
            eval_class:
            points: (N, 3), numpy 
            gt_boxes: (N, 7), numpy (lidar cordinates)
            gt_scores: (N,), numpy
            gt_labels: (N,), list
            
           
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
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


# 데이터 준비 및 병합
pseudo_label_batch = np.load('dt_boxes_lidar.npy')

print(pseudo_label_batch.shape)

statistic = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] #statistic = [# points in object, # of class]
cnt = 0
for pseudo_label in pseudo_label_batch.shape[0]:
    if cnt % 1000 == 0:
        print(cnt)
    cnt = cnt+1
    
    for idx in range(len(statistic)):
        boxes3d_lidar_region = eval_per_distance(gt_annos_dict=pseudo_label, eval_class=idx+1)
        statistic[idx][3] += boxes3d_lidar_region.shape[0]
        x = np.sum(boxes3d_lidar_region, axis=0)
        statistic[idx][0] += x[0]
        statistic[idx][1] += x[1]
        statistic[idx][2] += x[2]

print(statistic)
result = []
for i in statistic:
    list1 = []
    list1.append(float(i[0]) / float(i[3]))
    list1.append(float(i[1]) / float(i[3]))
    list1.append(float(i[2]) / float(i[3]))
    result.append(list1)
print(result)