import os
import scipy
import torch
import copy
from scipy.spatial import Delaunay
import sys
import numpy as np
import mayavi.mlab as mlab
from tools.visual_utils import visualize_utils as V
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# FarthestPoint 클래스를 import
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import FarthestPointSampling
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

points = np.load('points_SUV3.npy').astype(np.float32)
'''points = torch.from_numpy(points).cuda()
#points = torch.Tensor(points)  # points를 PyTorch Tensor로 변환

npoint = int(points.shape[0]/5)  # 원하는 샘플 수
print(npoint)
sampled_points = FarthestPointSampling.apply(points.unsqueeze(0), npoint)  # FPS를 적용하여 샘플 선택
sampled_points = sampled_points.squeeze()  # 내부 리스트 제거 및 1차원 텐서로 변환
sampled_points = sampled_points.long()

print(sampled_points)
# 이제 sampled_points는 1차원 텐서로 되어 있어야 합니다.

# sampled_points를 사용하여 선택된 포인트를 추출
selected_points = points[sampled_points]
points = points.cpu()
selected_points = selected_points.cpu()
print(points.shape)
print(selected_points.shape)'''
selected_points = farthest_point_sample(points, int(points.shape[0]/4))

np.save('points_ori2.npy', points)
np.save('points_changed2.npy', selected_points)