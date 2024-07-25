import numpy as np
import os
import scipy
from sklearn.cluster import KMeans
from sklearn import cluster
from tqdm import tqdm

def compute_angles(pc_np):
    tan_theta = pc_np[:, 2] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    theta = np.arctan(tan_theta)
    theta = (theta / np.pi) * 180

    sin_phi = pc_np[:, 1] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi_ = np.arcsin(sin_phi)
    phi_ = (phi_ / np.pi) * 180

    cos_phi = pc_np[:, 0] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi = np.arccos(cos_phi)
    phi = (phi / np.pi) * 180

    phi[phi_ < 0] = 360 - phi[phi_ < 0]
    phi[phi == 360] = 0

    return theta, phi

def beam_label(theta, beam):
    estimator=KMeans(n_clusters=beam, n_init=10)
    res=estimator.fit_predict(theta.reshape(-1, 1))
    label=estimator.labels_
    centroids=estimator.cluster_centers_
    return label, centroids[:,0]

kitti_dataset = './once/data/000027/lidar_roof/' 
files = os.listdir(kitti_dataset)
files = files[:50]
total = 0
for file in tqdm(files): 
    pc = np.fromfile(kitti_dataset + file, dtype=np.float32).reshape([-1, 4])[:, :3]
    theta, phi = compute_angles(pc)
    beam = 40
    label, centroids = beam_label(theta, beam)
    arr = [0] * beam
    for j in label:
        arr[j] += 1
    mean = np.mean(arr)
    total += mean
print(total / len(files))
