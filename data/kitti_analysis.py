import numpy as np 
import pickle
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
split_dir = 'kitti/ImageSets/train.txt'
sample_id_list = [x.strip() for x in open(split_dir).readlines()] 
abc = pickle.load(open('kitti/kitti_infos_train.pkl', 'rb'))
classes = ['Car', 'Pedestrian', 'Cyclist']
scenes = {}

for ab in abc:
    idx = ab['point_cloud']['lidar_idx']
    annos = ab['annos']
    name = annos['name']
    num_points = annos['num_points_in_gt']
    points = np.fromfile(f'kitti/training/velodyne/{idx}.bin', dtype = np.float32).reshape(-1, 4)
    total = []
    for cls in classes:
        mask = (name == cls)
        nums = num_points[mask] 
        if len(nums) == 0:
            total.append(0)
        else:
            total.append(nums.sum()/len(nums))
        
        
    total = np.array(total)
    total = total.sum() / 3
    scenes[idx] = total


ans = sorted(scenes.items(), key = lambda x: x[1], reverse = True)[:185]
with open('new.txt', 'w') as f: 
    for an in ans: 
        f.write(an[0] + '\n')
