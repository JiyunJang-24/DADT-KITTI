import os 
import numpy as np 

scenes = os.listdir('../once/data/')
avg = 0 
filelen = 0
for scene in scenes:
    root = f'../once/data/{scene}/lidar_roof/'

    # files = [x.strip() for x in open('ImageSets/train.txt').readlines()] 
    files = os.listdir(root)
    filelen += len(files)
    for file in files: 
    
        pts = np.fromfile(root + file, dtype = np.float32).reshape([-1, 4])
        avg += pts.shape[0]

print(avg/filelen)
