import torch
import tomli
import sys
import os
import matplotlib
from torch.utils.data import DataLoader
ROOT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
import FoggyCityscapesDBF as fcs
import FROSI as frosi
import SSF as ssf
import SSF_YCbCr as ssf_YCbCr


dataset = ssf.SSF_cls
cmap = matplotlib.colormaps['Greys']
classes = {}

train_set = dataset('/home/feet/Downloads/SSF/', 'train', (1,1), 3, cmap, 'BLUE', (0,0))
train_loader = DataLoader(train_set, 1, False, collate_fn=dataset.collate_fn)
for step, (_, labels) in enumerate(train_loader):
    label = labels[0].argmax().item()

    if label in classes:
        classes[label] += 1
    else:
        classes[label] = 1

print(classes)

#M1/4 (less than 1/4 mile), 1/4, 1/2, 3/4, 1, 1-1/4, 1-1/2, 2, 2-1/2, 3, 4, 5, 7, 10 and 10+ (greater than 10 miles)