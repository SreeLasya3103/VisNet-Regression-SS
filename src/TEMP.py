import torch
import tomli
import sys
import os
ROOT_DIR = os.path.dirname(__file__)
print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
import FoggyCityscapesDBF as fcs
import FROSI as frosi
import SSF as ssf
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import Integrated
import RMEP as rmep
import VisNet
import torch
from torch.utils.data import Dataset, DataLoader

dataset = ssf.SSF_reg('/home/feet/Downloads/SSF', 'train', (120, 160))
dataloader = DataLoader(dataset, 16, True, collate_fn=dataset.collate_fn)
data, labels = next(iter(dataloader))
print(labels.size())