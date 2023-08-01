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
from PIL import Image

img = Image.open('/home/feet/Downloads/FCS/train/a/foggy-road0.005.png').convert('YCbCr')

print(img.getdata()[0])