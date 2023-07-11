import torch
import tomllib
import sys
import matplotlib
sys.path.append("src/datasets")
import FoggyCityscapesDBF as fcs
import FROSI
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

def gray_fog_highlight(img):
    img = img.numpy()
            
    with np.nditer(img, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = (255 * 1.02**(x*255 - 255))/255
    
    img = np.concatenate((img, img, img))
    img = np.transpose(img, (1, 2, 0))
    img = np.array([img])
    
    return np.clip(img, 0, 1)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#0000ff', '#00ff00', '#ff0000', '#0000ff'])

#cmap = gray_fog_highlight

dataset = FROSI.FROSI('/home/feet/Downloads/FROSI', 'train', (60, 140), 3, cmap, 'AVERAGE')
dataloader = DataLoader(dataset, 16, True, collate_fn=dataset.collate_fn)
data, labels = next(iter(dataloader))
print(data.size())
transforms.ToPILImage()(data[1][0]).show('a')