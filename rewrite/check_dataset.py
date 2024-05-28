import torch
import torch.nn as nn
import datasets
import datasets.Webcams
from torch.utils.data import Dataset, DataLoader

dset_dir = '/home/feet/Documents/LAWN/datasets/Webcams'

dataset = datasets.Webcams.Webcams_cls(dset_dir, nine_limit=999999, ten_limit=999999)
dataloader = DataLoader(dataset, 1, False)

class_counts = dict()

for i, (data, label) in enumerate(dataloader):
    argmax = torch.argmax(label).item()
    if argmax in class_counts:
        class_counts[argmax] += 1
    else:
        class_counts[argmax] = 1

print(sorted(class_counts.items(), key=lambda x: x[0]))