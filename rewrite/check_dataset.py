import torch
import torch.nn as nn
import datasets
import datasets.Webcams
from torch.utils.data import Dataset, DataLoader

dset_dir = '/home/feet/Documents/LAWN/datasets/Webcams'

class_names = ('1.0', '1.25', '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0')

dataset = datasets.Webcams.Webcams_cls(dset_dir, nine_limit=999999, ten_limit=999999)
dataloader = DataLoader(dataset, 1, False)

class_counts = dict()

for i, (data, label) in enumerate(dataloader):
    argmax = torch.argmax(label).item()
    if argmax in class_counts:
        class_counts[argmax] += 1
    else:
        class_counts[argmax] = 1

_, counts_list = zip(*sorted(class_counts.items(), key=lambda x: x[0]))

counts_list = list(zip(class_names, counts_list))

for name, count in counts_list:
    print(name + ': ' + str(count), end=', ')
