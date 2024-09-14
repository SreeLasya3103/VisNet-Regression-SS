import torch
import torch.nn as nn
import dsets
import dsets.Webcams
from torch.utils.data import Dataset, DataLoader

dset_dir = '/home/feet/Documents/LAWN/datasets/Webcams'

class_names = ('1.0', '1.25', '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0')
# class_names = ('1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0')
# {2.5:100, 3.0:100, 4.0:100, 5.0:100, 6.0:100, 7.0:100, 8.0:100, 9.0:100, 10.0:100}
# {1.0:200, 1.25:200, 1.5:200, 1.75:200, 2.0:200, 2.25:200, 2.5:200, 3.0:200, 4.0:200, 5.0:200, 6.0:200, 7.0:200, 8.0:200, 9.0:200, 10.0:200}

dataset = dsets.Webcams.Webcams_cls(dset_dir, limits={})
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
    print(count, end=' ')
print('')

# 1.0: 135, 2.0: 200, 3.0: 263, 4.0: 254, 5.0: 313, 6.0: 382, 7.0: 450, 8.0: 407, 9.0: 7648, 10.0: 2193