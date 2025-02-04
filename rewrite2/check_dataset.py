import torch
import torch.nn as nn
import dsets
import dsets.WebcamSSFCombo
import dsets.Webcams
from torch.utils.data import Dataset, DataLoader
import shutil
import os
import os.path as path

dset_dir1 = '/home/feet/Documents/LAWN/datasets/SSF'
dset_dir2 = '/home/feet/Documents/LAWN/datasets/quality-labeled-webcams/by-network/good'

class_names = ('1.0', '1.25', '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0')
# class_names = ('1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0')
# {2.5:100, 3.0:100, 4.0:100, 5.0:100, 6.0:100, 7.0:100, 8.0:100, 9.0:100, 10.0:100}
# {1.0:200, 1.25:200, 1.5:200, 1.75:200, 2.0:200, 2.25:200, 2.5:200, 3.0:200, 4.0:200, 5.0:200, 6.0:200, 7.0:200, 8.0:200, 9.0:200, 10.0:200}

# dataset = dsets.Webcams.Webcams_cls_10_full(dset_dir2, transformer=lambda x:x, limits={1.5:220, 1.75:210, 2.0:210, 2.5:210, 3.0:630, 4.0:630, 5.0:630, 6.0:630, 7.0:630, 8.0:630, 9.0:630, 10.0:630})
# dataset = dsets.Webcams.Webcams_cls(dset_dir2, transformer=lambda x:x)
# dataset = dsets.WebcamSSFCombo.WebcamsSSF_cls_10((dset_dir2, dset_dir1), transformer=lambda x:x, limits=({1.5:220, 1.75:210, 2.0:210, 2.5:210, 3.0:630, 4.0:630, 5.0:630, 6.0:630, 7.0:630, 8.0:630, 9.0:630, 10.0:630}, {1.0:300, 2.0:300, 3.0:300, 4.0:300, 5.0:300, 6.0:300, 7.0:300, 8.0:300, 9.0:300, 10.0:300}))
dataset = dsets.Webcams.Webcams_cls_10(dset_dir2, transformer=lambda x:x, limits={1.0:265, 1.25:256, 1.5:0, 1.75:250, 2.0:250, 2.25:57, 2.5:0, 3.0:520, 4.0:520, 5.0:520, 6.0:520, 7.0:520, 8.0:520, 9.0:520, 10.0:520})
# dataset = dsets.SSF.SSF_cls_10(dset_dir, lambda x:x, limits={1.0:300, 2.0:300, 3.0:300, 4.0:300, 5.0:300, 6.0:300, 7.0:300, 8.0:300, 9.0:300, 10.0:300})
dataloader = DataLoader(dataset, 1, False)

class_counts = dict()

for i, (data, label, _) in enumerate(dataloader):
    argmax = torch.argmax(label).item()
    if argmax in class_counts:
        class_counts[argmax] += 1
    else:
        class_counts[argmax] = 1

_, counts_list = zip(*sorted(class_counts.items(), key=lambda x: x[0]))

counts_list = list(zip(class_names, counts_list))

for name, count in counts_list:
    print(count, end=', ')
print('')

copy_folder = '/home/feet/Documents/LAWN/datasets/Webcams-sample'
for file in dataset.files:
    parent_folder = path.basename(path.dirname(file))
    if not path.isdir(copy_folder + '/' + parent_folder):
        os.makedirs(copy_folder + '/' + parent_folder)
    
    shutil.copy(file, copy_folder + '/' + parent_folder + '/')
# 1.0: 135, 2.0: 200, 3.0: 263, 4.0: 254, 5.0: 313, 6.0: 382, 7.0: 450, 8.0: 407, 9.0: 7648, 10.0: 2193