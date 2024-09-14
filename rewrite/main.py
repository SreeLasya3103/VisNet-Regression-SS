import torch
import torch.utils.data as data
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from progress.bar import Bar
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as f
from torcheval.metrics.functional import r2_score
import train_val as tv
import dsets as dsets
import models as models
from progress.bar import Bar
import os
import importlib.util
import sys
import numpy as np
from random import Random
import matplotlib.pyplot as plt
from copy import deepcopy
import image_processing as ip
from torchvision.utils import save_image

spec = importlib.util.spec_from_file_location("config", os.path.join(os.getcwd(), 'config.py'))
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)
CONFIG = config.CONFIG


print('Preparing dataset...')
transformer = lambda x, agmnt: ip.resize_crop(x, CONFIG['dimensions'], agmnt)   #Function that processes images for use in a model
if hasattr(CONFIG['model module'], 'get_tf_function'):  #Use model's custom transformer if it has one 
    transformer = CONFIG['model module'].get_tf_function(CONFIG['dimensions'])
    
#Create dataset using all images    
dset = CONFIG['dataset'](CONFIG['dataset path'], transformer, **CONFIG['dataset params'])
# gen = torch.Generator()
# gen = gen.manual_seed(37)

#Create class lists for sorting out the images 
class_lists = [[] for _ in range(CONFIG['classes'])]

for i in range(0, dset.__len__()):
    class_idx = dset.labels[i].argmax()
    class_lists[class_idx].append(dset.files[i])

train_files = []
val_files = []
test_files = []

for i, c in enumerate(class_lists):
    train_point = CONFIG['split'][0] * len(c)
    val_point = CONFIG['split'][1] * len(c) + train_point
    splits = np.split(c, (int(train_point), int(val_point)))
    if CONFIG['balance batches']:
        class_lists[i] = splits[0].tolist()
    train_files += splits[0].tolist()
    val_files += splits[1].tolist()
    test_files += splits[2].tolist()

if CONFIG['balance batches']:
    train_set_bb = [CONFIG['dataset'](c, transformer, augment=CONFIG['augment']) for c in class_lists]
train_set = CONFIG['dataset'](train_files, transformer, augment=CONFIG['augment'])
    
val_set = CONFIG['dataset'](val_files, transformer)
test_set = CONFIG['dataset'](test_files, transformer)

# class_names = ('1.0', '1.25', '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0')
# for i in range(train_set.__len__()):
#     print(train_set.files[i])
#     data, label = train_set.__getitem__(i)
#     three = torch.cat((data[0], data[1], data[2]), 1)
#     # save_image(three, '/home/feet/Pictures/INPUT_COMPS/' + class_names[torch.argmax(label)] + '-' + str(i) + '.png')
#     plt.imshow(three.permute(1,2,0))
#     plt.show()

print('Calculating mean...')
sample = train_set.__getitem__(0)[0]
mean = torch.zeros(sample.size(), dtype=torch.float32)

train_loader = DataLoader(train_set, 32, False, num_workers=4)

bar = Bar()
bar.max = len(train_loader)

for i, (data, labels) in enumerate(train_loader):
    mean += torch.div(torch.mean(data, 0) * 64, train_set.__len__())
    bar.next()

print('\nCalculating std...')
bar = Bar()
bar.max = len(train_loader)

square_dif = torch.zeros(sample.size(), dtype=torch.float32)
    
for i, (data, labels) in enumerate(train_loader):
    square_dif += torch.div(torch.sum(torch.square(data - mean), dim=0), (train_set.__len__()-1))
    bar.next()

std = torch.sqrt(square_dif).apply_(lambda x: 1.0 if x == 0.0 else x)

# std = torch.ones(sample.size(), dtype=torch.float32)

model = CONFIG['model module'].Model(CONFIG['classes'], CONFIG['channels'], mean, std)

optimizer = CONFIG['optimizer'](model.parameters(), **CONFIG['optim params'])

scheduler = None
if CONFIG['scheduler']:
    scheduler = CONFIG['scheduler'](optimizer, **CONFIG['scheduler params'])

params = {
    'subbatch_size': CONFIG['subbatch size'],
    'accum_steps': CONFIG['accum steps'],
    'use_cuda': CONFIG['cuda'],
    'loss_fn': CONFIG['loss function'],
    'scheduler': scheduler,
    'optimizer': optimizer,
    'epochs': CONFIG['epochs'],
    'num_classes': CONFIG['classes'],
    'class_names': CONFIG['class names'],
    'num_channels': CONFIG['channels'],
    'learning_rate': CONFIG['optim params']['lr'],
    'model_name': CONFIG['model module'].__name__,
    'split': CONFIG['split'],
    'dset_name': CONFIG['dataset name'],
    'image_dim': CONFIG['dimensions']
}

if CONFIG['classes'] > 1:
    if CONFIG['balance batches']:
        tv.train_cls_bb(train_set_bb, val_set, test_set, model, params)
    else:
        tv.train_cls(train_set, val_set, test_set, model, params)
elif CONFIG['classes'] == 1:
    tv.train_reg(train_set, val_set, test_set, model, params)