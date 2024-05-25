import torch
import torch.utils.data as data
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from progress.bar import Bar
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as f
from torcheval.metrics.functional import r2_score
import train_val as tv
import datasets as datasets
import models as models
from progress.bar import Bar
import os
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("config", os.path.join(os.getcwd(), 'config.py'))
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)
CONFIG = config.CONFIG


print('Preparing dataset...')
transformer = lambda x: x
if hasattr(CONFIG['model module'], 'get_tf_function'):
    transformer = CONFIG['model module'].get_tf_function(CONFIG['dimensions'])
dset = CONFIG['dataset'](CONFIG['dataset path'], transformer)
train_set, val_set, test_set = data.random_split(dset, CONFIG['split'])

print('Calculating mean...')
sample = train_set.__getitem__(0)[0]
mean = torch.zeros(sample.size(), dtype=torch.float32)

train_loader = DataLoader(train_set, 64, False)

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
    'batch_size': CONFIG['batch size'],
    'use_cuda': CONFIG['cuda'],
    'loss_fn': CONFIG['loss function'],
    'scheduler': scheduler,
    'optimizer': optimizer,
    'epochs': CONFIG['epochs'],
    'num_classes': CONFIG['classes'],
    'num_channels': CONFIG['channels'],
    'learning_rate': CONFIG['optim params']['lr'],
    'model_name': CONFIG['model module'].__name__,
    'split': CONFIG['split'],
    'dset_name': CONFIG['dataset name']
}

if CONFIG['classes'] > 1:
    tv.train_cls(train_set, val_set, test_set, model, params)
elif CONFIG['classes'] == 1:
    tv.train_reg(train_set, val_set, test_set, model, params)