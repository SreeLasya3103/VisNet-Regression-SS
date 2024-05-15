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
from config import CONFIG

print('Preparing dataset...')
transformer = CONFIG['model module'].get_tf_function(CONFIG['dimensions'])
dset = CONFIG['dataset'](CONFIG['dataset path'], transformer)
train_set, val_set = data.random_split(dset, (CONFIG['split'], 1-CONFIG['split']))

sample = train_set.__getitem__(0)[0]
mean = torch.zeros(sample.size(), dtype=torch.float32)
std = torch.ones(sample.size(), dtype=torch.float32)

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
}

if CONFIG['classes'] > 1:
    tv.train_cls(train_set, val_set, model, params)
elif CONFIG['classes'] == 1:
    tv.train_reg(train_set, val_set, model, params)