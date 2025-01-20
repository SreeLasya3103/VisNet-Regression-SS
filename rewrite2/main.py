import torch
import torch.utils.data as data
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from progress.bar import ChargingBar
from progress.spinner import Spinner
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
from torchvision.utils import save_image
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms as tf
import image_cropping

#Import the config.py file where settings for training are
spec = importlib.util.spec_from_file_location("config", os.path.join(os.getcwd(), 'config.py'))
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)
CONFIG = config.CONFIG

DsetClass = CONFIG['dataset class']
dset_path = CONFIG['dataset path']
dims = CONFIG['dimensions']
dset_params = CONFIG['dataset parameters']
num_classes = CONFIG['num classes']
splits = CONFIG['splits']
augment = CONFIG['augment']
normalize = CONFIG['normalize']
num_workers = CONFIG['num workers']
subbatch_size = CONFIG['subbatch size']
subbatch_count = CONFIG['subbatch count']
model_module = CONFIG['model module']
ModelClass = model_module.Model
num_channels = CONFIG['num channels']
OptimizerClass = CONFIG['optimizer class']
optimizer_params = CONFIG['optimizer parameters']
use_cuda = CONFIG['cuda']
loss_fn = CONFIG['loss function']
epochs = CONFIG['epochs']
class_names = CONFIG['class names']
output_fn = CONFIG['output function']
labels_fn = CONFIG['label function']
existing_model = CONFIG['existing model']
test_only = CONFIG['test only']
buckets = CONFIG['buckets']

writer = SummaryWriter()

with open(os.path.join(os.getcwd(), 'config.py')) as config_file:
    config_string = config_file.read()
    writer.add_text('config', config_string)

print('Preparing dataset...')

model_custom_transform = model_module.get_tf_function()
resize_fn = image_cropping.get_resize_crop_fn(dims)

transformer = lambda x: model_custom_transform(resize_fn(x))

if augment:
    transform_list = [tf.RandomHorizontalFlip(0.5),
                      tf.RandomRotation(5)]
    augmenter = tf.Compose(transform_list)

    train_transformer = lambda x: model_custom_transform(augmenter(resize_fn(x)))
else:
    train_transformer = transformer

dset = DsetClass(dset_path, transformer, **dset_params)

if num_classes > 1:
    class_lists = ([[] for _ in range(num_classes)], [[] for _ in range(num_classes)])

    for i in range(len(dset.files)):
        class_idx = dset.labels[i].argmax()
        class_lists[0][class_idx].append(dset.files[i])
        class_lists[1][class_idx].append(dset.labels[i])

    train_files = []
    val_files = []
    test_files = []

    train_labels = []
    val_labels = []
    test_labels = []

    for i in range(len(class_lists[0])):
        train_point = int(splits[0] * len(class_lists[0][i]))
        val_point = int(splits[1] * len(class_lists[0][i]) + train_point)
        
        file_sets = np.split(class_lists[0][i], (train_point, val_point))
        label_sets = np.split(class_lists[1][i], (train_point, val_point))

        train_files += file_sets[0].tolist()
        val_files += file_sets[1].tolist()
        test_files += file_sets[2].tolist()

        train_labels += label_sets[0].tolist()
        val_labels += label_sets[1].tolist()
        test_labels += label_sets[2].tolist()

elif num_classes == 1:
    train_point = int(splits[0] * len(dset))
    val_point = int(splits[1] * len(dset) + train_point)
        
    file_sets = np.split(dset.files, (train_point, val_point))
    label_sets = np.split(dset.labels, (train_point, val_point))

    train_files = file_sets[0].tolist()
    val_files = file_sets[1].tolist()
    test_files = file_sets[2].tolist()

    train_labels = label_sets[0].tolist()
    val_labels = label_sets[1].tolist()
    test_labels = label_sets[2].tolist()

train_set = DsetClass((train_files, train_labels), train_transformer)
val_set = DsetClass((val_files, val_labels), transformer)
test_set = DsetClass((test_files, test_labels), transformer)

pin_device = 'cuda' if use_cuda else 'cpu'
train_loader = DataLoader(train_set, subbatch_size, True, num_workers=num_workers, pin_memory=True, pin_memory_device=pin_device)
val_loader = DataLoader(val_set, subbatch_size, True, num_workers=num_workers, pin_memory=True, pin_memory_device=pin_device)
test_loader = DataLoader(test_set, subbatch_size, True, num_workers=num_workers, pin_memory=True, pin_memory_device=pin_device)

loaders = (train_loader, val_loader, test_loader)

sample = train_set.__getitem__(0)[0]
mean = torch.zeros(sample.size(), dtype=torch.float32)
std = torch.ones(sample.size(), dtype=torch.float32)

if existing_model is None:
    if normalize:
        print('Calculating mean and standard deviation...')

        variance = torch.zeros(sample.size(), dtype=torch.float32)

        loader = DataLoader(train_set, subbatch_size, num_workers=num_workers)

        bar = ChargingBar()
        bar.max = len(loader)
        bar.width = 0
        spinner = Spinner()

        for data, _, _ in loader:
            mean += torch.div(torch.sum(data, 0), len(train_set))
            variance += torch.div(torch.sum(torch.square(data-mean), dim=0), len(train_set)-1)

            bar.next()
            spinner.next()

        std = torch.sqrt(variance).apply_(lambda x: 1.0 if x == 0.0 else x)
        std[std==0.0] = 1.0

    model = ModelClass(num_classes, num_channels, mean, std).train()
    model(torch.unsqueeze(sample, 0))

else:
    model = ModelClass(num_classes, num_channels, mean, std)
    model(torch.unsqueeze(sample, 0))
    model.load_state_dict(torch.load(existing_model, weights_only=True, map_location=torch.device('cpu')))
    model.train()

if use_cuda:
    model.cuda()
else:
    model.cpu()

optimizer = OptimizerClass(model.parameters(), **optimizer_params)

if test_only:
    epochs = 1
    loaders = (None, loaders[1], loaders[2])

if num_classes > 1:
    tv.train_cls(loaders, model, optimizer, loss_fn, epochs, use_cuda, subbatch_count, class_names, output_fn, labels_fn, writer)
elif num_classes == 1:
    tv.train_reg(loaders, model, optimizer, loss_fn, epochs, use_cuda, subbatch_count, output_fn, labels_fn, writer, buckets=buckets, class_names=class_names)
else:
    print('Number of classes must be > 0')
