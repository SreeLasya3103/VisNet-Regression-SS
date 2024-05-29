import torch
import torch.nn as nn
import datasets
import datasets.FROSI
import datasets.SSF
import datasets.Webcams
import models
import models.Minimum
import models.VisNet
from torch.optim import lr_scheduler as sched

CONFIG = {
    'model module': models.VisNet,
    'dimensions': (350,410),
    'classes': 15,
    'channels': 3,
    'split': (0.75, 0.10, 0.15),
    'batch size': 32,
    'batch splits': 1, 
    'cuda': True,
    'loss function': nn.CrossEntropyLoss(),
    'optimizer': torch.optim.Adam,
    'optim params': {
      'lr': 0.00001,
    },
    'scheduler': None,
    'scheduler params': {

    },
    'epochs': 80,
    'dataset': datasets.Webcams.Webcams_cls,
    'dataset name': 'Webcams cls',
    'dataset path': '/home/jmurray/LAWN/datasets/Webcams'
}


# VisNet FROSI

# CONFIG = {
#     'model module': models.VisNet,
#     'dimensions': (60,140),
#     'classes': 7,
#     'channels': 3,
#     'split': (0.70, 0.10, 0.20),
#     'batch size': 32,
#     'cuda': True,
#     'loss function': nn.CrossEntropyLoss(),
#     'optimizer': torch.optim.Adam,
#     'optim params': {
#       'lr': 0.00001,
#     },
#     'scheduler': None,
#     'scheduler params': {

#     },
#     'epochs': 10,
#     'dataset': datasets.FROSI.FROSI,
#     'dataset name': 'FROSI',
#     'dataset path': '/home/feet/Documents/LAWN/datasets/FROSI'
# }

# #RMEP SSF

# def lr_decay(epoch):
#   if epoch < 50:
#     return 1.0
#   else:
#     return (150.0 - epoch) / 100.0

# CONFIG = {
#     'model module': models.RMEP,
#     'dimensions': (120,160),
#     'classes': 1,
#     'channels': 3,
#     'split': (0.64, 0.16, 0.20),
#     'batch size': 16,
#     'cuda': True,
#     'loss function': nn.SmoothL1Loss(),
#     'optimizer': torch.optim.Adam,
#     'optim params': {
#       'lr': .00001,
#     },
#     'scheduler': None,
#     'scheduler params': {
#       'lr_lambda': lr_decay
#     },
#     'epochs': 150,
#     'dataset': datasets.SSF.SSF_reg,
#     'dataset name': 'SSF',
#     'dataset path': '/home/feet/Documents/LAWN/datasets/SSF'
# }