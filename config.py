import torch
import torch.nn as nn
import dsets
import dsets.FROSI
import dsets.SSF
import dsets.Webcams
import models
import models.Minimum
import models.VisNet
from torch.optim import lr_scheduler as sched
import models.VisNetReduced
from simloss import SimLoss

CONFIG = {
    'model module': models.VisNet,
    # 'dimensions': (245,320),
    'dimensions': (280,280),
    'classes': 10,
    # 'class names': ('1-3', '4-7', '8-10'),
    # 'class names': ('1.0', '1.25', '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0'),
    'class names': ('1.0','2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0'),
    'channels': 3,
    'split': (0.70, 0.20, 0.10),
    'subbatch size': 16,
    'accum steps': 2, 
    'cuda': True,
    'loss function': SimLoss(10, 0.01, 'cuda'),
    'optimizer': torch.optim.Adam,
    'optim params': {
      'lr': 0.00001,
      'weight_decay': 0.00
    },
    'scheduler': None,
    'scheduler params': {

    },
    'epochs': 80,
    'dataset': dsets.Webcams.Webcams_cls_10,
    'dataset params': {
      # 'limits':{1.0:174, 1.5:0, 2.0:174, 2.5:0, 3.0:174, 4.0:174, 5.0:174, 6.0:174, 7.0:174, 8.0:174, 9.0:174, 10.0:174}
      # 'limits':{3.0: 1050, 7.0: 1050, 10.0: 1050},
      # 'limits':{1.0:194, 1.5:0, 2.0:194, 2.5:0, 3.0:194, 4.0:194, 5.0:194, 6.0:194, 7.0:194, 8.0:194, 9.0:194, 10.0:194}
      'limits': {1.0:200, 1.25:200, 1.5:200, 1.75:200, 2.0:200, 2.25:200, 2.5:200, 3.0:200, 4.0:200, 5.0:200, 6.0:200, 7.0:200, 8.0:200, 9.0:200, 10.0:200}
      # 'limits': {9.0: 700, 10:700}
      # 'site_or': 'SITE41_'
    },
    'dataset name': 'Webcams cls',
    'dataset path': '/home/feet/Documents/LAWN/datasets/Webcams',
    'augment': True,
    'balance batches': False
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