import torch
import torch.nn as nn
import datasets
import models

CONFIG = {
    'model module': models.VisNet,
    'dimensions': (120,160),
    'classes': 1,
    'channels': 3,
    'split': 0.75, 
    'batch size': 32,
    'cuda': True,
    'loss function': nn.SmoothL1Loss(),
    'optimizer': torch.optim.Adam,
    'optim params': {
      'lr': 0.00001,  
    },
    'scheduler': None,
    'scheduler params': {
    
    },
    'epochs': 80,
    'dataset': datasets.SSF.SSF_reg,
    'dataset path': '/home/feet/Documents/LAWN/datasets/SSF'
}