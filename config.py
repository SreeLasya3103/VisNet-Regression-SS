import torch
import torch.nn as nn
import datasets
import datasets.FROSI
import models
import models.VisNet

#VisNet FROSI

CONFIG = {
    'model module': models.VisNets,
    'dimensions': (60,140),
    'classes': 7,
    'channels': 3,
    'split': (0.70, 0.10, 0.20),
    'batch size': 32,
    'cuda': True,
    'loss function': nn.CrossEntropyLoss(),
    'optimizer': torch.optim.Adam,
    'optim params': {
      'lr': 0.00001,
    },
    'scheduler': None,
    'scheduler params': {

    },
    'epochs': 10,
    'dataset': datasets.FROSI.FROSI,
    'dataset path': '/home/feet/Documents/LAWN/datasets/FROSI'
}
