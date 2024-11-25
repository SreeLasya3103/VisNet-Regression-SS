import torch
import torch.nn as nn
from dsets import *
from models import *
from torch.optim import lr_scheduler as sched
from simloss import SimLoss

CONFIG = {
    'model module': VisNet,
    # height x width
    'dimensions': (60,140),
    # number of classes to predict, use 1 for regression
    'classes': 7,
    # used for generated figures
    # 'class names': ('1.0','2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0'),
    'class names': ('50','100','150','200','250','300','400'),
    # nunmber of color channels in image, usually either 1 or 3
    'channels': 3,
    # percent split between training, validation, and test sets
    'split': (0.70, 0.20, 0.10),
    # batch size = subbatch size * accum steps. Batches are split up into smaller batches when there is not enough memory for an entire batch
    'subbatch size': 16,
    'accum steps': 1, 
    # whether or not to use the GPU
    'cuda': True,
    # instance of loss function to be used in training
    #'loss function': nn.SmoothL1Loss(),
    # 'loss function': nn.KLDivLoss(reduction='batchmean'),  # Using KLDivLoss
    'loss function': nn.CrossEntropyLoss(),
    # optimizer class to be used in training. NOT an instnace of the optimizer
    'optimizer': torch.optim.Adam,
    # parameters to be used to create instance of above optimizer class
    'optim params': {
      'lr': 0.00001,
      'weight_decay': 0.00
    },
    # scheduler is currently unused
    'scheduler': None,
    'scheduler params': {

    },
    # number of epochs to train for
    'epochs': 80,
    # dataset class. NOT an instance
    'dataset': FROSI.FROSI,
    # parameters for constructing the dataset. Dependent on what dataset is being used.
    ## In the case of the webcams dataset, the limits are the maximum number of images from each class to be used.
    'dataset params': {
      # 'limits': {1.0:400, 1.25:400, 1.5:0, 1.75:200, 2.0:200, 2.25:999, 2.5:0, 3.0:400, 4.0:400, 5.0:400, 6.0:400, 7.0:400, 8.0:400, 9.0:400, 10.0:400}
      # 'limits': {1.0:999, 1.25:999,10.0:500}
      # 'limits': {1.0:400, 1.25:400, 1.5:0, 1.75:200, 2.0:200, 2.25:999, 2.5:0, 3.0:400, 4.0:400, 5.0:400, 6.0:400, 7.0:400, 8.0:400, 9.0:400, 10.0:400}
    },
    # name is just used in recording data
    'dataset name': 'FROSI downscaled',
    # path to the folder containing the images
    'dataset path': '/home/feet/Documents/LAWN/datasets/FROSI',
    # whether or not to apply random augmentation to images to effectively increase size of the training set
    'augment': False,
    # make each batch have an even number of each class. Ignore this
    'balance batches': False
}