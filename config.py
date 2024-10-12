import torch
import torch.nn as nn
from dsets import Webcams
from models import RMEP
from torch.optim import lr_scheduler as sched
import models.VisNetReduced
from simloss import SimLoss

CONFIG = {
    'model module': RMEP,
    # height x width
    'dimensions': (128,128),
    # number of classes to predict, use 1 for regression
    'classes': 10,
    # used for generated figures
    'class names': ('1.0','2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0'),
    # nunmber of color channels in image, usually either 1 or 3
    'channels': 3,
    # percent split between training, validation, and test sets
    'split': (0.70, 0.20, 0.10),
    # batch size = subbatch size * accum steps. Batches are split up into smaller batches when there is not enough memory for an entire batch
    'subbatch size': 8,
    'accum steps': 2, 
    # whether or not to use the GPU
    'cuda': False,
    # instance of loss function to be used in training
    #'loss function': nn.SmoothL1Loss(),
    'loss function': nn.KLDivLoss(reduction='batchmean'),  # Using KLDivLoss
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
    'dataset': Webcams.Webcams_cls_10,
    # parameters for constructing the dataset. Dependent on what dataset is being used.
    ## In the case of the webcams dataset, the limits are the maximum number of images from each class to be used.
    'dataset params': {
      'limits': {1.0:10, 1.25:20, 1.5:0, 1.75:20, 2.0:20, 2.25:20, 2.5:0, 3.0:20, 4.0:20, 5.0:20, 6.0:20, 7.0:20, 8.0:20, 9.0:20, 10.0:20}
    },
    # name is just used in recording data
    'dataset name': 'Webcams cls 10',
    # path to the folder containing the images
    'dataset path': 'C:\\Users\\PC\\Desktop\\VisNet\\Visibility-Networks\\rewrite\\WebcamsSample',
    # whether or not to apply random augmentation to images to effectively increase size of the training set
    'augment': True,
    # make each batch have an even number of each class. Ignore this
    'balance batches': False
}