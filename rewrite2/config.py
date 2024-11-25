import torch
import torch.nn as nn
from dsets import *
from models import *
from torch.optim import lr_scheduler as sched
from pytorch_forecasting.metrics.point import MAPE

CONFIG = {
    'model module': VisNet,
    'existing model': None,
    # height x width
    'dimensions': (180,320),
    # number of classes to predict, use 1 for regression
    'num classes': 1,
    # used for generated figures
    'class names': ('1.0','2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0'),
    # 'class names': ('50','100','150','200','250','300','400'),
    # nunmber of color channels in image, usually either 1 or 3
    'num channels': 3,
    # percent split between training, validation, and test sets
    'splits': (0.70, 0.15, 0.15),
    # batch size = subbatch size * accum steps. Batches are split up into smaller batches when there is not enough memory for an entire batch
    'subbatch size': 16,
    'subbatch count': 2, 
    # whether or not to use the GPU
    'cuda': True,
    # instance of loss function to be used in training
    #'loss function': nn.SmoothL1Loss(),
    # 'loss function': nn.KLDivLoss(reduction='batchmean'),  # Using KLDivLoss
    'loss function': MAPE(),
    # optimizer class to be used in training. NOT an instnace of the optimizer
    'optimizer class': torch.optim.Adam,
    # parameters to be used to create instance of above optimizer class
    'optimizer parameters': {
      'lr': 0.00001,
      'weight_decay': 0.00
    },
    # scheduler is currently unused
    'scheduler': None,
    'scheduler params': {

    },
    # number of epochs to train for
    'epochs': 120,
    # dataset class. NOT an instance
    'dataset class': Jacobs.Jacobs,
    # parameters for constructing the dataset. Dependent on what dataset is being used.
    ## In the case of the webcams dataset, the limits are the maximum number of images from each class to be used.
    'dataset parameters': {
      # 'limits': {1.0:400, 1.25:400, 1.5:0, 1.75:200, 2.0:200, 2.25:999, 2.5:0, 3.0:400, 4.0:400, 5.0:400, 6.0:400, 7.0:400, 8.0:400, 9.0:400, 10.0:400}
      # 'limits': {1.0:999, 1.25:999,10.0:500}
      # 'limits': {1.0:212, 1.25:0, 1.5:0, 1.75:0, 2.0:400, 2.25:0, 2.5:0, 3.0:400, 4.0:400, 5.0:400, 6.0:400, 7.0:400, 8.0:400, 9.0:400, 10.0:400}
      # 'limits': {1.0:200, 2.0:200, 3.0:200, 4.0:200, 5.0:200, 6.0:200, 7.0:200, 8.0:200, 9.0:200, 10.0:200}
      'max_images': 5000
    },
    # path to the folder containing the images
    'dataset path': '/home/feet/Documents/LAWN/datasets/jacobs/linear',
    # whether or not to apply random augmentation to images to effectively increase size of the training set
    'augment': False,
    'normalize': True,
    'num workers': 4,
    'output function': None,
    'label function': None
}