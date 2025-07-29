import torch
import torch.nn as nn
from dsets import *
from models import *
from torch.optim import lr_scheduler as sched
from pytorch_forecasting.metrics.point import MAPE, SMAPE
from torchmetrics.regression import RelativeSquaredError as RSE

#Train KAN on 5. Keep convolution.
#Collage

CONFIG = {
    'model module': VisNet,
    'existing model': None,
    'test only': False,
    # height x width
    'dimensions': (310,470),
    # number of classes to predict, use 1 for regression
    'num classes': 1,
    'buckets': None,
    # 'buckets': ((-999999,3.5), (3.5, 7.5), (7.5, 999999)),
    # used for generated figures
    'class names': ('1.0','2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0'),
    # 'class names': ('[0,3)', '[3,5)', '[5,10]'),
    # 'class names': ('1-2.5', '3-4', '5-6', '7-8', '9-10'),
    # 'class names': ('1.0','1.25', '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0'),
    # 'class names': ('50','100','150','200','250','300','400'),
    # nunmber of color channels in image, usually either 1 or 3
    'num channels': 3,
    # percent split between training, validation, and test sets
    'splits': (0.70, 0.15, 0.15),
    # batch size = subbatch size * accum steps. Batches are split up into smaller batches when there is not enough memory for an entire batch
    'subbatch size': 8,
    'subbatch count': 4, 
    # whether or not to use the GPU
    'cuda': True,
    # instance of loss function to be used in training
    #'loss function': nn.SmoothL1Loss(),
    # 'loss function': nn.KLDivLoss(reduction='batchmean'),  # Using KLDivLoss
    'loss function': nn.SmoothL1Loss(),
    # optimizer class to be used in training. NOT an instnace of the optimizer
    'optimizer class': torch.optim.Adam,
    # parameters to be used to create instance of above optimizer class
    'optimizer parameters': {
      'lr': 0.0000005,
      'weight_decay': 0.00
    },
    # scheduler is currently unused
    'scheduler': None,
    'scheduler params': {

    },
    # number of epochs to train for
    'epochs': 80,
    # dataset class. NOT an instance
    'dataset class': Webcams.Webcams_reg,
    # parameters for constructing the dataset. Dependent on what dataset is being used.
    ## In the case of the webcams dataset, the limits are the maximum number of images from each class to be used.
    'dataset parameters': {
      # 'limits': {1.0:300, 2.0:300, 3.0:300, 4.0:300, 5.0:300, 6.0:300, 7.0:300, 8.0:300, 9.0:300, 10.0:300},

      # 'limits': {1.0:220, 1.25:220, 1.5:220, 1.75:220, 2.0:220, 2.5:220, 3.0:660, 4.0:660, 5.0:220, 6.0:220, 7.0:220, 8.0:220, 9.0:220, 10.0:220}
      # 'limits': {1.75:220, 2.0:220, 3.0:440, 4.0:440, 5.0:440, 6.0:440, 7.0:440, 8.0:440, 9.0:440, 10.0:440}
      # 'limits': {1.0:220, 1.25:220, 1.5:220, 1.75:220, 2.0:220, 2.5:220, 3.0:650, 4.0:650, 5.0:650, 6.0:650, 7.0:650, 8.0:650, 9.0:650, 10.0:650}
      # 'limits': {1.0:220, 1.25:220, 1.5:220, 1.75:220, 2.0:220, 2.5:220, 3.0:220, 4.0:385, 5.0:385, 6.0:385, 7.0:385, 8.0:515, 9.0:515, 10.0:515}

      # 'limits': ({1.5:220, 1.75:210, 2.0:210, 2.5:210, 3.0:630, 4.0:630, 5.0:630, 6.0:630, 7.0:630, 8.0:630, 9.0:630, 10.0:630}, {1.0:300, 2.0:300, 3.0:300, 4.0:300, 5.0:300, 6.0:300, 7.0:300, 8.0:300, 9.0:300, 10.0:300})

      'limits': {1.0:265, 1.25:256, 1.5:0, 1.75:250, 2.0:250, 2.25:57, 2.5:0, 3.0:520, 4.0:520, 5.0:520, 6.0:520, 7.0:520, 8.0:520, 9.0:520, 10.0:520}

      # 'max_images': 5000
    },
    # path to the folder containing the images
    #'dataset path': "D:\\Research\\NewGoodOnlyWebcams",
    'dataset path': "D:\\research\\VEIA",
    # whether or not to apply random augmentation to images to effectively increase size of the training set
    'augment': True,
    'normalize': True,
    'num workers': 0,
    'output function': None,
    'label function': None
}