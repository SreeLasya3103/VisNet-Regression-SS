import matplotlib.pyplot
import torch
import torch.nn as nn
import matplotlib
import math
import torchvision.transforms as tf
import functools
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
from math import ceil

class Model(nn.Module):
    def __init__(self, num_classes, num_channels, mean, std):
        super(Model, self).__init__()
        
        # self.register_buffer('mean', mean)
        # self.register_buffer('std', std)
        self.normalize = tf.Normalize(mean, std)

        self.model = tv.models.resnet50(num_classes=num_classes)

    def forward(self, x):
        x = self.normalize(x)
        
        return self.model(x)


def get_tf_function():
    def transform(img):
        return img
    
    return transform