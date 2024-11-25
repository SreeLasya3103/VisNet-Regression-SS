import matplotlib.pyplot
import torch
import torch.nn as nn
import matplotlib
import math
import torchvision.transforms as tf
import functools
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(256, 256, 3, 1, 0),
                 nn.InstanceNorm2d(256),
                 nn.ReLU(True)]
        
        model += [nn.Dropout(0.5)]

        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(256, 256, 3, 1, 0),
                  nn.InstanceNorm2d(256)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)

class Model(nn.Module):
    def __init__(self, num_classes, num_channels, mean, std):
        super(Model, self).__init__()
        
        # self.register_buffer('mean', mean)
        # self.register_buffer('std', std)
        self.normalize = tf.Normalize(mean, std)

        img_dim = (mean.size(1), mean.size(2))
        
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(num_channels, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True)]
        
        model += [nn.Conv2d(64, 128, 3, 2, 1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(128, 256, 3, 2, 1),
                  nn.InstanceNorm2d(256),
                  nn.ReLU(True)]
        
        model += [ResNet(), ResNet(), ResNet(), ResNet(), ResNet(), ResNet()]

        model += [nn.Conv2d(256, 256, 4, 2, 1),
                  nn.InstanceNorm2d(256),
                  nn.ReLU(True)]
        
        kernelSize = ( ceil(img_dim[0]/(2**4)), ceil(img_dim[1]/(2**4)))
        stride = ( int(img_dim[0]/(2**4)), int(img_dim[1]/(2**4)))

        model += [nn.MaxPool2d(kernelSize, stride)]

        model += [nn.Conv2d(256, 128, 3, 1, 1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(128, 64, 3, 1, 1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(True)]
        
        model += [nn.Flatten(), nn.LazyLinear(num_classes)]

        self.model = nn.Sequential(*model);
    
    def forward(self, x):
        x = self.normalize(x)
        
        return self.model(x)


def get_tf_function():
    def transform(img):
        return img
    
    return transform