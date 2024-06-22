# https://github.com/konstantinkobs/SimLoss

import torch
import numpy as np


class SimLoss(torch.nn.Module):
    def __init__(self, number_of_classes, reduction_factor, device="cpu", epsilon=1e-8):
        super().__init__()

        assert number_of_classes > 0

        self.__number_of_classes = number_of_classes
        self.__device = device
        self.epsilon = epsilon
        self.r = reduction_factor

    def forward(self, x, y):
        x = torch.softmax(x, 1)
        y = torch.argmax(y, 1)
        w = self.__w[y, :]
        return torch.mean(-torch.log(torch.sum(w * x, dim=1) + self.epsilon))

    @property
    def r(self):
        return self.__r
    
    @r.setter
    def r(self, r):
        assert r >= 0.0
        assert r < 1.0

        self.__r = r
        self.__w = self.__generate_w(self.__number_of_classes, self.__r, self.__device)

    def __generate_w(self, number_of_classes, reduction_factor, device):
        w = torch.zeros((number_of_classes, number_of_classes)).to(device)
        for j in range(number_of_classes):
            for i in range(number_of_classes):
                w[j, i] = reduction_factor ** np.abs(i - j)
                # w[j, i] = (reduction_factor ** (np.abs(i-j))) * ((j+1)/10)
        
        return w

    def __repr__(self):
        return "SimilarityBasedCrossEntropy"