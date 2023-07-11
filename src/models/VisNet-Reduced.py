import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
import copy
from progress.bar import Bar
import numpy as np
import matplotlib

#****************************
#*******HEIGHT FIRST*********
#****************************
IMG_SIZE = (60, 140)
NUM_CLASSES = 7
NUM_CHANNELS = 1

pc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#0000ff', '#00ff00', '#ff0000'])

def gray_fog_highlight(img):
    img = img.numpy()
            
    with np.nditer(img, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = (255 * 1.02**(x*255 - 255))/255
    
    img = np.concatenate((img, img, img))
    img = np.transpose(img, (1, 2, 0))
    img = np.array([img])
    
    return np.clip(img, 0, 1)

class VisNet(nn.Module):
    def __init__(self):
        super(VisNet, self).__init__()
        
        conv_1 = [nn.Conv2d(NUM_CHANNELS, 32, 1),
                  nn.Conv2d(32, 32, 3),
                  nn.MaxPool2d(2, 2)]
        
        conv_2 = [nn.Conv2d(32, 64, 1),
                  nn.Conv2d(64, 64, 3),
                  nn.MaxPool2d(2, 2)]
        
        conv_3 = [nn.Conv2d(64, 128, 1),
                  nn.Conv2d(128, 128, 3, 2),
                  nn.Conv2d(128, 128, 1),
                  nn.MaxPool2d(2, 2)]
        
        linear_fft = [nn.Flatten(),
                      nn.LazyLinear(512),
                      nn.Dropout(0.4)]
        
        linear_pc_orig = [nn.Flatten(),
                          nn.LazyLinear(1024),
                          nn.Dropout(0.4)]
        
        linear = [nn.Linear(1536, 2048),
                  nn.Linear(2048, NUM_CLASSES)]
        
        self.fft_1 = nn.Sequential(*copy.deepcopy(conv_1))
        self.fft_2 = nn.Sequential(*copy.deepcopy(conv_2))
        self.fft_3 = nn.Sequential(*copy.deepcopy(conv_3))
        
        self.pc_1 = nn.Sequential(*copy.deepcopy(conv_1))
        self.pc_2 = nn.Sequential(*copy.deepcopy(conv_2))
        self.pc_3 = nn.Sequential(*copy.deepcopy(conv_3))
        
        self.orig_1 = nn.Sequential(*copy.deepcopy(conv_1))
        self.orig_2 = nn.Sequential(*copy.deepcopy(conv_2))
        self.orig_3 = nn.Sequential(*copy.deepcopy(conv_3))
        
        self.linear_fft = nn.Sequential(*linear_fft)
        self.linear_pc_orig = nn.Sequential(*linear_pc_orig)
        self.linear = nn.Sequential(*linear)
        
    def forward(self, x):
        fft = self.fft_1(x[2])
        pc = self.pc_1(x[1])
        orig = self.orig_1(x[0])
        
        fft = torch.add(torch.add(pc, orig), fft)
        
        fft = self.fft_2(fft)
        pc = self.pc_2(pc)
        orig = self.orig_2(orig)
        
        fft = torch.add(torch.add(pc, orig), fft)
        
        fft = self.fft_3(fft)
        pc = self.pc_3(pc)
        orig = self.orig_3(orig)
        
        pc_orig = torch.add(pc, orig)
        
        fft = self.linear_fft(fft)
        pc_orig = self.linear_pc_orig(pc_orig)
        
        cat = torch.cat((fft, pc_orig), 1)
        return self.linear(cat)

def create_and_save():
    net = VisNet()
    net.eval()
    net(torch.rand((3, 1, NUM_CHANNELS, *IMG_SIZE)))
    m = torch.jit.trace(net, torch.rand((3, 1, NUM_CHANNELS, *IMG_SIZE)))
    m.save('VisNet_Reduced-' + str(NUM_CHANNELS) + 'x' + str(IMG_SIZE[1]) + 'x' + str(IMG_SIZE[0]) + '-' + str(NUM_CLASSES) + '.pt')
