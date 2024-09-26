import matplotlib.pyplot
import torch
import torch.nn as nn
import image_processing as ip
import matplotlib
import math
import torchvision.transforms as tf
import functools
import numpy as np
import matplotlib.pyplot as plt

PC_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#000000', '#3F003F', '#7E007E',
                                                                   '#4300BD', '#0300FD', '#003F82',
                                                                   '#007D05', '#7CBE00', '#FBFE00',
                                                                   '#FF7F00', '#FF0500'])
matplotlib.colors.LinearSegmentedColormap

class KAN(nn.Module):
    def __init__(self, input_size, inner_size, out_size):
        super(KAN, self).__init__()

        self.inner_weights = nn.Linear(input_size, inner_size)
        self.outer_weights = nn.Linear(inner_size, out_size)
        self.phi = nn.Tanh()

    def forward(self, x):
        z = self.inner_weights(x)
        v = self.phi(z)
        y = self.outer_weights(v)

        return y

class Model(nn.Module):
    def __init__(self, num_classes, num_channels, mean, std):
        super(Model, self).__init__()
 
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.normalize = tf.Normalize(mean, std)
        
        def conv_1(): 
            return [nn.Conv2d(num_channels, 64, 1),
                    nn.Conv2d(64, 64, 3),
                    nn.MaxPool2d(2, 2)]
        
        def conv_2(): 
            return [nn.Conv2d(64, 128, 1),
                    nn.Conv2d(128, 128, 3),
                    nn.MaxPool2d(2, 2)]
        
        def conv_3(): 
            return [nn.Conv2d(128, 256, 1),
                    nn.Conv2d(256, 256, 3, 2),
                    nn.Conv2d(256, 256, 1),
                    nn.MaxPool2d(2, 2)]
        
        linear_fft = [nn.Flatten(),
                      nn.LazyLinear(1024),
                      nn.Dropout(0.4)]
        
        linear_pc_orig = [nn.Flatten(),
                          nn.LazyLinear(2048),
                          nn.Dropout(0.4)]
        
        # linear = [nn.Linear(3072, 4096),
        #           nn.Linear(4096, num_classes)]

        linear = [nn.Linear(3072, 4096),
                  KAN(4096, 32, num_classes)]
        
        self.fft_1 = nn.Sequential(*conv_1())
        self.fft_2 = nn.Sequential(*conv_2())
        self.fft_3 = nn.Sequential(*conv_3())
        
        self.pc_1 = nn.Sequential(*conv_1())
        self.pc_2 = nn.Sequential(*conv_2())
        self.pc_3 = nn.Sequential(*conv_3())
        
        self.orig_1 = nn.Sequential(*conv_1())
        self.orig_2 = nn.Sequential(*conv_2())
        self.orig_3 = nn.Sequential(*conv_3())
        
        self.linear_fft = nn.Sequential(*linear_fft)
        self.linear_pc_orig = nn.Sequential(*linear_pc_orig)
        self.linear = nn.Sequential(*linear)
        
    def forward(self, x):
        x = self.normalize(x)
        x = x.permute((1, 0, 2, 3, 4))
        
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

def create(img_dim, num_classes, num_channels):
    net = Model(num_classes, num_channels)
    net.eval()
    net(torch.rand((3, 1, num_channels, *img_dim)))
    
    return net

def create_and_save(img_dim, num_classes, num_channels):
    net = create(img_dim, num_classes, num_channels)
    m = torch.jit.script(net)
    m.save('VisNet-' + str(num_channels) + 'x' + str(img_dim[1]) + 'x' + str(img_dim[0]) + '-' + str(num_classes) + '.pt')

@functools.cache
def highpass_mask(mask_radius, dim):
    mask = torch.ones(dim, dtype=torch.float32)
    mask_radius = np.multiply(dim, mask_radius)
    center = ((dim[0]-1)/2, (dim[1]-1)/2)
    center_tl = np.subtract(np.floor(center), mask_radius).astype(int)
    center_br = np.add(np.ceil(center), mask_radius).astype(int)
    
    for h in range(center_tl[0], center_br[0]):
        for w in range(center_tl[1], center_br[1]):
            h_dist = abs(h-center[0]) / mask_radius[0]
            w_dist = abs(w-center[1]) / mask_radius[1]
            distance = math.sqrt(h_dist**2 + w_dist**2)
            distance = min(1.0, distance)
            mask[h][w] = distance**8
    
    return mask
    

def highpass_filter(img, mask_radius=0.1):
    orig_dim = (img.size(0), img.size(1))
    fft = torch.fft.fft2(img)
    fft = torch.fft.fftshift(fft)

    mask = highpass_mask(mask_radius, fft.shape)

    fft = fft*mask
    
    fft = torch.fft.ifftshift(fft)    
    fft = torch.fft.ifft2(fft, orig_dim)
    fft = fft.type(torch.float32)
    
    
    fft = torch.clamp(fft, 0.0, 1.0)
    
    return fft

def satmap(orig):
    img = orig.permute(1,2,0).contiguous()
    
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            mx = mn = img[h][w][0]
            if img[h][w][1] > mx:
                mx = img[h][w][1]
            elif img[h][w][1] < mn:
                mn = img[h][w][1]
            
            if img[h][w][2] > mx:
                mx = img[h][w][2]
            elif img[h][w][2] < mn:
                mn = img[h][w][2]
            
            orig[2][h][w] = 0.0
            if mx != 0.0:
                orig[2][h][w] = 1.0 - mn
    
    return orig

def get_tf_function(dim):
    def transform(img, agmnt=False):
        if agmnt:
            img = ip.random_augment(img)
        img = ip.resize_crop(img, dim, agmnt).unsqueeze(0)
        img = img.repeat(3, 1, 1, 1)
        
        img[1] = torch.from_numpy(PC_CMAP(img[1][2])).permute((2,0,1))[:3,:,:]
        # img[0] = torch.zeros((3, *dim))
        # img[1] = torch.zeros((3, *dim))
        # img[1] = torch.from_numpy(PC_CMAP(satmap(img[1])[2])).permute((2,0,1))[:3,:,:]
        # plt.imshow(img[1].permute(1,2,0))
        # plt.show()
        # img[1] = torch.zeros(img[0].shape, dtype=torch.float32)
        
        img[2][2] = highpass_filter(img[2][2], 0.05)
        img[2] = torch.from_numpy(PC_CMAP(img[2][2])).permute((2,0,1))[:3,:,:]
        # img[2] = highpass_filter(img[2][2], 0.05)
        
        return img
    
    return transform