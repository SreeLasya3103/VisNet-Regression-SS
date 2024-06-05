import torch
import torch.nn as nn
import image_processing as ip
import matplotlib
import math
import torchvision.transforms as tf

PC_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#000000', '#3F003F', '#7E007E',
                                                                   '#4300BD', '#0300FD', '#003F82',
                                                                   '#007D05', '#7CBE00', '#FBFE00',
                                                                   '#FF7F00', '#FF0500'])

class Model(nn.Module):
    def __init__(self, num_classes, num_channels, mean, std):
        super(Model, self).__init__()
 
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
        
        linear = [nn.Linear(3072, 4096),
                  nn.Linear(4096, num_classes)]
        
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

def highpass_filter(img, mask_radius=0.1):
    orig_dim = (img.size(1), img.size(2))
    img = torch.fft.rfft2(img)
    img = torch.fft.fftshift(img)
    
    center = ((img.size(1)-1)/2, (img.size(2)-1)/2)
    
    for h, row in enumerate(img[0]):
        for w, p in enumerate(row):
            h_dist = abs(h-center[0]) / ((img.size(1)-1)/2)
            w_dist = abs(w-center[1]) / ((img.size(2)-1)/2)
            distance = math.sqrt(h_dist**2 + w_dist**2)
            
            if distance < mask_radius:
                img[0][h][w] *= (distance/mask_radius)**8
    
    img = torch.fft.ifftshift(img)    
    img = torch.fft.irfft2(img, orig_dim)
    img = img.type(torch.float32)
    
    img = torch.clamp(img, 0.0, 1.0)
    
    return img

def get_tf_function(dim):
    def transform(img, agmnt=False):
        if agmnt:
            img = ip.random_augment(img)
        img = ip.resize_crop(img, dim, agmnt)
        
        pc = torch.from_numpy(PC_CMAP(img[2].unsqueeze(0))).permute((0,3,1,2))
        pc = torch.stack((pc[0][0], pc[0][1], pc[0][2]))
        
        fft = img[2].detach().clone()
        fft = torch.unsqueeze(fft, 0)
        fft = highpass_filter(fft)
        fft = torch.from_numpy(PC_CMAP(fft)).permute((0,3,1,2))
        fft = torch.stack((fft[0][0], fft[0][1], fft[0][2]))
        
        stack = torch.stack((img,pc,fft))
        
        return stack
    
    return transform