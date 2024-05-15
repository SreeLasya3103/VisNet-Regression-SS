import torch
import torch.nn as nn
import torchvision.transforms as tf
 
class Model(nn.Module):
    def __init__(self, num_classes, num_channels, mean, std):
        super(Model, self).__init__()
        
        self.normalize = tf.Normalize(mean, std)

        def conv_1(): 
            return [nn.Conv2d(num_channels, 32, 1),
                    nn.Conv2d(32, 32, 3),
                    nn.MaxPool2d(2, 2)]
        
        def conv_2(): 
            return [nn.Conv2d(32, 64, 1),
                    nn.Conv2d(64, 64, 3),
                    nn.MaxPool2d(2, 2)]
        
        def conv_3(): 
            return [nn.Conv2d(64, 128, 1),
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
                  nn.Linear(2048, num_classes)]
        
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
        x = self.normalize(self.mean, self.std)
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