import torch
import torch.nn as nn
import torch.nn.functional as f

IMG_SIZE = (256, 256)
NUM_CLASSES = 3
NUM_CHANNELS = 3

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
        
        self.model = nn.Sequential(*model);

    def forward(self, x):
        return x + self.model(x)


class RMEP(nn.Module):
    def __init__(self):
        super(RMEP, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(NUM_CHANNELS, 64, 7),
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
        
        model += [nn.AdaptiveMaxPool2d((2,2))]

        model += [nn.Conv2d(256, 128, 3, 1, 1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(128, 64, 3, 1, 1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(True)]
        
        model += [nn.Flatten(), nn.LazyLinear(NUM_CLASSES)]

        self.model = nn.Sequential(*model);
    
    def forward(self, x):
        return self.model(x)

def create_and_save():
    net = RMEP()
    net.eval()
    net(torch.rand((1, NUM_CHANNELS, IMG_SIZE[0], IMG_SIZE[1])))
    m = torch.jit.trace(net, torch.rand((1, NUM_CHANNELS, IMG_SIZE[0], IMG_SIZE[1])))
    m.save('RMEP-' + str(NUM_CHANNELS) + 'x' + str(IMG_SIZE[1]) + 'x' + str(IMG_SIZE[0]) + '-' + str(NUM_CLASSES) + '.pt')
    