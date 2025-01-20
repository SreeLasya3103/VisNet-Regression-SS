import torch
import torch.nn as nn
import torchvision.transforms as tf
import matplotlib

PC_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#0000ff', '#00ff00', '#ff0000', '#0000ff'])

class Xception(nn.Module):
    def __init__(self, num_channels):
        super(Xception, self).__init__()

        class EntryFlow(nn.Module):
            def __init__(self):
                super(EntryFlow, self).__init__()

                self.l0 = nn.Sequential(nn.Conv2d(num_channels, 32, 3, 2, 1), nn.ReLU(),
                                        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU())

                self.c0 = nn.Conv2d(64, 128, 1, 2)

                self.l1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, groups=64),
                                        nn.Conv2d(64, 128, 1), nn.ReLU(),
                                        nn.Conv2d(128, 128, 3, 1, 1, groups=128),
                                        nn.Conv2d(128, 128, 1), nn.MaxPool2d(3, 2, 1))
                
                self.c1 = nn.Conv2d(128, 256, 1, 2)

                self.l2 =nn.Sequential(nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1, groups=128),
                                       nn.Conv2d(128, 256, 1), nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, 1, 1, groups=256),
                                       nn.Conv2d(256, 256, 1), nn.MaxPool2d(3, 2, 1))
                
                self.c2 = nn.Conv2d(256, 728, 1, 2)

                self.l3 = nn.Sequential(nn.ReLU(), nn.Conv2d(256, 256, 3, 1, 1, groups=256),
                                        nn.Conv2d(256, 728, 1), nn.ReLU(),
                                        nn.Conv2d(728, 728, 3, 1, 1, groups=728),
                                        nn.Conv2d(728, 728, 1), nn.MaxPool2d(3, 2, 1))
            def forward(self, x):
                x = self.l0(x)
                c = self.c0(x)
                x = torch.add(self.l1(x), c)
                c = self.c1(c)
                x = torch.add(self.l2(x), c)
                c = self.c2(x)
                x = torch.add(self.l3(x), c)

                return x
        
        class MiddleFlow(nn.Module):
            def __init__(self):
                super(MiddleFlow, self).__init__()

                self.l0 = nn.Sequential(*nn.ModuleList([nn.Sequential(nn.ReLU(), nn.Conv2d(728, 728, 3, 1, 1, groups=728), nn.Conv2d(728, 728, 1)) for i in range(3)]))

            def forward(self, x):
                return torch.add(self.l0(x), x)
            
        class ExitFlow(nn.Module):
            def __init__(self):
                super(ExitFlow, self).__init__()

                self.c0 = nn.Conv2d(728, 1024, 1, 2)

                self.l1 = nn.Sequential(nn.ReLU(), nn.Conv2d(728, 728, 3, 1, 1, groups=728),
                                        nn.Conv2d(728, 728, 1), nn.ReLU(),
                                        nn.Conv2d(728, 728, 3, 1, 1, groups=728),
                                        nn.Conv2d(728, 1024, 1), nn.MaxPool2d(3, 2, 1))
                
                self.l2 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, groups=1024),
                                        nn.Conv2d(1024, 1536, 1), nn.ReLU(),
                                        nn.Conv2d(1536, 1536, 3, 1, 1, groups=1536),
                                        nn.Conv2d(1536, 2048, 1), nn.ReLU())
                
            def forward(self, x):
                return self.l2(torch.add(self.c0(x), self.l1(x)))

        entry = EntryFlow()
        middle = nn.Sequential(*nn.ModuleList([MiddleFlow() for i in range(8)]))
        exit = ExitFlow()

        self.model = nn.Sequential(entry, middle, exit)

    def forward(self, x):
        return self.model(x)
    
class VGG(nn.Module):
    def __init__(self, num_channels):
        super(VGG, self).__init__()

        model = [nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(),
                 nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
                 nn.MaxPool2d(2, 2)]
        
        model += [nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
                  nn.MaxPool2d(2, 2)]
        
        model += [nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
                  nn.MaxPool2d(2, 2)]
        
        model += [nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
                  nn.MaxPool2d(2, 2)]
        
        model += [nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
                  nn.MaxPool2d(2, 2)]
        
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Model(nn.Module):
    def __init__(self, num_classes, num_channels, mean, std):
        super(Model, self).__init__()

        if mean is None or std is None:
            self.normalize = nn.Identity()
        else:
            self.normalize = tf.Normalize(mean, std)

        self.VGG = nn.Sequential(VGG(num_channels), nn.Flatten(), nn.LazyLinear(128))
        self.Xception = nn.Sequential(Xception(num_channels), nn.Flatten(), nn.LazyLinear(128))

        self.end = nn.Sequential(nn.Linear(256, 512), nn.Dropout(0.1), nn.Linear(512, num_classes))
    
    def forward(self, x):
        x = self.normalize(x)
        x = x.permute((1, 0, 2, 3, 4))
        
        orig = self.VGG(x[0])
        pc = self.Xception(x[1])
        return self.end(torch.cat((orig, pc), 1))
 
def get_tf_function():
    def transform(img):
        img = img.repeat(2, 1, 1, 1)

        img[1] = torch.from_numpy(PC_CMAP(img[1][2])).permute((2,0,1))[:3,:,:]
        
        return img
    
    return transform