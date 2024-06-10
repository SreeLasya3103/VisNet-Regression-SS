import torch
import torch.nn as nn
import torchvision.transforms as tf

class Model(nn.Module):
    def __init__(self, num_classes, num_channels, mean, std):
        super(Model, self).__init__()

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.normalize = tf.Normalize(mean, std)

        model = [nn.Conv2d(num_channels, 16, 3),
                 nn.InstanceNorm2d(16),
                 ]
        
        model += [nn.MaxPool2d((2,2))]
        
        model += [nn.Flatten(), nn.LazyLinear(num_classes)]

        self.model = nn.Sequential(*model);
    
    def forward(self, x):
        x = self.normalize(x)
        
        return self.model(x)

def create(img_dim, num_classes, num_channels):
    net = Model(num_classes, num_channels)
    net.eval()
    net(torch.rand((1, num_channels, img_dim[0], img_dim[1])))
    
    return net

def create_and_save(img_dim, num_classes, num_channels):
    net = create(img_dim, num_classes, num_channels)
    m = torch.jit.script(net)
    m.save('Minim-' + str(num_channels) + 'x' + str(img_dim[1]) + 'x' + str(img_dim[0]) + '-' + str(num_classes) + '.pt')