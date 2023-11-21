import torch
import tomli
import sys
import os
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as vfunc

def resize_crop(img, img_dim):
    target_ratio = img_dim[0] / img_dim[1]
    ratio = img.size(1) / img.size(2)
    
    if ratio > target_ratio:
        img = vfunc.center_crop(img, (round(img.size(2)*target_ratio), img.size(2)))
    elif ratio < target_ratio:
        img = vfunc.center_crop(img, (img.size(1), round(img.size(1)/target_ratio)))
    
    img = vfunc.resize(img, img_dim, vfunc.InterpolationMode.BICUBIC, antialias=False)

    return img

f = open('config.toml', 'rb')
config = tomli.load(f)
img_dim = config['imgDim']

with torch.inference_mode():
    model = torch.jit.load(config['modelPath'], 'cpu')
    
    imagePath = input("Image path: ")
    rotation = input("Rotation in degrees: ")
    if rotation == '':
        rotation = '0'

    orig = Image.open(imagePath).convert('RGB').rotate(int(rotation))
    orig = transforms.PILToTensor()(orig)
    orig = resize_crop(orig, img_dim) / 255


    data = orig.view((1, 1, -1, *img_dim))



    data = data[0]

    output = model(data.float())

    if config['numClasses'] == 1:
        print(output.item())
    else:
        print(config['classes'][torch.argmax(output).item()])
    