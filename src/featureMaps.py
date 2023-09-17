import tomli
from os import path as os_path
from sys import path as sys_path
import torch
import torchvision.transforms.functional as vfunc
from PIL import Image
from torchvision import transforms
from matplotlib.colors import LinearSegmentedColormap
import torch
import tomli
import sys
import os
ROOT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import Integrated
import RMEP as rmep
import VisNet


import matplotlib.pyplot as plt

ROOT_DIR = os_path.dirname(__file__)
sys_path.append(os_path.join(ROOT_DIR, 'models'))

def resize_crop(img, img_dim):
    target_ratio = img_dim[0] / img_dim[1]
    ratio = img.size(1) / img.size(2)
    
    if ratio > target_ratio:
        img = vfunc.center_crop(img, (round(img.size(2)*target_ratio), img.size(2)))
    elif ratio < target_ratio:
        img = vfunc.center_crop(img, (img.size(1), round(img.size(1)/target_ratio)))
    
    img = vfunc.resize(img, img_dim, vfunc.InterpolationMode.BICUBIC, antialias=False)

    return img

def highpass_filter(img, mask_dim):
    orig_dim = (img.size(1), img.size(2))
    img = torch.fft.rfft2(img)
    img = torch.fft.fftshift(img)
    
    h_start = img.size(1)//2 - mask_dim[0]//2
    w_start = img.size(2)//2 - mask_dim[0]//2//2

    for i in range(h_start, h_start+mask_dim[0]):
        for j in range(w_start, w_start+mask_dim[1]//2):
            img[0][i][j] = 0

    img = torch.fft.ifftshift(img)    
    img = torch.fft.irfft2(img, orig_dim)
    
    return img

f = open('config.toml', 'rb')
config = tomli.load(f)

img_dim = config['imgDim']

model = torch.jit.load(config['modelPath'], torch.device('cpu'))
state = model.state_dict()
model = rmep.RMEP()
model.load_state_dict(state)

# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []# get all the model children as list
model_children = list(model.modules())#counter to keep count of the conv layers
counter = 0#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == torch.nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == torch.nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == torch.nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print("Total convolution layers:" + str(counter))
print("conv_layers")

image = Image.open('./example/8953_20130516_131311.jpg').convert('RGB')
image = transforms.PILToTensor()(image)
image = resize_crop(image, img_dim) / 255

outputs = []
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    processed.append(feature_map.data.cpu().numpy())

for i in range(len(processed)):
    print("Convolution " + str(i+1))
    fig = plt.figure(figsize=(64, 24))
    for j in range(len(processed[i])):
        a = fig.add_subplot(12, 24, j+1)
        imgplot = plt.imshow(processed[i][j])
        a.axis("off")
    plt.savefig('./featuremaps/conv'+str(i+1)+'.jpg', bbox_inches='tight')