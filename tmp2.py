import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import os
import torchvision.io as io
import torchvision.transforms.functional as f
import torchvision
from random import Random
from math import ceil

image_paths = list(glob('/home/feet/Documents/LAWN/datasets/WebcamsAlreadyCropped/**/*.png', recursive=True))

# for image_path in image_paths:
#     data = io.read_image(image_path, io.ImageReadMode.RGB)/255
#     #Remove 12.81% top, 3 bottom, 3 left, 3 right
#     crop_top = ceil(0.1281 * data.size(1))
#     crop_bot = 4
#     sub_vert = crop_top + crop_bot
#     dims = (data.size(1)-sub_vert, data.size(2)-6)
#     data = f.crop(data, crop_top, 3, dims[0], dims[1])
#     torchvision.utils.save_image(data, image_path.replace('Webcams', 'WebcamsAlreadyCropped'))

for image_path in image_paths:
    data = io.read_image(image_path, io.ImageReadMode.RGB)
    if data.size(1) < 300:
        os.system('rm ' + image_path)