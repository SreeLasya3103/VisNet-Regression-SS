import os

import torch.utils
import torch.utils.data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import sortedcontainers as sc
from dsets.Webcams import Webcams_cls_10, Webcams_cls
import sklearn_extra.cluster as skc
import matplotlib.pyplot as plt
import torchvision as tv
import torchvision.transforms.functional as tff
import torch
import pickle
from models import VisNet
import image_processing as ip
import shutil
import os.path

LIMITS = {1.0:0, 1.25:0, 1.5:0, 1.75:0, 2.0:0, 2.25:0, 2.5:0, 3.0:0, 4.0:0, 5.0:0, 6.0:0, 7.0:0, 8.0:0, 9.0:0, 10.0:0}
DIM = (280,280)
PASS_POINTS = (0.1, 0.05)

H = 15
W = 3

# dset = Webcams_cls_10('/home/feet/Documents/LAWN/datasets/Webcams', limits={1.0:75, 1.25:25, 1.5:0, 1.75:25, 2.0:50, 2.25:25, 2.5:0, 3.0:100, 4.0:100, 5.0:100, 6.0:100, 7.0:100, 8.0:100, 9.0:100, 10.0:100})
# for img in dset.files:
#     os.makedirs('/home/feet/Documents/LAWN/datasets/WebcamsSample/' + os.path.basename(os.path.dirname(img)), exist_ok=True)
#     shutil.copy(img, '/home/feet/Documents/LAWN/datasets/WebcamsSample/' + os.path.basename(os.path.dirname(img)) + '/' + os.path.basename(img))

transform to make image with original, each fft, and combined
def transform(img, augment):
    img = ip.resize_crop(img, DIM).unsqueeze(0)
    img = img.repeat(5, 1, 1, 1)

    img[1][0] = VisNet.pass_filter(img[1][2], VisNet.lowpass_mask(PASS_POINTS[1], img[1][2].shape))
    img[1][0] = (img[1][0] - torch.mean(img[1][0])) / torch.std(img[1][0])

    img[1][1] = VisNet.pass_filter(img[1][2], VisNet.bandpass_mask(PASS_POINTS, img[1][2].shape))
    img[1][1] = (img[1][1] - torch.mean(img[1][1])) / torch.std(img[1][1])
    
    img[1][2] = VisNet.pass_filter(img[1][2], VisNet.highpass_mask(PASS_POINTS[0], img[1][2].shape))
    img[1][2] = (img[1][2] - torch.mean(img[1][2])) / torch.std(img[1][2])

    img[2] = torch.zeros(img[0].shape)
    img[2][0] = img[1][0]

    img[3] = torch.zeros(img[0].shape)
    img[3][1] = img[1][1]

    img[4] = torch.zeros(img[0].shape)
    img[4][2] = img[1][2]

    final = torch.cat((img[0], img[1], img[2], img[3], img[4]), 2)

    return final

for i in range(1, 11):
    limits = LIMITS.copy()
    limits[float(i)] = H*W

    match i:
        case 1:
            limits[1.25] = H*W
        case 2:
            limits[1.75] = limits[2.25] = H*W

    dset = Webcams_cls_10('/home/feet/Documents/LAWN/datasets/quality-labeled-webcams', limits=limits, transform=transform)
    dl = torch.utils.data.DataLoader(dset, 1, False, num_workers=4)

    loaded_images = []
    for img, _ in dl:
        loaded_images.append(img[0])

    final_image = None

    for y in range(H):
        row = loaded_images[y*W]
        for x in range(1, W):
            next_img = loaded_images[y*W + x]
            row = torch.cat((row, next_img), 2)
        
        if y == 0:
            final_image = row
        else:
            final_image = torch.cat((final_image, row), 1)

    tv.utils.save_image(final_image, '/home/feet/Documents/LAWN/result-stuff/10-08-24/FFT_VIS/0.1_0.05/' + str(i) + 'MI.png')

# img = torch.zeros((3, *DIM))

# img[0] = VisNet.lowpass_mask(PASS_POINTS[1], img[0].shape)
# img[1] = VisNet.bandpass_mask(PASS_POINTS, img[0].shape)
# img[2] = VisNet.highpass_mask(PASS_POINTS[0], img[0].shape)

# tv.utils.save_image(img, '/home/feet/Documents/LAWN/result-stuff/10-08-24/FFT_VIS/0.1_0.05/mask.png')