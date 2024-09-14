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

LIMITS = {1.0:0, 1.25:0, 1.5:0, 1.75:0, 2.0:0, 2.25:0, 2.5:0, 3.0:0, 4.0:0, 5.0:0, 6.0:0, 7.0:0, 8.0:0, 9.0:0, 10.0:2000}

w = 39
h = 39
which = 0

dsets = []

dsets.append(Webcams_cls('/home/feet/Documents/LAWN/datasets/Webcams', limits=LIMITS, transform=VisNet.get_tf_function((309,470))))

loaded_images = []

dl = torch.utils.data.DataLoader(dsets[0], 1, False, num_workers=4)
for img, _ in dl:
    loaded_images.append(img[0][which])
    
    
final_image = None

for y in range(h):
    row = loaded_images[y*w]
    for x in range(1, w):
        next_img = loaded_images[y*w + x]
        row = torch.cat((row, next_img), 2)
    
    if y == 0:
        final_image = row
    else:
        final_image = torch.cat((final_image, row), 1)

tv.utils.save_image(final_image, "/home/feet/Documents/result stuff/9-3-24/compositeclassimages/10MI-images.jpg")
    

# for i in range(10):
#     lim = LIMITS.copy()
#     lim[i+1.0] = 1000
    
#     dsets.append(Webcams_cls_10('/home/feet/Documents/LAWN/datasets/Webcams', limits=lim, transform=VisNet.get_tf_function((309,470))))

# loaded_images = [[] for _ in range(10)]

# for i in range(10):
#     dl = torch.utils.data.DataLoader(dsets[i], 1, False, num_workers=4)
#     for img, _ in dl:
#         # img = torch.permute(img[0], (1,0,2,3))
#         loaded_images[i].append(img[0][which])
        
# final_image = None

# for i in range(10):
#     for y in range(h):
#         row = loaded_images[i][y*w]
#         for x in range(1, w):
#             next_img = loaded_images[i][y*w + x]
#             row = torch.cat((row, next_img), 2)
        
#         if y == 0:
#             final_image = row
#         else:
#             final_image = torch.cat((final_image, row), 1)

#     final_image = final_image

#     # tv.utils.save_image(final_image, "../ClassSamples/ttt"+str(i)+".png")
#     tv.utils.save_image(final_image, "1MI-images.png")
