import torch
import torchvision.transforms.functional as tff
import torchvision.transforms as tf
import math
import random

def get_resize_crop_fn(dim):
    def resize_crop(image):
        #height over width
        target_ratio = dim[0] / dim[1]
        ratio = image.size(1) / image.size(2)
        
        #if the the image is too tall, crop the top and bottom
        #otherwise crop sides
        if ratio > target_ratio:
            crop = (round(target_ratio*image.size(2)), image.size(2))
        else:
            crop = (image.size(1), round(image.size(1)/target_ratio))

        image = tff.center_crop(image, crop)
        image = tff.resize(image, dim)

        return image
    
    return resize_crop
