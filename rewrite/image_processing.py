import torch
import torchvision.transforms.functional as tff
import torchvision.transforms as tf
import math
import random

def resize_crop(image: torch.Tensor, dim, rand_crop=False):
    if rand_crop:
        image = tf.RandomResizedCrop(dim, (1,1), (dim[0]/dim[1], dim[0]/dim[1]))(image)
    else:
        # height / width
        target_ratio = dim[0] / dim[1]
        ratio = image.size(1) / image.size(2)
        
        #if the current ratio is taller crop top and bottom, else crop sides
        if ratio > target_ratio:
            crop = (round(target_ratio*image.size(2)), image.size(2))
        else:
            crop = (image.size(1), round(image.size(1)/target_ratio))

        image = tff.center_crop(image, crop)
        kernel_size = math.floor(crop[0] / dim[0])
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if kernel_size > 1:
            image = tff.gaussian_blur(image, kernel_size)
            image = tff.resize(image, dim, tff.InterpolationMode.NEAREST, antialias=False)
        else:
            image = tff.resize(image, dim, tff.InterpolationMode.BILINEAR, antialias=False)

    return image

def random_augment(image):
    # flip, color, rotate, noise?, blur?
    transform_list = [tf.RandomHorizontalFlip(0.5)]
    sharpness_factor = random.uniform(0.75, 1.25)
    transform_list += [tf.RandomAdjustSharpness(sharpness_factor, 0.5)]
    transform_list += [tf.ColorJitter(0.25, 0.2, 0.2, 0.1)]
    transform_list += [tf.RandomRotation(10)]
    
    transforms = tf.RandomApply(torch.nn.ModuleList(transform_list), 0.75)
    
    return transforms(image)