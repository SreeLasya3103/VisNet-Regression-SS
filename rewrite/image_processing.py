import torch
import torchvision.transforms.functional as tff
import torchvision.transforms as tf
import math
import random

def resize_crop(image: torch.Tensor, dim, rand_crop=False):
    target_ratio = dim[0] / dim[1]
    ratio = image.size(1) / image.size(2)
    
    if rand_crop:
        #if the current ratio is taller crop top and bottom, else crop sides
        if ratio > target_ratio:
            crop = (round(target_ratio*image.size(2)), image.size(2))
            room = image.size(1) - crop[0]
            corner = random.randrange(room)
            image = tff.resized_crop(image, corner, 0, crop[0], crop[1], dim)
        else:
            crop = (image.size(1), round(image.size(1)/target_ratio))
            room = image.size(2) - crop[1]
            corner = random.randrange(room)
            image = tff.resized_crop(image, 0, corner, crop[0], crop[1], dim)
    else:
        #if the current ratio is taller crop top and bottom, else crop sides
        if ratio > target_ratio:
            crop = (round(target_ratio*image.size(2)), image.size(2))
        else:
            crop = (image.size(1), round(image.size(1)/target_ratio))
        image = tff.center_crop(image, crop)
        image = tff.resize(image, dim)
        


    return image

def random_augment(image):
    # flip, color, rotate, noise?, blur?
    transform_list = [tf.RandomHorizontalFlip(0.5)]
    sharpness_factor = random.uniform(0.85, 1.15)
    transform_list += [tf.RandomAdjustSharpness(sharpness_factor, 0.5)]
    transform_list += [tf.ColorJitter(0.15, 0.0, 0.25, 0.0)]
    transform_list += [tf.RandomRotation(10, tf.InterpolationMode.BILINEAR)]
    
    transforms = tf.RandomApply(torch.nn.ModuleList(transform_list), 0.75)
    
    return transforms(image)