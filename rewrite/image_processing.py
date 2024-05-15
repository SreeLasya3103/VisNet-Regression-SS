import torch
import torchvision.transforms.functional as tf
import math

def resize_crop(image: torch.Tensor, dim):
    #height / width
    target_ratio = dim[0] / dim[1]
    ratio = image.size(1) / image.size(2)
    
    #if the current ratio is taller crop top and bottom, else crop sides
    if ratio > target_ratio:
        crop = (round(target_ratio*image.size(2)), image.size(2))
    else:
        crop = (image.size(1), round(image.size(1)/target_ratio))
    
    image = tf.center_crop(image, crop)
    kernel_size = math.floor(crop[0] / dim[0])
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    if kernel_size > 1:
        image = tf.gaussian_blur(image, kernel_size)
        image = tf.resize(image, dim, tf.InterpolationMode.NEAREST, antialias=False)
    else:
        image = tf.resize(image, dim, tf.InterpolationMode.BICUBIC, antialias=False)

    return image