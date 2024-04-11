import glob
import random
import os

files = glob.glob('/home/jmurr/Documents/LAWN/datasets/CombinedWebcams/*.png')
random.shuffle(files)
random.shuffle(files)
random.shuffle(files)
train = files[:1054]
val = files[1054:]

for img in train:
    os.replace(img, '/home/jmurr/Documents/LAWN/datasets/CombinedWebcams/train/' + os.path.basename(img))

for img in val:
    os.replace(img, '/home/jmurr/Documents/LAWN/datasets/CombinedWebcams/val/' + os.path.basename(img))