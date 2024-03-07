import glob
import random
import os

files = glob.glob('/home/jmurr/Downloads/database/images/*.png')
random.shuffle(files)
random.shuffle(files)
random.shuffle(files)
train = files[:480]
val = files[480:]

for img in train:
    os.replace(img, '/home/jmurr/Downloads/database/images/train/' + os.path.basename(img))

for img in val:
    os.replace(img, '/home/jmurr/Downloads/database/images/val/' + os.path.basename(img))