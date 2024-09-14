import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms.functional as vfunc
import numpy
import csv
import glob

def resize_crop(img, img_dim):
    target_ratio = img_dim[0] / img_dim[1]
    ratio = img.size(1) / img.size(2)
    
    if ratio > target_ratio:
        img = vfunc.center_crop(img, (round(img.size(2)*target_ratio), img.size(2)))
    elif ratio < target_ratio:
        img = vfunc.center_crop(img, (img.size(1), round(img.size(1)/target_ratio)))
    
    img = vfunc.resize(img, img_dim, vfunc.InterpolationMode.BICUBIC, antialias=False)

    return img

class SSF_reg(Dataset):
    def __init__(self, dataset_dir, set_type, img_dim, channels=3):
        self.files = glob.glob(os.path.join(dataset_dir, set_type) + '/*.jpg')
        self.img_dim = img_dim
        self.labels_dict = None
        
        with open(os.path.join(dataset_dir, 'label.csv'), mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)
            self.labels_dict = {rows[0]:float(rows[7]) for rows in reader}
        
    def __len__(self):
        return len(self.files)
            
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.files[idx]
        dict_key = os.path.basename(img_path)[-19:]
        value = torch.tensor([[self.labels_dict[dict_key]]])
        
        orig = None
        
        orig = Image.open(img_path).convert('YCbCr')
        orig = transforms.PILToTensor()(orig)
        orig = resize_crop(orig, self.img_dim) / 255
        
        data = orig.view((1, 1, -1, *self.img_dim))
        
        return (data, value)
    
    @staticmethod
    def collate_fn(data):
        length = len(data)

        orig = torch.cat([data[i][0][0] for i in range(length)])
        values = torch.cat([data[i][1] for i in range(length)])

        orig = orig.view((1, *orig.size()))
        
        data = orig
        
        return (data, values)        
