import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import torchvision.io as io
import torchvision.transforms.functional as f
from random import Random
from math import ceil

class Webcams_reg(Dataset):
    def __init__(self, dataset_dir, transform=lambda x, augment:x, augment=False, limits=dict()):
        if isinstance(dataset_dir, list):
            tmp_files = dataset_dir
        else: 
            tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.files = []
        self.transform = transform
        self.augment = augment
        
        tmp_files.sort()
        Random(37).shuffle(tmp_files)
        
        counts = dict.fromkeys(limits, 0)
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            string_value = path.basename(img_path)
            string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
            float_value = float(string_value)
            
            if float_value > 10.0 and 10.0 in limits:
                if counts[10.0] < limits[10.0]:
                    self.files.append(img_path)
                    counts[10.0] += 1
            elif float_value in limits:
                if counts[float_value] < limits[float_value]:
                    self.files.append(img_path)
                    counts[float_value] += 1
            else:
                self.files.append(img_path)
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        
        #Remove 47 top, 19 bottom, 3 left, 3 right
        dims = (data.size(1)-66, data.size(2)-6)
        data = f.crop(data, 46, 2, dims[0], dims[1])
        data = self.transform(data, self.augment)
        data = data.to(torch.float32)
        
        string_value = path.basename(img_path)
        string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
        float_value = float(string_value)
        float_value = 10.0 if float_value >= 10.0 else float_value
        
        value = torch.Tensor([float_value]).to(torch.float32)
        
        return (data, value)
    
class Webcams_cls(Dataset):
    def __init__(self, dataset_dir, transform=lambda x, augment:x, augment=False, limits=dict()):
        if isinstance(dataset_dir, list):
            tmp_files = dataset_dir
        else: 
            tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.files = []
        self.transform = transform
        self.augment = augment
        
        tmp_files.sort()
        Random(37).shuffle(tmp_files)
        
        counts = dict.fromkeys(limits, 0)
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            string_value = path.basename(img_path)
            string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
            float_value = float(string_value)
            
            if float_value > 10.0 and 10.0 in limits:
                if counts[10.0] < limits[10.0]:
                    self.files.append(img_path)
                    counts[10.0] += 1
            elif float_value in limits:
                if counts[float_value] < limits[float_value]:
                    self.files.append(img_path)
                    counts[float_value] += 1
            else:
                self.files.append(img_path)
                
                    
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        
        #Remove 9.58% top, 3.99% bottom, 3 left, 3 right
        crop_top = ceil(0.0958 * data.size(1))
        crop_bot = ceil(0.0399 * data.size(1))
        sub_vert = crop_top + crop_bot
        dims = (data.size(1)-sub_vert, data.size(2)-6)
        data = f.crop(data, crop_top, 2, dims[0], dims[1])
        data = self.transform(data, self.augment)
        data = data.float()
        
        string_value = path.basename(img_path)
        string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
        float_value = float(string_value)
        float_value = 10.0 if float_value >= 10.0 else float_value
        
        class_index = 0
        
        match float_value:
            case 1.0:
                class_index = 0
            case 1.25:
                class_index = 1
            case 1.5:
                class_index = 2
            case 1.75:
                class_index = 3
            case 2.0:
                class_index = 4
            case 2.25:
                class_index = 5
            case 2.5:
                class_index = 6
            case 3.0:
                class_index = 7
            case 4.0:
                class_index = 8
            case 5.0:
                class_index = 9
            case 6.0:
                class_index = 10
            case 7.0:
                class_index = 11
            case 8.0:
                class_index = 12
            case 9.0:
                class_index = 13
            case 10.0:
                class_index = 14
            
        value = torch.zeros((15))
        value[class_index] = 1.0
        
        return (data, value)

# Add a classifcation here
