import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import csv
from random import Random
from sys import maxsize
import torchvision.io as io

class SSF_reg(Dataset):
    def __init__(self, dataset_dir, transformer, max_ten_plus=99999999):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return
        
        tmp_files = glob(path.normpath(dataset_dir + '/**/*.jpg'), recursive=True)
        
        tmp_files.sort()
        Random(36).shuffle(tmp_files)

        self.files = []
        self.labels = []
        labels_dict = None
        
        with open(path.join(dataset_dir, 'label.csv'), mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)
            labels_dict = {rows[0]:float(rows[7]) for rows in reader}
            
        ten_plus_count = 0
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            dict_key = path.basename(img_path)[-19:]
            vis = torch.tensor([[labels_dict[dict_key]]])
            if vis >= 10.0:
                if ten_plus_count < max_ten_plus:
                    self.files.append(img_path)
                    self.labels.append(torch.Tensor([10.0]).float())
                    ten_plus_count += 1
            else:
                self.files.append(img_path)
                self.labels.append(torch.Tensor([vis]).float())

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path =  self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        
        data = self.transformer(data)
        data = data.float()

        label = self.labels[idx]
        
        return (data, label)
        
class SSF_cls_10(Dataset):
    def __init__(self, dataset_dir, transformer, limits=dict()):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return
        
        tmp_files = glob(path.normpath(dataset_dir + '/**/*.jpg'), recursive=True)
        self.files = []
        self.labels = []
        labels_dict = None
        
        tmp_files.sort()
        Random(36).shuffle(tmp_files)
        
        with open(path.join(dataset_dir, 'label.csv'), mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)
            labels_dict = {rows[0]:float(rows[7]) for rows in reader}
        
        counts = dict.fromkeys(limits, 0)
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            dict_key = path.basename(img_path)[-19:]
            vis = torch.tensor([[labels_dict[dict_key]]])
            
            if vis > 9.5:
                vis = 10.0
                class_index = 9
            elif vis > 8.5:
                vis = 9.0
                class_index = 8
            elif vis > 7.5:
                vis = 8.0
                class_index = 7
            elif vis > 6.5:
                vis = 7.0
                class_index = 6
            elif vis > 5.5:
                vis = 6.0
                class_index = 5
            elif vis > 4.5:
                vis = 5.0
                class_index = 4
            elif vis > 3.5:
                vis = 4.0
                class_index = 3
            elif vis > 2.5:
                vis = 3.0
                class_index = 2
            elif vis > 1.5:
                vis = 2.0
                class_index = 1   
            elif vis >= 0:
                vis = 1.0
                class_index = 0
            else:
                continue
            
            label = torch.zeros((10)).float()
            label[class_index] = 1.0

            if vis in limits:
                if counts[vis] < limits[vis]:
                    self.files.append(img_path)
                    self.labels.append(label)
                    counts[vis] += 1
            else:
                self.files.append(img_path)
                self.labels.append(label)
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path =  self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        
        data = self.transformer(data)
        data = data.float()

        label = torch.tensor(self.labels[idx]).float()
        
        return (data, label, img_path)