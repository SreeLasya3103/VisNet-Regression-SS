import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import torchvision.io as io
import sqlite3
from random import Random

class Jacobs(Dataset):
    def __init__(self, dataset_dir, transformer, max_images=99999999):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return
        
        tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)

        self.files = []
        self.labels = []

        Random(36).shuffle(self.files)

        for file in tmp_files:
            if len(self.files) == max_images:
                break

            parts = file.split('-')
            value = float(parts[2][0:-4])

            if value < 1.0:
                continue

            self.files.append(file)
            self.labels.append(torch.Tensor([value]).float())
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path =  self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        
        data = self.transformer(data)
        data = data.float()

        label = torch.Tensor(self.labels[idx])
        
        return (data, label)