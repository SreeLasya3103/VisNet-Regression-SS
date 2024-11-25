import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import torchvision.io as io
from random import Random

class FCS(Dataset):
    def __init__(self, dataset_dir, transformer):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return
        
        tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)

        tmp_files.sort()
        Random(36).shuffle(tmp_files)

        self.files = []
        self.labels = []

        for file in tmp_files:
            value = None
            if '0.02.png' in img_path:
                value = torch.Tensor([1,0,0])
            elif '0.01.png' in img_path:
                value = torch.Tensor([0,1,0])
            elif '0.005.png' in img_path:
                value = torch.Tensor([0,0,1])
            
        self.files.append(file)
        self.labels.append(value.float())

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
        
        