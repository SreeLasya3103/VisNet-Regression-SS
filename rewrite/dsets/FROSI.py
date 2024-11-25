import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import torchvision.io as io
from random import Random

class FROSI(Dataset):
    def __init__(self, dataset_dir, transform=lambda x, augment:x, augment=False):
        if isinstance(dataset_dir, list):
            self.files = dataset_dir
        else: 
            self.files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
            self.files.sort()
            Random(36).shuffle(self.files)

        self.labels = []

        for file in self.files:
            value = None
            if 'fog_50' in file:
                value = torch.Tensor([1,0,0,0,0,0,0])
            elif 'fog_100' in file:
                value = torch.Tensor([0,1,0,0,0,0,0])
            elif 'fog_150' in file:
                value = torch.Tensor([0,0,1,0,0,0,0])
            elif 'fog_200' in file:
                value = torch.Tensor([0,0,0,1,0,0,0])
            elif 'fog_250' in file:
                value = torch.Tensor([0,0,0,0,1,0,0])
            elif 'fog_300' in file:
                value = torch.Tensor([0,0,0,0,0,1,0])
            elif 'fog_400' in file:
                value = torch.Tensor([0,0,0,0,0,0,1])
            value = value.to(torch.float32)

            self.labels.append(value)


        self.transform = transform
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        data = self.transform(data, False)
        data = data.to(torch.float32)
        

        
        return (data, self.labels[idx])
        
        