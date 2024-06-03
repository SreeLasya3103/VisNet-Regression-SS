import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import torchvision.io as io

class FCS(Dataset):
    def __init__(self, dataset_dir, transform=lambda x:x):
        if isinstance(dataset_dir, list):
            self.files = dataset_dir
        else: 
            self.files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.transform = transform
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        data = self.transform(data)
        data = data.to(torch.float32)
        
        value = None
        if '0.02.png' in img_path:
            value = torch.Tensor([1,0,0])
        elif '0.01.png' in img_path:
            value = torch.Tensor([0,1,0])
        elif '0.005.png' in img_path:
            value = torch.Tensor([0,0,1])
        value = value.to(torch.float32)
        
        return (data, value)
        
        