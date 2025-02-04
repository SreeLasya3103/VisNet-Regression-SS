import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import torchvision.io as io
from random import Random

class FROSI(Dataset):
    def __init__(self, dataset_dir, transformer):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return
        
        self.files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.files.sort()
        Random(36).shuffle(self.files)

        self.labels = []

        for file in self.files:
            if 'fog_50' in file:
                class_index = 0
            elif 'fog_100' in file:
                class_index = 1
            elif 'fog_150' in file:
                class_index = 2
            elif 'fog_200' in file:
                class_index = 3
            elif 'fog_250' in file:
                class_index = 4
            elif 'fog_300' in file:
                class_index = 5
            elif 'fog_400' in file:
                class_index = 6
            else:
                SystemExit('Invalid FROSI class')

            value = torch.zeros((7), dtype=torch.float32)
            value[class_index] = 1.0

            self.labels.append(value)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path =  self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        
        data = self.transformer(data)
        data = data.float()

        label = torch.Tensor(self.labels[idx])
        
        return (data, label, img_path)
        