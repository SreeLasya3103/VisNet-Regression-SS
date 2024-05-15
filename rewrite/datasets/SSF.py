import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import csv
from random import Random
from sys import maxsize
import torchvision.io as io

class SSF_reg(Dataset):
    def __init__(self, dataset_dir, transform=lambda x:x, ten_plus_limit=350):
        self.files = []
        self.transform = transform
        self.labels_dict = None
        
        tmp_files = glob(path.normpath(dataset_dir + '/**/*.jpg'), recursive=True)
        
        with open(path.join(dataset_dir, 'label.csv'), mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)
            self.labels_dict = {rows[0]:float(rows[7]) for rows in reader}
            
        tmp_files.sort()
        Random(37).shuffle(tmp_files)
        
        ten_plus_count = 0
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            dict_key = path.basename(img_path)[-19:]
            vis = torch.tensor([[self.labels_dict[dict_key]]])
            if vis >= 10.0:
                if ten_plus_count < ten_plus_limit:
                    self.files.append(img_path)
                    self.labels_dict[dict_key] = 10.0
                    ten_plus_count += 1
            else:
                self.files.append(img_path)
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.files[idx]
        dict_key = path.basename(img_path)[-19:]
        value = torch.tensor([self.labels_dict[dict_key]]).float()
        
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        data = self.transform(data)
        data = data.float()
        
        return (data, value)
        
        