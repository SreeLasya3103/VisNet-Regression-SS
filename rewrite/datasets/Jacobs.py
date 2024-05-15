import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import torchvision.io as io
import sqlite3

class Jacobs(Dataset):
    def __init__(self, dataset_dir, transform=lambda x:x):
        self.files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.transform = transform
        self.database_path = path.normpath(dataset_dir + '/images/fog_v1.db')
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        data = self.transform(data)
        data = data.float()
        
        db_img_path = "images/" + path.basename(img_path)
        con = sqlite3.connect(self.database_path)
        cur = con.cursor()
        res = cur.execute(f"SELECT fogFarVisDist from capture WHERE path='{db_img_path}'")

        value = torch.tensor([res.fetchone()[0]])
        
        return (data, value)