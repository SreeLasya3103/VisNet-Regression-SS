import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import torchvision.io as io
import torchvision.transforms.functional as f

class Webcams_reg(Dataset):
    def __init__(self, dataset_dir, transform=lambda x:x):
        self.files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.transform = transform
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        
        #Remove 47 top, 3 bottom, 3 sides
        dims = (data.size(1)-50, data.size(2)-6)
        data = f.crop(data, 46, 2, dims[0], dims[1])
        data = self.transform(data)
        data = data.float()
        
        string_value = path.basename(img_path)
        string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
        float_value = float(string_value)
        float_value = 10.0 if float_value >= 10.0 else float_value
        
        value = torch.Tensor([float_value])
        
        return (data, value)

# Add a classifcation here