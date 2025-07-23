import torch
from torch.utils.data import Dataset
from glob import glob
from os import path
import torchvision.io as io
import torchvision.transforms.functional as f
from random import Random
from math import ceil

class Webcams_reg(Dataset):
    def __init__(self, dataset_dir, transformer, limits=dict(), site_filter=None):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return
        
        tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.files = []
        self.labels = []
        
        tmp_files.sort()
        Random(36).shuffle(tmp_files)
        
        counts = dict.fromkeys(limits, 0)
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            site_id = path.basename(img_path).split('_')[0]
            if site_filter is not None and site_id not in site_filter:
                continue

            string_value = path.basename(img_path)
            string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
            if string_value == '10+':
                float_value = 10.0
            else:
                float_value = float(string_value)
                if float_value > 10.0:
                    float_value = 10.0
            
            if float_value in limits:
                if counts[float_value] < limits[float_value]:
                    self.files.append(img_path)
                    self.labels.append(torch.Tensor([float_value]).float())
                    counts[float_value] += 1
            else:
                self.files.append(img_path)
                self.labels.append(torch.Tensor([float_value]).float())

        Random(36).shuffle(self.files)
        Random(36).shuffle(self.labels)

        
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path =  self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        
        #Remove 12.81% top, 3 bottom, 3 left, 3 right
        crop_top = ceil(0.1281 * data.size(1))
        crop_bot = 3
        sub_vert = crop_top + crop_bot
        dims = (data.size(1)-sub_vert, data.size(2)-6)
        data = f.crop(data, crop_top, 2, dims[0], dims[1])
        data = self.transformer(data)
        data = data.float()

        label = torch.Tensor(self.labels[idx])
        
        return (data, label, img_path)
    
class Webcams_cls(Dataset):
    def __init__(self, dataset_dir, transformer, limits=dict()):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return
        
        tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
            
        self.files = []
        self.labels = []
        
        tmp_files.sort()
        Random(36).shuffle(tmp_files)
        
        counts = dict.fromkeys(limits, 0)
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            string_value = path.basename(img_path)
            string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
            if string_value == '10+':
                float_value = 10.0
            else:
                float_value = float(string_value)
            float_value = min(float_value, 10.0)
                
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
            
            label = torch.zeros((15)).float()
            label[class_index] = 1.0
            
            if float_value in limits:
                if counts[float_value] < limits[float_value]:
                    self.files.append(img_path)
                    self.labels.append(label)
                    counts[float_value] += 1
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
        
        #Remove 12.81% top, 3 bottom, 3 left, 3 right
        crop_top = ceil(0.1281 * data.size(1))
        crop_bot = 3
        sub_vert = crop_top + crop_bot
        dims = (data.size(1)-sub_vert, data.size(2)-6)
        data = f.crop(data, crop_top, 2, dims[0], dims[1])
        data = self.transformer(data)
        data = data.float()

        label = torch.Tensor(self.labels[idx])
        
        return (data, label, img_path)

class Webcams_cls_10(Dataset):
    def __init__(self, dataset_dir, transform=lambda x, augment:x, augment=False, limits=dict(), site_filter=None):
        from glob import glob
        from random import Random

        if isinstance(dataset_dir, list):
            tmp_files = dataset_dir
        else: 
            tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)

        self.files = []
        self.labels = []
        self.transform = transform
        self.augment = augment

        tmp_files.sort()
        Random(36).shuffle(tmp_files)

        counts = dict.fromkeys(limits, 0)

        for img_path in tmp_files:
            fname = path.basename(img_path)

            site_id = fname.split('_')[0]
            if site_filter is not None and site_id not in site_filter:
                continue

            try:
                vis_token = fname.split('_')[2].split('.')[0]
                string_value = vis_token.split('S')[1].split('m')[0].replace('-', '.')
                if string_value == '10+':
                    float_value = 10.0
                else:
                    float_value = float(string_value)
                float_value = min(float_value, 10.0)
                if float_value == 0.0:
                    continue
            except:
                continue  # skip malformed filenames

            match float_value:
                case 1.0 | 1.25:
                    class_index = 0
                    float_value = 1.0
                case 1.75 | 2.0 | 2.25:
                    class_index = 1
                    float_value = 2.0
                case 3.0:
                    class_index = 2
                case 4.0:
                    class_index = 3
                case 5.0:
                    class_index = 4
                case 6.0:
                    class_index = 5
                case 7.0:
                    class_index = 6
                case 8.0:
                    class_index = 7
                case 9.0:
                    class_index = 8
                case 10.0:
                    class_index = 9
                case _:
                    continue  # skip values like 1.5, 2.5

            label = torch.zeros((10)).float()
            label[class_index] = 1.0

            if float_value in limits:
                if counts[float_value] < limits[float_value]:
                    self.files.append(img_path)
                    self.labels.append(label)
                    counts[float_value] += 1
            else:
                self.files.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255

        # Remove borders
        crop_top = ceil(0.1281 * data.size(1))
        crop_bot = 3
        sub_vert = crop_top + crop_bot
        dims = (data.size(1)-sub_vert, data.size(2)-6)
        data = f.crop(data, crop_top, 2, dims[0], dims[1])
        data = self.transform(data, self.augment).float()

        return (data, self.labels[idx], self.files[idx])

class Webcams_cls_5(Dataset):
    def __init__(self, dataset_dir, transformer, limits=dict()):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return
        
        tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.files = []
        self.labels = []
        
        tmp_files.sort()
        Random(36).shuffle(tmp_files)
        
        counts = dict.fromkeys(limits, 0)
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            string_value = path.basename(img_path)
            string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
            if string_value == '10+':
                float_value = 10.0
            else:
                float_value = float(string_value)
            float_value = min(float_value, 10.0)
            
            if float_value <= 2.5:
                class_index = 0
            elif float_value <= 4.0:
                class_index = 1
            elif 5.0 <= float_value <= 6.0:
                class_index = 2
            elif 7.0 <= float_value <= 8.0:
                class_index = 3
            else:
                class_index = 4
                
            label = torch.zeros((5)).float()
            label[class_index] = 1.0
            
            if float_value in limits:
                if counts[float_value] < limits[float_value]:
                    self.files.append(img_path)
                    self.labels.append(label)
                    counts[float_value] += 1
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
        
        #Remove 12.81% top, 3 bottom, 3 left, 3 right
        crop_top = ceil(0.1281 * data.size(1))
        crop_bot = 3
        sub_vert = crop_top + crop_bot
        dims = (data.size(1)-sub_vert, data.size(2)-6)
        data = f.crop(data, crop_top, 2, dims[0], dims[1])
        data = self.transformer(data)
        data = data.float()

        label = torch.tensor(self.labels[idx]).float()
        
        return (data, label, img_path)

class Webcams_cls_3(Dataset):
    def __init__(self, dataset_dir, transformer, limits=dict()):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return
        
        tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.files = []
        self.labels = []
        
        tmp_files.sort()
        Random(36).shuffle(tmp_files)
        
        counts = dict.fromkeys(limits, 0)
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            string_value = path.basename(img_path)
            string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
            if string_value == '10+':
                float_value = 10.0
            else:
                float_value = float(string_value)
            float_value = min(float_value, 10.0)
            
            if float_value <= 3.0:
                class_index = 0
            elif float_value <= 7.0:
                class_index = 1
            else:
                class_index = 2
                
            label = torch.zeros((3)).float()
            label[class_index] = 1.0
            
            if float_value in limits:
                if counts[float_value] < limits[float_value]:
                    self.files.append(img_path)
                    self.labels.append(label)
                    counts[float_value] += 1
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
            crop_top = ceil(0.1281 * data.size(1))
            crop_bot = 3
            sub_vert = crop_top + crop_bot
            dims = (data.size(1)-sub_vert, data.size(2)-6)
            data = f.crop(data, crop_top, 2, dims[0], dims[1])
            data = self.transformer(data).float()
            label = self.labels[idx]  # already a float tensor like torch.Tensor([4.0])
            return (data, label, img_path)


class Webcams_cls_3lmh(Dataset):
    def __init__(self, dataset_dir, transformer, limits=dict()):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return
        
        tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.files = []
        self.labels = []
        
        tmp_files.sort()
        Random(36).shuffle(tmp_files)
        
        counts = dict.fromkeys(limits, 0)
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            string_value = path.basename(img_path)
            string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
            if string_value == '10+':
                float_value = 10.0
            else:
                float_value = float(string_value)
            float_value = min(float_value, 10.0)
            
            if float_value < 3.0:
                class_index = 0
            elif float_value < 5.0:
                class_index = 1
            else:
                class_index = 2
                
            label = torch.zeros((3)).float()
            label[class_index] = 1.0
            
            if float_value in limits:
                if counts[float_value] < limits[float_value]:
                    self.files.append(img_path)
                    self.labels.append(label)
                    counts[float_value] += 1
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
        
        #Remove 12.81% top, 3 bottom, 3 left, 3 right
        crop_top = ceil(0.1281 * data.size(1))
        crop_bot = 3
        sub_vert = crop_top + crop_bot
        dims = (data.size(1)-sub_vert, data.size(2)-6)
        data = f.crop(data, crop_top, 2, dims[0], dims[1])
        data = self.transformer(data)
        data = data.float()

        label = torch.tensor(self.labels[idx]).float()
        
        return (data, label, img_path)

class Webcams_cls_1_10(Dataset):
    def __init__(self, dataset_dir, transformer, limits=dict()):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return


        tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.files = []
        self.labels = []
        
        tmp_files.sort()
        Random(36).shuffle(tmp_files)
        
        counts = dict.fromkeys(limits, 0)
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            string_value = path.basename(img_path)
            string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
            float_value = float(string_value)
            float_value = min(float_value, 10.0)
            
            if float_value <= 1.25:
                class_index = 0
            elif float_value >= 10.0:
                class_index = 1
            else:
                continue
                
            label = torch.zeros((2)).float()
            label[class_index] = 1.0
            
            if float_value in limits:
                if counts[float_value] < limits[float_value]:
                    self.files.append(img_path)
                    self.labels.append(label)
                    counts[float_value] += 1
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
        
        #Remove 12.81% top, 3 bottom, 3 left, 3 right
        crop_top = ceil(0.1281 * data.size(1))
        crop_bot = 3
        sub_vert = crop_top + crop_bot
        dims = (data.size(1)-sub_vert, data.size(2)-6)
        data = f.crop(data, crop_top, 2, dims[0], dims[1])
        data = self.transformer(data)
        data = data.float()

        label = self.labels[idx]
        
        return (data, label)

class Webcams_cls_10_full(Dataset):
    def __init__(self, dataset_dir, transformer, limits=dict()):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            self.files = dataset_dir[0]
            self.labels = dataset_dir[1]
            return
        
        tmp_files = glob(path.normpath(dataset_dir + '/**/*.png'), recursive=True)
        self.files = []
        self.labels = []

        tmp_files.sort()
        Random(36).shuffle(tmp_files)
        
        counts = dict.fromkeys(limits, 0)
        
        for i in range(len(tmp_files)):
            img_path = tmp_files[i]
            string_value = path.basename(img_path)
            string_value = string_value.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')
            if string_value == '10+':
                float_value = 10.0
            else:
                float_value = float(string_value)
            float_value = min(float_value, 10.0)

            if float_value == 0.0:
                continue

            match float_value:
                case 1.0:
                    class_index = 0
                case 1.25:
                    class_index = 0
                    # float_value = 1.0
                case 1.5:
                    class_index = 0
                case 1.75:
                    class_index = 1
                    # float_value = 2.0
                case 2.0:
                    class_index = 1
                case 2.25:
                    class_index = 1
                    # float_value = 2.0
                case 2.5:
                    class_index = 1
                case 3.0:
                    class_index = 2
                case 4.0:
                    class_index = 3
                case 5.0:
                    class_index = 4
                case 6.0:
                    class_index = 5
                case 7.0:
                    class_index = 6
                case 8.0:
                    class_index = 7
                case 9.0:
                    class_index = 8
                case 10.0:
                    class_index = 9

            label = torch.zeros((10)).float()
            label[class_index] = 1.0
            
            if float_value in limits:
                if counts[float_value] < limits[float_value]:
                    self.files.append(img_path)
                    self.labels.append(label)
                    counts[float_value] += 1
            else:
                self.files.append(img_path)
                self.labels.append(label)
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        data = io.read_image(img_path, io.ImageReadMode.RGB)/255
        
        #Remove 12.81% top, 3 bottom, 3 left, 3 right
        crop_top = ceil(0.1281 * data.size(1))
        crop_bot = 3
        sub_vert = crop_top + crop_bot
        dims = (data.size(1)-sub_vert, data.size(2)-6)
        data = f.crop(data, crop_top, 2, dims[0], dims[1])
        data = self.transformer(data)
        data = data.float()

        label = torch.Tensor(self.labels[idx])
        
        return (data, label, img_path)