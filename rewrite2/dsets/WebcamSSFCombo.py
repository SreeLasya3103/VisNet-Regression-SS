from dsets import SSF, Webcams
import sys
import io
from random import Random
import torch
from math import ceil
from torch.utils.data import Dataset

#FUUUUCK FUCK FUCK FUCK FUCK FUUUCK FUCK
#Maybe make file list a list of tuples where each file is tagged with which dataset it comes from? Would that work with existing code?
#I think it would work

class WebcamsSSF_cls_10(Dataset):
    def __init__(self, dataset_dir, transformer, limits=(dict(), dict())):
        self.transformer = transformer

        if type(dataset_dir) is tuple:
            if type(dataset_dir[0]) is list:
                tmp_files = dataset_dir[0]
                tmp_labels = dataset_dir[1]

                webcams_files = []
                webcams_labels = []
                ssf_files = []
                ssf_labels = []

                self.files = []
                self.labels = []

                for i in range(len(tmp_files)):
                    if tmp_files[i][2] == 'webcams':
                        webcams_files.append(tmp_files[i][1])
                        webcams_labels.append(tmp_labels[i])
                    elif tmp_files[i][2] == 'ssf':
                        ssf_files.append(tmp_files[i][1])
                        ssf_labels.append(tmp_labels[i])
                    else:
                        sys.exit("how")

                self.webcams = Webcams.Webcams_cls_10((webcams_files, webcams_labels), transformer=transformer)
                self.ssf = SSF.SSF_cls_10((ssf_files, ssf_labels), transformer=transformer)

                for i in range(len(self.webcams.files)):
                    self.files.append((i, self.webcams.files[i], 'webcams'))
                    self.labels.append(self.webcams.labels[i])
                for i in range(len(self.ssf.files)):
                    self.files.append((i, self.ssf.files[i], 'ssf'))
                    self.labels.append(self.ssf.labels[i])


            elif type(dataset_dir[0]) is str:
                self.files = []
                self.labels = []
                self.webcams = Webcams.Webcams_cls_10(dataset_dir[0], transformer=transformer, limits=limits[0])
                self.ssf = SSF.SSF_cls_10(dataset_dir[1], transformer=transformer, limits=limits[1])

                for i in range(len(self.webcams.files)):
                    self.files.append((i, self.webcams.files[i], 'webcams'))
                    self.labels.append(self.webcams.labels[i])
                for i in range(len(self.ssf.files)):
                    self.files.append((i, self.ssf.files[i], 'ssf'))
                    self.labels.append(self.ssf.labels[i])
            else:
                sys.exit("dataset_dir needs to be a tuple of lists or tuple of strings")
        else:
            sys.exit("dataset_dir needs to be a tuple")

        Random(36).shuffle(self.files)
        Random(36).shuffle(self.labels)
                
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.files[idx]
        if img_path[2] == 'webcams':
            item = self.webcams.__getitem__(img_path[0])
        else:
            item = self.ssf.__getitem__(img_path[0])
        
        return item