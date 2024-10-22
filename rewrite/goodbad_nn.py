import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision as tv
import torchvision.io as io
from torcheval.metrics import BinaryAccuracy
from glob import glob
import os
from os import path
from random import Random
from math import ceil
import torchvision.transforms.functional as tff
import torchvision.transforms as tf
import sys
from progress.bar import Bar

DSET_DIR = '/home/feet/Documents/LAWN/datasets/Webcams'
MODEL_PATH = '/home/feet/Documents/LAWN/Visibility-Networks/rewrite/goodbad-bestloss1.pt'
LABELED_DSET_PATH = '/home/feet/Documents/LAWN/datasets/quality-labeled-webcams'
CLEANED_DSET_PATH = '/home/feet/Documents/LAWN/datasets/quality-labeled-webcams/by-network'

USE_CUDA = True
EPOCHS = 100
BATCH_SIZE = 8
LR = 0.0000005
IMG_RES = (310,470)
MAX_GOOD = 300
MAX_BAD = 300

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class GoodBadWebcams(Dataset):
    def __init__(self, max_good, max_bad):
        good_images = glob(path.normpath(LABELED_DSET_PATH + '/good/**/*.png'), recursive=True)
        bad_images = glob(path.normpath(LABELED_DSET_PATH + '/bad/**/*.png'), recursive=True)
        Random().shuffle(good_images)
        Random().shuffle(bad_images)

        if len(good_images) > max_good:
            good_images = good_images[:max_good]
        if len(bad_images) > max_bad:
            bad_images = bad_images[:max_bad]
        
        self.img_label_pairs = [(img, 1.0) for img in good_images]
        self.img_label_pairs += [(img, 0.0) for img in bad_images]
        
        Random(37).shuffle(self.img_label_pairs)

        self.augment = False

        self.tf = tf.RandomHorizontalFlip()
        
    def __len__(self):
        return len(self.img_label_pairs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_label_pair = self.img_label_pairs[idx]
        data = io.read_image(img_label_pair[0], io.ImageReadMode.RGB)/255
        
        #Remove 12.81% top, 3 bottom, 3 left, 3 right
        crop_top = ceil(0.1281 * data.size(1))
        crop_bot = 3
        sub_vert = crop_top + crop_bot
        dims = (data.size(1)-sub_vert, data.size(2)-6)
        data = tff.crop(data, crop_top, 2, dims[0], dims[1])
        data = tff.resize(data, IMG_RES)
        if self.augment:
            data = self.tf(data)
        data = data.float()

        return (data, torch.Tensor([img_label_pair[1]]).to(torch.float32))

def train():
    print('train_loss,train_acc,val_loss,val_acc,bad_acc,good_acc')

    model = tv.models.resnet34(num_classes=1)
    # model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    if USE_CUDA:
        model.cuda()
        
    dset = GoodBadWebcams(MAX_GOOD, MAX_BAD)
    train_set, val_set = random_split(dset, [0.75, 0.25])
    train_loader = DataLoader(train_set, BATCH_SIZE, True, num_workers=4)
    val_loader = DataLoader(val_set, BATCH_SIZE, True, num_workers=4)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), LR)

    best_loss = 99999.0

    for epoch in range(EPOCHS):
        eprint('\nEpoch ' + str(epoch+1))
        eprint('Training...')
        
        model.train()
        train_set.dataset.augment = True
        
        running_loss = 0.0
        accuracy = BinaryAccuracy()
        
        count = 0
        
        for step, (data, labels) in enumerate(train_loader):
            bs =  labels.size(0)
            count += bs
            
            if USE_CUDA:
                data = data.cuda()
                labels = labels.cuda()
                
            output = torch.sigmoid(model(data))
            loss = loss_fn(output, labels)
            loss.backward()
            running_loss += labels.size(0) * loss.item()
            
            accuracy.update(torch.round(output).reshape((bs)), labels.reshape((bs)))
            
            optimizer.step()
            optimizer.zero_grad()
            
        print(running_loss/count, end=',')
        print(accuracy.compute().item(), end=',')
        
        
        eprint('\nValidating...')

        model.eval()
        val_loader.dataset.augment = False

        running_loss = 0.0
        accuracy = BinaryAccuracy()
        
        count = 0

        good = bad = goodAsBad = badAsGood = 0
        
        for step, (data, labels) in enumerate(val_loader):
            bs =  labels.size(0)
            count += bs
            
            if USE_CUDA:
                data = data.cuda()
                labels = labels.cuda()
                
            output = torch.sigmoid(model(data))
            loss = loss_fn(output, labels)
            running_loss += labels.size(0) * loss.item()
            
            accuracy.update(torch.round(output).reshape((bs)), labels.reshape((bs)))

            for i in range(bs):
                if round(labels[i].item()) == 0:
                    bad += 1
                    if round(output[i].item()) == 1:
                        badAsGood += 1
                else:
                    good += 1
                    if round(output[i].item()) == 0:
                        goodAsBad += 1

        val_loss = running_loss/count
        print(val_loss, end=',')
        print(accuracy.compute().item(), end=',')

        print(1.0 - badAsGood/bad, end=',')
        print(1.0 - goodAsBad/good)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'goodbad-bestloss.pt')

def clean_dset():
    with torch.inference_mode():
        images = glob(path.normpath(DSET_DIR + '/**/*.png'), recursive=True)

        model = tv.models.resnet34(num_classes=1)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        if USE_CUDA:
            model.cuda()

        bar = Bar()
        bar.max = len(images)

        for img in images:
            data = io.read_image(img, io.ImageReadMode.RGB)/255
            if USE_CUDA:
                data = data.cuda()

            #Remove 12.81% top, 3 bottom, 3 left, 3 right
            crop_top = ceil(0.1281 * data.size(1))
            crop_bot = 3
            sub_vert = crop_top + crop_bot
            dims = (data.size(1)-sub_vert, data.size(2)-6)
            data = tff.crop(data, crop_top, 2, dims[0], dims[1])
            data = tff.resize(data, IMG_RES)
            data = data.unsqueeze(0)
            data = data.float()

            output = torch.sigmoid(model(data))

            label = 'good'
            if round(output.item()) == 0:
                label = 'bad'

            cntningDir = os.path.basename(os.path.dirname(img))
            newDir = os.path.join(LABELED_DSET_PATH, label, cntningDir)
            if not os.path.isdir(newDir):
                os.makedirs(newDir)
            
            os.rename(img, os.path.join(newDir, os.path.basename(img)))

            bar.next()
                
clean_dset()