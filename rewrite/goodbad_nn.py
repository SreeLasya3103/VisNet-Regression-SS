import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision as tv
import torchvision.io as io
from torcheval.metrics import BinaryAccuracy
from glob import glob
from os import path
from random import Random

DSET_DIR = ''

USE_CUDA = True
EPOCHS = 80
LR = 0.1
IMG_RES = (280,280)

class GoodBadWebcams(Dataset):
    def __init__(self):
        good_images = glob(path.join(DSET_DIR, 'good'))
        bad_images = glob(path.join(DSET_DIR, 'bad'))
        
        self.img_label_pairs = [(img, 1.0) for img in good_images]
        self.img_label_pairs += [(img, 0.0) for img in bad_images]
        
        self.img_label_pairs
        Random(37).shuffle(self.img_label_pairs)
        
    def __len__(self):
        return len(self.img_label_pairs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_label_pair = self.img_label_pairs[idx]
        data = io.read_image(img_label_pair[0], io.ImageReadMode.RGB)/255
        
        return (data.to(torch.float32), torch.Tensor([img_label_pair[1]]).to(torch.float32))

model = tv.models.resnet50(kwargs={'num_classes':1})
if USE_CUDA:
    model.cuda()
    
dset = GoodBadWebcams()
train_loader, val_loader = random_split(dset, [0.75, 0.25])
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), LR)

for epoch in range(EPOCHS):
    print('\nEpoch ' + str(epoch+1))
    print('Training...')
    
    model.train()
    
    running_loss = 0.0
    accuracy = BinaryAccuracy()
    
    count = 0
    
    for step, (data, labels) in enumerate(train_loader):
        count += labels.size(0)
        
        if USE_CUDA:
            data = data.cuda()
            labels = labels.cuda()
            
        output = torch.sigmoid(model(data))
        loss = loss_fn(output, labels)
        loss.backward()
        running_loss += labels.size(0) * loss.item()
        
        accuracy.update(output, labels)
        
        optimizer.step()
        optimizer.zero_grad()
        
    print('Training loss:', running_loss/count)
    print('Training accuracy:', accuracy.compute().item())
    
    
    print('\nValidating...')
    
    running_loss = 0.0
    accuracy = BinaryAccuracy()
    
    count = 0
    
    for step, (data, labels) in enumerate(val_loader):
        count += labels.size(0)
        
        if USE_CUDA:
            data = data.cuda()
            labels = labels.cuda()
            
        output = torch.sigmoid(model(data))
        loss = loss_fn(output, labels)
        running_loss += labels.size(0) * loss.item()
        
        accuracy.update(output, labels)
        
    print('Validation loss:', running_loss/count)
    print('Validation accuracy:', accuracy.compute().item())