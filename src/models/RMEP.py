from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
import copy
from progress.bar import Bar
import numpy as np
import torcheval.metrics


IMG_SIZE = (256, 256)
NUM_CLASSES = 3
NUM_CHANNELS = 3

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(256, 256, 3, 1, 0),
                 nn.InstanceNorm2d(256),
                 nn.ReLU(True)]
        
        model += [nn.Dropout(0.5)]

        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(256, 256, 3, 1, 0),
                  nn.InstanceNorm2d(256)]
        
        self.model = nn.Sequential(*model);

    def forward(self, x):
        return x + self.model(x)


class RMEP(nn.Module):
    def __init__(self):
        super(RMEP, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(NUM_CHANNELS, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True)]
        
        model += [nn.Conv2d(64, 128, 3, 2, 1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(128, 256, 3, 2, 1),
                  nn.InstanceNorm2d(256),
                  nn.ReLU(True)]
        
        model += [ResNet(), ResNet(), ResNet(), ResNet(), ResNet(), ResNet()]

        model += [nn.Conv2d(256, 256, 4, 2, 1),
                  nn.InstanceNorm2d(256),
                  nn.ReLU(True)]
        
        model += [nn.AdaptiveMaxPool2d((2,2))]

        model += [nn.Conv2d(256, 128, 3, 1, 1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(128, 64, 3, 1, 1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(True)]
        
        model += [nn.Flatten(), nn.LazyLinear(NUM_CLASSES)]

        self.model = nn.Sequential(*model);
    
    def forward(self, x):
        return self.model(x)

def create_and_save():
    net = RMEP()
    net.eval()
    net(torch.rand((1, NUM_CHANNELS, IMG_SIZE[0], IMG_SIZE[1])))
    m = torch.jit.trace(net, torch.rand((1, NUM_CHANNELS, IMG_SIZE[0], IMG_SIZE[1])))
    m.save('RMEP-' + str(NUM_CHANNELS) + 'x' + str(IMG_SIZE[1]) + 'x' + str(IMG_SIZE[0]) + '-' + str(NUM_CLASSES) + '.pt')

def train_classification(config, use_cuda, dataset):
    print('Preparing dataset...')
    
    train_set = dataset(config['dataPath'], 'train', config['imgDim'], config['channels'])
    val_set = dataset(config['dataPath'], 'val', config['imgDim'], config['channels'])
    
    train_loader = DataLoader(train_set, config['batchSize'], True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_set, config['batchSize'], collate_fn=dataset.collate_fn)
    
    print('Preparing model...')
    model = torch.jit.load(config['modelPath']);
    if use_cuda:
        model.cuda()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.0002)
    
    metrics_file = open('results.csv', 'a')
    metrics_file.write('training loss,training accuracy,validation loss,validation accuracy\n')
    metrics_file.close()
    
    best_val = (0.0, 0.0)
    
    for epoch in range(config['epochs']):
        if epoch > 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0002 * ((150 - epoch + 50) / 150)
        
        print('\nEpoch ' + str(epoch+1))
        print('Training...')
        
        model.train()
        
        correct = 0
        total = 0
        running_loss = 0.0
        
        bar = Bar()
        bar.max = len(train_loader)
        for step, (data, labels) in enumerate(train_loader):
            total += labels.size(0)

            data = data[0]
            
            if use_cuda:
                data = data.cuda()
                labels = labels.to(torch.device('cuda'))

            optimizer.zero_grad()
            
            output = model(data.float())
            loss = loss_fn(output, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            
            for i in range(output.size(0)):
                if output[i].argmax() == labels[i].argmax():
                    correct = correct + 1

            bar.next()
        
        train_loss = running_loss/total
        train_accuracy = correct/total
        print('\nTraining loss: ' + str(train_loss))
        print('Training accuracy: ' + str(train_accuracy))

        print('Validating...')
        correct = 0
        total = 0
        running_loss = 0.0
        
        with torch.no_grad():
            model.eval()
            
            bar = Bar()
            bar.max = len(val_loader)
            for step, (data, labels) in enumerate(val_loader):
                total += labels.size(0)
                
                data = data[0]

                if use_cuda:
                    data = data.cuda()
                    labels = labels.cuda()

                output = model(data.float())
                loss = loss_fn(output, labels)
                
                running_loss += loss.item()
                
                for i in range(output.size(0)):
                    if output[i].argmax() == labels[i].argmax():
                        correct = correct + 1

                bar.next()
        
        val_loss = running_loss/total
        val_accuracy = correct/total
        print('\nValidation loss: ' + str(val_loss))
        print('Validation accuracy: ' + str(val_accuracy))
        
        metrics_file = open('results.csv', 'a')
        metrics_file.write(str(train_loss) + ',' +
                           str(train_accuracy) + ',' +
                           str(val_loss) + ',' +
                           str(val_accuracy) + '\n')
        metrics_file.close()
        
        model.save('last.pt')
        
        if (val_accuracy > best_val[1] or
            (val_accuracy == best_val[1] and val_loss < best_val[0])):
            best_val = (val_loss, val_accuracy)
            model.save('best-acc.pt')

def train_regression(config, use_cuda, dataset):
    print('Preparing dataset...')
    
    train_set = dataset(config['dataPath'], 'train', config['imgDim'], config['channels'])
    val_set = dataset(config['dataPath'], 'val', config['imgDim'], config['channels'])
    
    train_loader = DataLoader(train_set, config['batchSize'], True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_set, config['batchSize'], collate_fn=dataset.collate_fn)
    
    print('Preparing model...')
    model = torch.jit.load(config['modelPath']);
    if use_cuda:
        model.cuda()
    
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), 0.0002)
    
    metrics_file = open('results.csv', 'a')
    metrics_file.write('training MAE,training R2,training RMSE,validation MAE,validation R2,validation RMSE\n')
    metrics_file.close()
    
    best_val = (0.0, 0.0)
    
    for epoch in range(config['epochs']):
        if epoch > 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0002 * ((150 - epoch + 50) / 150)
                
        print('\nEpoch ' + str(epoch+1))
        print('Training...')
        
        model.train()
        
        running_loss = 0.0
        total = 0
        all_outputs = torch.empty((0, 1)).cuda()
        all_labels = torch.empty((0, 1)).cuda()
        running_rmse = 0.0
        
        bar = Bar()
        bar.max = len(train_loader)
        for step, (data, labels) in enumerate(train_loader):
            total += labels.size(0)

            data = data[0]

            if use_cuda:
                data = data.cuda()
                labels = labels.to(torch.device('cuda'))

            optimizer.zero_grad()
            
            output = model(data.float())
            all_outputs = torch.cat((all_outputs, output))
            all_labels = torch.cat((all_labels, labels))
            loss = loss_fn(output, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            running_rmse += sqrt(nn.MSELoss()(output, labels).item())

            bar.next()
        
        train_loss = running_loss/total
        train_r2 = torcheval.metrics.functional.r2_score(all_outputs, all_labels).item()
        train_rmse = running_rmse/total
        print('\nTraining MAE: ' + str(train_loss))
        print('Training R2: ' + str(train_r2))
        print('Training RMSE: ' + str(train_rmse))

        print('Validating...')
        running_loss = 0.0
        total = 0
        all_outputs = torch.empty((0, 1)).cuda()
        all_labels = torch.empty((0, 1)).cuda()
        running_rmse = 0.0
        
        with torch.no_grad():
            model.eval()
            
            bar = Bar()
            bar.max = len(val_loader)
            for step, (data, labels) in enumerate(val_loader):
                total += labels.size(0)

                data = data[0]

                if use_cuda:
                    data = data.cuda()
                    labels = labels.cuda()

                output = model(data.float())
                all_outputs = torch.cat((all_outputs, output))
                all_labels = torch.cat((all_labels, labels))
                loss = loss_fn(output, labels)
                
                running_loss += loss.item()
                running_rmse += sqrt(nn.MSELoss()(output, labels).item())

                bar.next()
        
        val_loss = running_loss/total
        val_r2 = torcheval.metrics.functional.r2_score(all_outputs, all_labels).item()
        val_rmse = running_rmse/total
        print('\nValidation MAE: ' + str(val_loss))
        print('Validation R2: ' + str(val_r2))
        print('Validation RMSE: ' + str(val_rmse))
        
        metrics_file = open('results.csv', 'a')
        metrics_file.write(str(train_loss) + ',' +
                           str(train_r2) + ',' +
                           str(train_rmse) + ',' +
                           str(val_loss) + ',' +
                           str(val_r2) + ',' +
                           str(val_rmse) + '\n')
        metrics_file.close()
        
        model.save('last.pt')
        
        if (val_r2 > best_val[1] or
            (val_r2 == best_val[1] and val_loss < best_val[0])):
            best_val = (val_loss, val_r2)
            model.save('best-r2.pt')