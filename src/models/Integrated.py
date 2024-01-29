from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
from progress.bar import Bar
import numpy as np
import matplotlib
import torcheval.metrics

#****************************
#*******HEIGHT FIRST*********
#****************************
IMG_SIZE = (120, 160)
NUM_CLASSES = 1
NUM_CHANNELS = 3

pc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#0000ff', '#00ff00', '#ff0000', '#0000ff'])

def gray_fog_highlight(img):
    img = img.numpy()
            
    with np.nditer(img, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = (255 * 1.02**(x*255 - 255))/255
    
    img = np.concatenate((img, img, img))
    img = np.transpose(img, (1, 2, 0))
    img = np.array([img])
    
    return np.clip(img, 0, 1)

class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()

        class EntryFlow(nn.Module):
            def __init__(self):
                super(EntryFlow, self).__init__()

                self.l0 = nn.Sequential(nn.Conv2d(NUM_CHANNELS, 32, 3, 2, 1), nn.ReLU(),
                                        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU())

                self.c0 = nn.Conv2d(64, 128, 1, 2)

                self.l1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, groups=64),
                                        nn.Conv2d(64, 128, 1), nn.ReLU(),
                                        nn.Conv2d(128, 128, 3, 1, 1, groups=128),
                                        nn.Conv2d(128, 128, 1), nn.MaxPool2d(3, 2, 1))
                
                self.c1 = nn.Conv2d(128, 256, 1, 2)

                self.l2 =nn.Sequential(nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1, groups=128),
                                       nn.Conv2d(128, 256, 1), nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, 1, 1, groups=256),
                                       nn.Conv2d(256, 256, 1), nn.MaxPool2d(3, 2, 1))
                
                self.c2 = nn.Conv2d(256, 728, 1, 2)

                self.l3 = nn.Sequential(nn.ReLU(), nn.Conv2d(256, 256, 3, 1, 1, groups=256),
                                        nn.Conv2d(256, 728, 1), nn.ReLU(),
                                        nn.Conv2d(728, 728, 3, 1, 1, groups=728),
                                        nn.Conv2d(728, 728, 1), nn.MaxPool2d(3, 2, 1))
            def forward(self, x):
                x = self.l0(x)
                c = self.c0(x)
                x = torch.add(self.l1(x), c)
                c = self.c1(c)
                x = torch.add(self.l2(x), c)
                c = self.c2(x)
                x = torch.add(self.l3(x), c)

                return x
        
        class MiddleFlow(nn.Module):
            def __init__(self):
                super(MiddleFlow, self).__init__()

                self.l0 = nn.Sequential(*nn.ModuleList([nn.Sequential(nn.ReLU(), nn.Conv2d(728, 728, 3, 1, 1, groups=728), nn.Conv2d(728, 728, 1)) for i in range(3)]))

            def forward(self, x):
                return torch.add(self.l0(x), x)
            
        class ExitFlow(nn.Module):
            def __init__(self):
                super(ExitFlow, self).__init__()

                self.c0 = nn.Conv2d(728, 1024, 1, 2)

                self.l1 = nn.Sequential(nn.ReLU(), nn.Conv2d(728, 728, 3, 1, 1, groups=728),
                                        nn.Conv2d(728, 728, 1), nn.ReLU(),
                                        nn.Conv2d(728, 728, 3, 1, 1, groups=728),
                                        nn.Conv2d(728, 1024, 1), nn.MaxPool2d(3, 2, 1))
                
                self.l2 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, groups=1024),
                                        nn.Conv2d(1024, 1536, 1), nn.ReLU(),
                                        nn.Conv2d(1536, 1536, 3, 1, 1, groups=1536),
                                        nn.Conv2d(1536, 2048, 1), nn.ReLU())
                
            def forward(self, x):
                return self.l2(torch.add(self.c0(x), self.l1(x)))

        entry = EntryFlow()
        middle = nn.Sequential(*nn.ModuleList([MiddleFlow() for i in range(8)]))
        exit = ExitFlow()

        self.model = nn.Sequential(entry, middle, exit)

    def forward(self, x):
        return self.model(x)
    
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        model = [nn.Conv2d(NUM_CHANNELS, 64, 3, 1, 1), nn.ReLU(),
                 nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
                 nn.MaxPool2d(2, 2)]
        
        model += [nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
                  nn.MaxPool2d(2, 2)]
        
        model += [nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
                  nn.MaxPool2d(2, 2)]
        
        model += [nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
                  nn.MaxPool2d(2, 2)]
        
        model += [nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
                  nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
                  nn.MaxPool2d(2, 2)]
        
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Integrated(nn.Module):
    def __init__(self):
        super(Integrated, self).__init__()

        self.VGG = nn.Sequential(VGG(), nn.Flatten(), nn.LazyLinear(128))
        self.Xception = nn.Sequential(Xception(), nn.Flatten(), nn.LazyLinear(128))

        self.end = nn.Sequential(nn.Linear(256, 512), nn.Dropout(0.1), nn.Linear(512, NUM_CLASSES))
    
    def forward(self, x):
        orig = self.VGG(x[0])
        pc = self.Xception(x[1])
        return self.end(torch.cat((orig, pc), 1))
    
def create_and_save():
    net = Integrated()
    net.eval()
    net(torch.rand((2, 1, NUM_CHANNELS, *IMG_SIZE)))
    m = torch.jit.script(net)
    #m = torch.jit.trace(net, torch.rand((2, 1, NUM_CHANNELS, *IMG_SIZE)))
    m.save('Integrated-' + str(NUM_CHANNELS) + 'x' + str(IMG_SIZE[1]) + 'x' + str(IMG_SIZE[0]) + '-' + str(NUM_CLASSES) + '.pt')
    
def train_classification(config, use_cuda, dataset):
    print('Preparing dataset...')
    
    if config['channels'] == 1:
        cmap = gray_fog_highlight
    elif config['channels'] == 3:
        cmap = pc_cmap
    
    train_set = dataset(config['dataPath'], 'train', config['imgDim'], config['channels'], cmap, 'AVERAGE')
    val_set = dataset(config['dataPath'], 'val', config['imgDim'], config['channels'], cmap, 'AVERAGE')
    
    train_loader = DataLoader(train_set, config['batchSize'], True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_set, config['batchSize'], collate_fn=dataset.collate_fn)
    
    print('Preparing model...')
    model = torch.jit.load(config['modelPath']);
    if use_cuda:
        model.to(torch.device('cuda'))
    
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), config['learningRate'])
    
    metrics_file = open('results.csv', 'a')
    metrics_file.write('training loss,training accuracy,validation loss,validation accuracy\n')
    metrics_file.close()
    
    best_val = (0.0, 0.0)
    
    for epoch in range(config['epochs']):
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
            
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()
            
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
    
    if config['channels'] == 1:
        cmap = gray_fog_highlight
    elif config['channels'] == 3:
        cmap = pc_cmap
    
    train_set = dataset(config['dataPath'], 'train', config['imgDim'], config['channels'], cmap, 'AVERAGE')
    val_set = dataset(config['dataPath'], 'val', config['imgDim'], config['channels'], cmap, 'AVERAGE')
    
    train_loader = DataLoader(train_set, config['batchSize'], True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(val_set, config['batchSize'], collate_fn=dataset.collate_fn)
    
    print('Preparing model...')
    model = torch.jit.load(config['modelPath']);
    if use_cuda:
        model.cuda()
    
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), config['learningRate'])
    
    metrics_file = open('results.csv', 'a')
    metrics_file.write('training MAE,training R2,training RMSE,validation MAE,validation R2,validation RMSE\n')
    metrics_file.close()
    
    best_val = (0.0, 0.0)
    
    for epoch in range(config['epochs']):
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

def test_classification(config, use_cuda, dataset):
    print('Preparing dataset...')
    
    if config['channels'] == 1:
        cmap = gray_fog_highlight
    elif config['channels'] == 3:
        cmap = pc_cmap
    
    test_set = None
    subset = None
    if config['mode'] == 'VALIDATE':
        subset = 'val'
    else:
        subset = 'test'
     
    test_set = dataset(config['dataPath'], subset, config['imgDim'], config['channels'], cmap, 'AVERAGE')
    
    test_loader = DataLoader(test_set, config['batchSize'], collate_fn=dataset.collate_fn)

    ci_right = [[] for i in range(config['numClasses'])]
    ci_wrong = [[] for i in range(config['numClasses'])]

    if config['recordCI']:
        for i in range(0, config['numClasses']):
            ci_file = open('ci'+str(i)+'.csv', 'a')
            ci_file.write('right' + config['numClasses']*','+'wrong\n')
            ci_file.close()
    
    print('Preparing model...')
    model = torch.jit.load(config['modelPath']);
    if use_cuda:
        model.cuda()

    loss_fn = nn.CrossEntropyLoss()

    if subset == 'val':
        print('Validating...')
    else:
        print('Testing...')
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        model.eval()
        
        bar = Bar()
        bar.max = len(test_loader)
        for step, (data, labels) in enumerate(test_loader):
            total += labels.size(0)

            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()

            output = model(data.float())
            loss = loss_fn(output, labels)
            
            running_loss += loss.item()
            
            for i in range(output.size(0)):
                if output[i].argmax() == labels[i].argmax():
                    correct = correct + 1
                    if config['recordCI']:
                        ci_right[labels[i].argmax()].append(nn.Softmax(0)(output[i]))
                elif config['recordCI']:
                    ci_wrong[labels[i].argmax()].append(nn.Softmax(0)(output[i]))

            bar.next()
    
    test_loss = running_loss/total
    test_accuracy = correct/total
    if subset == 'val':
        print('\nValidation loss: ' + str(test_loss))
        print('Validation accuracy: ' + str(test_accuracy))
    else:
        print('\nTest loss: ' + str(test_loss))
        print('Test accuracy: ' + str(test_accuracy))

    if config['recordCI']:
        for i in range(0, config['numClasses']):
            ci_file = open('ci'+str(i)+'.csv', 'a')
            for j in range(0, max(len(ci_right[i]), len(ci_wrong[i]))):
                if j < len(ci_right[i]):
                    for k in ci_right[i][j]:
                        ci_file.write(str(k.item()) + ',')
                else:
                    ci_file.write(config['numClasses'] * ',')
                
                if j < len(ci_wrong[i]):
                    for k in ci_wrong[i][j]:
                        ci_file.write(str(k.item()) + ',')
                ci_file.write('\n')
            ci_file.close()

def test_regression(config, use_cuda, dataset):
    print('Preparing dataset...')

    if config['channels'] == 1:
        cmap = gray_fog_highlight
    elif config['channels'] == 3:
        cmap = pc_cmap

    test_set = None
    subset = None
    if config['mode'] == 'VALIDATE':
        subset = 'val'
    else:
        subset = 'test'
     
    test_set = dataset(config['dataPath'], subset, config['imgDim'], config['channels'], cmap, 'AVERAGE')
    
    test_loader = DataLoader(test_set, config['batchSize'], collate_fn=dataset.collate_fn)
    
    print('Preparing model...')
    model = torch.jit.load(config['modelPath'], torch.device('cpu'));
    if use_cuda:
        model.cuda()
    
    loss_fn = nn.SmoothL1Loss()

    running_loss = 0.0
    total = 0
    all_outputs = torch.empty((0, 1))
    all_labels = torch.empty((0, 1))
    running_rmse = 0.0

    if use_cuda:
        all_outputs = all_outputs.cuda()
        all_labels = all_labels.cuda()


    with torch.no_grad():
        model.eval()

        bar = Bar()
        bar.max = len(test_loader)

        for step, (data, labels) in enumerate(test_loader):
            total += labels.size(0)
            #data = data[0]

            if use_cuda:
                data = data.cuda()
                labels = labels.to(torch.device('cuda'))
            
            output = model(data.float())
            all_outputs = torch.cat((all_outputs, output))
            all_labels = torch.cat((all_labels, labels))
            loss = loss_fn(output, labels)

            running_loss += loss.item()
            running_rmse += sqrt(nn.MSELoss()(output, labels).item())

            bar.next()

    test_loss = running_loss/total
    test_r2 = torcheval.metrics.functional.r2_score(all_outputs, all_labels).item()
    test_rmse = running_rmse/total
    if subset == 'val':
        print('\nValidation MAE: ' + str(test_loss))
        print('Validation R2: ' + str(test_r2))
        print('Validation RMSE: ' + str(test_rmse))
    else:
        print('\nTest MAE: ' + str(test_loss))
        print('Test R2: ' + str(test_r2))
        print('Test RMSE: ' + str(test_rmse))
#shit