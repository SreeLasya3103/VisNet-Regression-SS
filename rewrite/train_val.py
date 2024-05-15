import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from progress.bar import Bar
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as f
from torcheval.metrics.functional import r2_score
import math

def train_cls(train_set: Dataset, val_set: Dataset, model: nn.Module, params):
    writer = SummaryWriter()
    
    batch_size = params['batch_size']
    use_cuda = params['use_cuda']
    loss_fn = params['loss_fn']
    scheduler = params['scheduler']
    optimizer = params['optimizer']
    epochs = params['epochs']
    num_classes = params['num_classes']
    num_channels = params['num_channels']
    learning_rate = params['learning_rate']
    
    hparams = {
        'model': params['model_name'],
        'dataset': train_set.__class__.__name__,
        'train split': params['split'],
        'loss function': loss_fn.__class__.__name__,
        'optimizer': optimizer.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'batch size': batch_size,
        'epochs': epochs,
        'num classes': num_classes,
        'num channels': num_channels
    }
    
    writer.add_hparams(hparams, {})
    
    train_loader = DataLoader(train_set, batch_size, True)
    
    if use_cuda:
        model.cuda()
        
    for epoch in range(epochs):
        print('\nEpoch ' + str(epoch+1))
        print('Training...')
        
        model.train()
        
        bar = Bar()
        bar.max = len(train_loader)
        
        correct = 0
        total = 0
        running_loss = 0.0
        
        current_lr = 0.0
        if scheduler:
            current_lr = scheduler.get_lr()
        else:
            current_lr = learning_rate
        
        writer.add_hparams({'learning rate': current_lr}, {}, global_step=epoch+1)
        
        for step, (data, labels) in enumerate(train_loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()
            
            total += labels.size(0)
            
            optimizer.zero_grad()
            
            output = model(data)
            loss = loss_fn(output, labels)
            loss.backward()
            running_loss += output.size(0) * loss.item()
            
            optimizer.step()
            
            for i, guess in enumerate(output):
                if guess.argmax() == labels[i].argmax():
                    correct += 1
            
            bar.next()
        
        train_loss = running_loss/total
        train_accuracy = correct/total
        print('\nTraining loss: ' + str(train_loss))
        print('Training accuracy: ' + str(train_accuracy))
        
        val_loss, val_accuracy = val_cls(val_set, batch_size, model, use_cuda, loss_fn)
        
        print('\nValidating loss: ' + str(val_loss))
        print('Validation accuracy: ' + str(val_accuracy))
        
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Acc/train', train_accuracy, epoch+1)
        writer.add_scalar('Loss/val', val_loss, epoch+1)
        writer.add_scalar('Acc/val', val_accuracy, epoch+1)
        writer.flush()
        
        if scheduler:
            scheduler.step()
    
    writer.close()
        
def val_cls(dataset, batch_size, model, use_cuda, loss_fn):
    val_loader = DataLoader(dataset, batch_size, True)
    
    if use_cuda:
        model.cuda()
        
    with torch.inference_mode():
        print('Validating...')
        
        model.eval()
        
        bar = Bar()
        bar.max = len(val_loader)
        
        correct = 0
        total = 0
        running_loss = 0.0
        
        for step, (data, labels) in enumerate(val_loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()
            
            total += labels.size(0)
    
            
            output = model(data)
            loss = loss_fn(output, labels)
            running_loss += output.size(0) * loss.item()
            
            for i, guess in enumerate(output):
                if guess.argmax() == labels[i].argmax():
                    correct += 1
            
            bar.next()
            
        val_loss = running_loss/total
        val_accuracy = correct/total
        
        return val_loss, val_accuracy


    val_loader = DataLoader(dataset, batch_size, True)
    
    if use_cuda:
        model.cuda()
        
    with torch.inference_mode():
        print('Validating...')
        
        model.eval()
        
        bar = Bar()
        bar.max = len(val_loader)
        
        correct = 0
        total = 0
        running_loss = 0.0
        
        for step, (data, labels) in enumerate(val_loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()
            
            total += labels.size(0)
    
            
            output = model(data)
            loss = loss_fn(output, labels)
            running_loss += output.size(0) * loss.item()
            
            for i, guess in enumerate(output):
                if guess.argmax() == labels[i].argmax():
                    correct += 1
            
            bar.next()
            
        val_loss = running_loss/total
        val_accuracy = correct/total
        
        return val_loss, val_accuracy
    
def train_reg(train_set: Dataset, val_set: Dataset, model: nn.Module, params):
    writer = SummaryWriter()
    
    batch_size = params['batch_size']
    use_cuda = params['use_cuda']
    loss_fn = params['loss_fn']
    scheduler = params['scheduler']
    optimizer = params['optimizer']
    epochs = params['epochs']
    num_classes = params['num_classes']
    num_channels = params['num_channels']
    learning_rate = params['learning_rate']
    
    hparams = {
        'model': params['model_name'],
        'dataset': train_set.__class__.__name__,
        'train split': params['split'],
        'loss function': loss_fn.__class__.__name__,
        'optimizer': optimizer.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'batch size': batch_size,
        'epochs': epochs,
        'num classes': num_classes,
        'num channels': num_channels
    }
    
    writer.add_hparams(hparams, {})
    
    train_loader = DataLoader(train_set, batch_size, True)
    
    if use_cuda:
        model.cuda()
        
    for epoch in range(epochs):
        print('\nEpoch ' + str(epoch+1))
        print('Training...')
        
        model.train()
        
        bar = Bar()
        bar.max = len(train_loader)
        
        total = 0
        running_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        running_r2 = 0.0
        
        current_lr = 0.0
        if scheduler:
            current_lr = scheduler.get_lr()
        else:
            current_lr = learning_rate
        
        writer.add_hparams({'learning rate': current_lr}, {}, global_step=epoch+1)
        
        for step, (data, labels) in enumerate(train_loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()
            
            total += labels.size(0)
    
            optimizer.zero_grad()
            
            output = model(data)
            loss = loss_fn(output, labels)
            loss.backward()
            running_loss += output.size(0) * loss.item()
            
            optimizer.step()
            
            running_mae += f.l1_loss(output, labels, reduction='sum').item()
            running_mse += f.mse_loss(output, labels, reduction='sum').item()
            running_r2 += r2_score(output, labels).item() * labels.size(0)
            
            bar.next()
        
        train_loss = running_loss/total
        train_mae = running_mae/total
        train_rmse = math.sqrt(running_mse/total)
        train_r2 = running_r2/total
        print('\nTraining loss: ' + str(train_loss))
        print('Training MAE : ' + str(train_mae))
        print('Training RMSE: ' + str(train_rmse))
        print('Training R2  : ' + str(train_r2))
        
        val_loss, val_mae, val_rmse, val_r2 = val_reg(val_set, batch_size, model, use_cuda, loss_fn)
        
        print('\nValidation loss: ' + str(val_loss))
        print('Validation MAE : ' + str(val_mae))
        print('Validation RMSE: ' + str(val_rmse))
        print('Validation R2  : ' + str(val_r2))
        
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('MAE/train', train_mae, epoch+1)
        writer.add_scalar('RMSE/train', train_rmse, epoch+1)
        writer.add_scalar('R2/train', train_r2, epoch+1)
        writer.add_scalar('Loss/val', val_loss, epoch+1)
        writer.add_scalar('MAE/val', val_mae, epoch+1)
        writer.add_scalar('RMSE/val', val_rmse, epoch+1)
        writer.add_scalar('R2/val', val_r2, epoch+1)
        writer.flush()
        
        if scheduler:
            scheduler.step()
    
    writer.close()
        
def val_reg(dataset, batch_size, model, use_cuda, loss_fn):
    val_loader = DataLoader(dataset, batch_size, True)
    
    if use_cuda:
        model.cuda()
        
    with torch.inference_mode():
        print('Validating...')
        
        model.eval()
        
        bar = Bar()
        bar.max = len(val_loader)
        
        total = 0
        running_loss = 0.0
        running_mae = 0.0
        running_mse = 0.0
        running_r2 = 0.0
        
        for step, (data, labels) in enumerate(val_loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()
            
            total += labels.size(0)
            
            output = model(data)
            loss = loss_fn(output, labels)
            running_loss += output.size(0) * loss.item()

            running_mae += f.l1_loss(output, labels, reduction='sum').item()
            running_mse += f.mse_loss(output, labels, reduction='sum').item()
            running_r2 += r2_score(output, labels).item() * labels.size(0)
            
            bar.next()
        
        val_loss = running_loss/total
        val_mae = running_mae/total
        val_rmse = math.sqrt(running_mse/total)
        val_r2 = running_r2/total
        
        return val_loss, val_mae, val_rmse, val_r2