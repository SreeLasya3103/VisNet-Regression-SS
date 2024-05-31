import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from progress.bar import Bar
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as f
from torcheval.metrics.functional import r2_score, multiclass_confusion_matrix
import math
from os import path
import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
import seaborn as sn
import pandas as pd

def train_cls(train_set: Dataset, val_set: Dataset, test_set: Dataset, model: nn.Module, params):
    writer = SummaryWriter()
    sn.set_theme(font_scale=0.4)

    subbatch_size = params['subbatch_size']
    accum_steps = params['accum_steps']
    batch_size = accum_steps * subbatch_size
    use_cuda = params['use_cuda']
    loss_fn = params['loss_fn']
    scheduler = params['scheduler']
    optimizer = params['optimizer']
    epochs = params['epochs']
    num_classes = params['num_classes']
    class_names = params['class_names']
    num_channels = params['num_channels']
    learning_rate = params['learning_rate']
    image_dim = params['image_dim']

    hparams = {
        'model': params['model_name'],
        'dataset': params['dset_name'],
        'split': str(params['split']),
        'loss function': loss_fn.__class__.__name__,
        'learning rate': str(learning_rate),
        'optimizer': optimizer.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'batch size': str(batch_size),
        'epochs': str(epochs),
        'classes': str(num_classes),
        'channels': str(num_channels),
        'image dimensions': str(image_dim)
    }

    writer.add_hparams(hparams, {})

    train_loader = DataLoader(train_set, subbatch_size, True)

    if use_cuda:
        model.cuda()

    best_loss = float('-inf')

    for epoch in range(epochs):
        print('\nEpoch ' + str(epoch+1))
        print('Training...')

        model.train()

        bar = Bar()
        bar.max = len(train_loader)

        correct = 0
        total = 0
        running_loss = 0.0

        pred_indices = None
        targ_indices = None

        for step, (data, labels) in enumerate(train_loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()

            total += labels.size(0)

            output = model(data)
            loss = loss_fn(output, labels) / accum_steps
            loss.backward()
            running_loss += labels.size(0) * loss.item()

            for i, guess in enumerate(output):
                if step == 0:
                    pred_indices = torch.Tensor([guess.argmax().detach()])
                    targ_indices = torch.Tensor([labels[i].argmax()])
                else:
                    pred_indices = torch.cat((pred_indices, torch.Tensor([guess.argmax().detach()])))
                    targ_indices = torch.cat((targ_indices, torch.Tensor([labels[i].argmax()])))
                if guess.argmax() == labels[i].argmax():
                    correct += 1

            if (step+1) % accum_steps == 0 or (step+1) == len(train_loader):
                print('stepppp')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            bar.next()

        train_loss = running_loss/total
        train_accuracy = correct/total
        print('\nTraining loss: ' + str(train_loss))
        print('Training accuracy: ' + str(train_accuracy))
        
        train_conf_mat = multiclass_confusion_matrix(pred_indices.to(torch.int64), targ_indices.to(torch.int64), num_classes, normalize='true')
        tcm = pd.DataFrame(train_conf_mat, index=class_names, columns=class_names)
        plot = sn.heatmap(tcm, annot=True, vmin=0.0, vmax=1.0)
        plot.set_xlabel('Predicted Value')
        plot.set_ylabel('True Value')
        writer.add_figure('ConfMat/train', plt.gcf(), epoch+1)

        print('Validating...')
        val_loss, val_accuracy, val_conf_mat = val_cls(val_set, subbatch_size, accum_steps, model, use_cuda, loss_fn, num_classes)

        print('\nValidation loss: ' + str(val_loss))
        print('Validation accuracy: ' + str(val_accuracy))

        vcm = pd.DataFrame(val_conf_mat, index=class_names, columns=class_names)
        plot = sn.heatmap(vcm, annot=True, vmin=0.0, vmax=1.0)
        plot.set_xlabel('Predicted Value')
        plot.set_ylabel('True Value')
        writer.add_figure('ConfMat/val', plt.gcf(), epoch+1)

        torch.save(model.state_dict(), path.normpath(writer.get_logdir()+'/last.pt'))

        if val_loss > best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), path.normpath(writer.get_logdir()+'/best-loss.pt'))

        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Acc/train', train_accuracy, epoch+1)
        writer.add_scalar('Loss/val', val_loss, epoch+1)
        writer.add_scalar('Acc/val', val_accuracy, epoch+1)
        if test_set.__len__() > 0:
            print('Testing...')
            test_loss, test_accuracy, test_conf_mat = val_cls(test_set, subbatch_size, accum_steps, model, use_cuda, loss_fn, num_classes)
            
            tcm = pd.DataFrame(test_conf_mat, index=class_names, columns=class_names)
            plot = sn.heatmap(tcm, annot=True, vmin=0.0, vmax=1.0)
            plot.set_xlabel('Predicted Value')
            plot.set_ylabel('True Value')
            writer.add_figure('ConfMat/test', plt.gcf(), epoch+1)
            
            writer.add_scalar('Loss/test', test_loss, epoch+1)
            writer.add_scalar('Acc/test', test_accuracy, epoch+1)
        writer.flush()

        if scheduler:
            scheduler.step()

    writer.close()

def val_cls(dataset, batch_size, accum_steps, model, use_cuda, loss_fn, num_classes):
    val_loader = DataLoader(dataset, batch_size, True)

    if use_cuda:
        model.cuda()

    with torch.inference_mode():
        model.eval()

        bar = Bar()
        bar.max = len(val_loader)

        correct = 0
        total = 0
        running_loss = 0.0
        
        pred_indices = None
        targ_indices = None

        for step, (data, labels) in enumerate(val_loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()

            total += labels.size(0)

            output = model(data)
            loss = loss_fn(output, labels) / accum_steps
            running_loss += labels.size(0) * loss.item()

            for i, guess in enumerate(output):
                if step == 0:
                    pred_indices = torch.Tensor([guess.argmax().detach()])
                    targ_indices = torch.Tensor([labels[i].argmax()])
                else:
                    pred_indices = torch.cat((pred_indices, torch.Tensor([guess.argmax().detach()])))
                    targ_indices = torch.cat((targ_indices, torch.Tensor([labels[i].argmax()])))
                if guess.argmax() == labels[i].argmax():
                    correct += 1

            bar.next()

        val_loss = running_loss/total
        val_accuracy = correct/total
        
        val_conf_mat = multiclass_confusion_matrix(pred_indices.to(torch.int64), targ_indices.to(torch.int64), num_classes, normalize='true')

        return val_loss, val_accuracy, val_conf_mat

def train_reg(train_set: Dataset, val_set: Dataset, test_set: Dataset, model: nn.Module, params):
    writer = SummaryWriter()

    subbatch_size = params['subbatch_size']
    accum_steps = params['accum_steps']
    batch_size = accum_steps * subbatch_size
    use_cuda = params['use_cuda']
    loss_fn = params['loss_fn']
    scheduler = params['scheduler']
    optimizer = params['optimizer']
    epochs = params['epochs']
    num_classes = params['num_classes']
    num_channels = params['num_channels']
    learning_rate = params['learning_rate']
    image_dim = params['image_dim']

    hparams = {
        'model': params['model_name'],
        'dataset': params['dset_name'],
        'split': str(params['split']),
        'loss function': loss_fn.__class__.__name__,
        'learning rate': str(learning_rate),
        'optimizer': optimizer.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'batch size': str(batch_size),
        'epochs': str(epochs),
        'classes': str(num_classes),
        'channels': str(num_channels),
        'image dimensions': str(image_dim)
    }

    writer.add_hparams(hparams, {})

    train_loader = DataLoader(train_set, subbatch_size, True)

    if use_cuda:
        model.cuda()

    best_r2 = float('-inf')

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

        all_outputs = torch.empty((0, 1))
        all_labels = torch.empty((0, 1))
        if use_cuda:
            all_outputs = all_outputs.cuda()
            all_labels = all_labels.cuda()


        for step, (data, labels) in enumerate(train_loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()

            total += labels.size(0)

            output = model(data)
            loss = loss_fn(output, labels) / accum_steps
            loss.backward()

            all_outputs = torch.cat((all_outputs, output.detach()))
            all_labels = torch.cat((all_labels, labels))

            running_loss += labels.size(0) * loss.item()
            running_mae += f.l1_loss(output, labels, reduction='sum').item()
            running_mse += f.mse_loss(output, labels, reduction='sum').item()

            if (step+1) % accum_steps == 0 or (step+1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            bar.next()

        train_loss = running_loss/total
        train_mae = running_mae/total
        train_rmse = math.sqrt(running_mse/total)
        train_r2 = r2_score(all_outputs, all_labels).item()
        print('\nTraining loss: ' + str(train_loss))
        print('Training MAE : ' + str(train_mae))
        print('Training RMSE: ' + str(train_rmse))
        print('Training R2  : ' + str(train_r2))

        print('Validating...')
        val_loss, val_mae, val_rmse, val_r2 = val_reg(val_set, subbatch_size, accum_steps, model, use_cuda, loss_fn)

        print('\nValidation loss: ' + str(val_loss))
        print('Validation MAE : ' + str(val_mae))
        print('Validation RMSE: ' + str(val_rmse))
        print('Validation R2  : ' + str(val_r2))

        torch.save(model.state_dict(), path.normpath(writer.get_logdir()+'/last.pt'))

        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), path.normpath(writer.get_logdir()+'/best-r2.pt'))

        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('MAE/train', train_mae, epoch+1)
        writer.add_scalar('RMSE/train', train_rmse, epoch+1)
        writer.add_scalar('R2/train', train_r2, epoch+1)
        writer.add_scalar('Loss/val', val_loss, epoch+1)
        writer.add_scalar('MAE/val', val_mae, epoch+1)
        writer.add_scalar('RMSE/val', val_rmse, epoch+1)
        writer.add_scalar('R2/val', val_r2, epoch+1)
        if test_set.__len__() > 0:
            print('Testing...')
            test_loss, test_mae, test_rmse, test_r2 = val_reg(test_set, subbatch_size, accum_steps, model, use_cuda, loss_fn)
            writer.add_scalar('Loss/test', test_loss, epoch+1)
            writer.add_scalar('MAE/test', test_mae, epoch+1)
            writer.add_scalar('RMSE/test', test_rmse, epoch+1)
            writer.add_scalar('R2/test', test_r2, epoch+1)
        writer.flush()

        if scheduler:
            scheduler.step()

    writer.close()

def val_reg(dataset, batch_size, accum_steps, model, use_cuda, loss_fn):
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

        all_outputs = torch.empty((0, 1))
        all_labels = torch.empty((0, 1))
        if use_cuda:
            all_outputs = all_outputs.cuda()
            all_labels = all_labels.cuda()

        for step, (data, labels) in enumerate(val_loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()

            total += labels.size(0)

            output = model(data)
            loss = loss_fn(output, labels) / accum_steps

            all_outputs = torch.cat((all_outputs, output.detach()))
            all_labels = torch.cat((all_labels, labels))

            running_loss += labels.size(0) * loss.item()
            running_mae += f.l1_loss(output, labels, reduction='sum').item()
            running_mse += f.mse_loss(output, labels, reduction='sum').item()

            bar.next()

        val_loss = running_loss/total
        val_mae = running_mae/total
        val_rmse = math.sqrt(running_mse/total)
        val_r2 = r2_score(all_outputs, all_labels).item()

        return val_loss, val_mae, val_rmse, val_r2