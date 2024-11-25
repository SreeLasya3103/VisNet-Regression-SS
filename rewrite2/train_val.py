import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from progress.bar import ChargingBar
from progress.spinner import Spinner
import progress.counter
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as f
import torcheval.metrics as tem
import torcheval.metrics.functional as temf
from torcheval.metrics.functional import r2_score, multiclass_confusion_matrix
import math
from os import path
import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
import seaborn as sn
import pandas as pd
from itertools import cycle
from torchvision.utils import save_image
import os
import csv

def train_cls(loaders, model, optimizer, loss_fn, epochs, use_cuda, subbatch_count, class_names, output_fn, labels_fn, writer):
    sn.set_theme(font_scale=0.4)

    best_loss = float('inf')

    for epoch in range(epochs):
        print('\nEpoch ' + str(epoch+1))
        print('Training...')
        

        model.train()
        loader = loaders[0]

        all_outputs = None
        all_labels = None

        running_loss = 0.0

        bar = ChargingBar()
        bar.max = len(loader)
        bar.width = 0
        spinner = Spinner()

        for step, (data, labels) in enumerate(loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()

            output = model(data)

            if output_fn:
                output = output_fn(output)
            if labels_fn:
                labels = labels_fn(labels)

            loss = loss_fn(output, labels)
            loss.backward()
            running_loss += loss.item() * labels.size(0) / len(loader.dataset)

            if (step+1) % subbatch_count == 0 or (step+1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()
            
            if step == 0:
                all_outputs = output.detach().clone()
                all_labels = labels.detach().clone()
            else:
                all_outputs = torch.cat((all_outputs, output.detach().clone()), 0)
                all_labels = torch.cat((all_labels, labels.detach().clone()), 0)

            
            bar.next()
            spinner.next()

        all_outputs = all_outputs.cpu()
        all_labels = all_labels.cpu()

        training_loss = running_loss
        training_acc = temf.multiclass_accuracy(all_outputs, torch.argmax(all_labels, 1)).item()
        print('\nTraining loss: ' + str(training_loss))
        print('Training accuracy: ' + str(training_acc))

        training_confmat = multiclass_confusion_matrix(all_outputs, torch.argmax(all_labels, 1), all_labels.size(1), normalize='true')
        tcm = pd.DataFrame(training_confmat, index=class_names, columns=class_names)
        plt.figure(dpi = 300)
        plot = sn.heatmap(tcm, annot=True, vmin=0.0, vmax=1.0)
        plot.set_xlabel('Predicted Value')
        plot.set_ylabel('True Value')
        writer.add_figure('ConfMat/train', plt.gcf(), epoch+1)

        torch.save(model.state_dict(), path.normpath(writer.get_logdir()+'/last.pt'))

        print('Validating...')
        validation_data = valtest_cls(loaders[1], model, loss_fn, use_cuda, output_fn, labels_fn)

        validation_loss = validation_data['loss']
        validation_acc = validation_data['acc']
        print('\nValidation loss: ' + str(validation_loss))
        print('Validation accuracy: ' + str(validation_acc))

        validation_confmat = validation_data['confmat']
        tcm = pd.DataFrame(validation_confmat, index=class_names, columns=class_names)
        plt.figure(dpi = 300)
        plot = sn.heatmap(tcm, annot=True, vmin=0.0, vmax=1.0)
        plot.set_xlabel('Predicted Value')
        plot.set_ylabel('True Value')
        writer.add_figure('ConfMat/val', plt.gcf(), epoch+1)

        if validation_loss < best_loss:
            best_loss == validation_loss
            torch.save(model.state_dict(), path.normpath(writer.get_logdir()+'/best-loss.pt'))

        writer.add_scalar('Loss/train', training_loss, epoch+1)
        writer.add_scalar('Acc/train', training_acc, epoch+1)
        writer.add_scalar('Loss/val', validation_loss, epoch+1)
        writer.add_scalar('Acc/val', validation_acc, epoch+1)

        if len(loaders[2]) > 0:
            print('Testing...')
            testing_data = valtest_cls(loaders[2], model, loss_fn, use_cuda, output_fn, labels_fn)

            testing_loss = testing_data['loss']
            testing_acc = testing_data['acc']

            testing_confmat = testing_data['confmat']
            tcm = pd.DataFrame(testing_confmat, index=class_names, columns=class_names)
            plt.figure(dpi = 300)
            plot = sn.heatmap(tcm, annot=True, vmin=0.0, vmax=1.0)
            plot.set_xlabel('Predicted Value')
            plot.set_ylabel('True Value')
            writer.add_figure('ConfMat/test', plt.gcf(), epoch+1)

            writer.add_scalar('Loss/test', testing_loss, epoch+1)
            writer.add_scalar('Acc/test', testing_acc, epoch+1)

        writer.flush()

    writer.close()


@torch.inference_mode
def valtest_cls(loader, model, loss_fn, use_cuda, output_fn, labels_fn):
    model.eval()

    running_loss = 0.0

    all_outputs = None
    all_labels = None

    bar = ChargingBar()
    bar.max = len(loader)
    bar.width = 0
    spinner = Spinner()

    for step, (data, labels) in enumerate(loader):
        if use_cuda:
            data = data.cuda()
            labels = labels.cuda()

        output = model(data)

        if output_fn:
            output = output_fn(output)
        if labels_fn:
            labels = labels_fn(labels)

        loss = loss_fn(output, labels)
        running_loss += loss.item() * labels.size(0) / len(loader.dataset)
        
        if step == 0:
            all_outputs = output.detach().clone()
            all_labels = labels.detach().clone()
        else:
            all_outputs = torch.cat((all_outputs, output.detach().clone()), 0)
            all_labels = torch.cat((all_labels, labels.detach().clone()), 0)

        bar.next()
        spinner.next()

    all_outputs = all_outputs.cpu()
    all_labels = all_labels.cpu()

    loss = running_loss
    acc = temf.multiclass_accuracy(all_outputs, torch.argmax(all_labels, 1)).item()
    
    confmat = multiclass_confusion_matrix(all_outputs, torch.argmax(all_labels, 1), all_labels.size(1), normalize='true')

    return {'loss':loss, 'acc':acc, 'confmat':confmat}

def train_reg(loaders, model, optimizer, loss_fn, epochs, use_cuda, subbatch_count, output_fn, labels_fn, writer):
    sn.set_theme(font_scale=0.4)

    best_loss = float('inf')

    for epoch in range(epochs):
        print('\nEpoch ' + str(epoch+1))
        print('Training...')
        
        model.train()
        loader = loaders[0]

        all_outputs = None
        all_labels = None

        running_loss = 0.0

        bar = ChargingBar()
        bar.max = len(loader)
        bar.width = 0
        spinner = Spinner()

        for step, (data, labels) in enumerate(loader):
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()

            output = model(data)

            if output_fn:
                output = output_fn(output)
            if labels_fn:
                labels = labels_fn(labels)

            loss = loss_fn(output, labels)
            loss.backward()
            running_loss += loss.item() * labels.size(0) / len(loader.dataset)

            if (step+1) % subbatch_count == 0 or (step+1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()
            
            if step == 0:
                all_outputs = output.detach().clone()
                all_labels = labels.detach().clone()
            else:
                all_outputs = torch.cat((all_outputs, output.detach().clone()), 0)
                all_labels = torch.cat((all_labels, labels.detach().clone()), 0)

            
            bar.next()
            spinner.next()

        all_outputs = all_outputs.cpu()
        all_labels = all_labels.cpu()

        training_loss = running_loss
        training_mae = torch.abs(torch.subtract(all_outputs, all_labels))
        training_mae = torch.mean(training_mae).item()
        training_mse = torch.square(torch.subtract(all_outputs, all_labels))
        training_mse = torch.mean(training_mse).item()
        training_rmse = math.sqrt(training_mse)
        training_mape = torch.abs(torch.subtract(all_outputs, all_labels))
        training_mape = torch.div(training_mape, all_labels)
        training_mape = torch.mean(training_mape).item()

        print('\nTraining loss: ' + str(training_loss))
        print('Training MAE: ' + str(training_mae))
        print('Training MSE: ' + str(training_mse))
        print('Training RMSE: ' + str(training_rmse))
        print('Training MAPE: ' + str(training_mape))

        torch.save(model.state_dict(), path.normpath(writer.get_logdir()+'/last.pt'))

        print('Validating...')
        validation_data = valtest_reg(loaders[1], model, loss_fn, use_cuda, output_fn, labels_fn)

        validation_loss = validation_data['loss']
        validation_mae = validation_data['mae']
        validation_mse = validation_data['mse']
        validation_rmse = validation_data['rmse']
        validation_mape = validation_data['mape']

        print('\nValidation loss: ' + str(validation_loss))
        print('Validation MAE: ' + str(validation_mae))
        print('Validation MSE: ' + str(validation_mse))
        print('Validation RMSE: ' + str(validation_rmse))
        print('Validation MAPE: ' + str(validation_mape))

        if validation_loss < best_loss:
            best_loss == validation_loss
            torch.save(model.state_dict(), path.normpath(writer.get_logdir()+'/best-loss.pt'))

        writer.add_scalar('Loss/train', training_loss, epoch+1)
        writer.add_scalar('MAE/train', training_mae, epoch+1)
        writer.add_scalar('MSE/train', training_mse, epoch+1)
        writer.add_scalar('RMSE/train', training_rmse, epoch+1)
        writer.add_scalar('MAPE/train', training_mape, epoch+1)

        writer.add_scalar('Loss/val', validation_loss, epoch+1)
        writer.add_scalar('MAE/val', validation_mae, epoch+1)
        writer.add_scalar('MSE/val', validation_mse, epoch+1)
        writer.add_scalar('RMSE/val', validation_rmse, epoch+1)
        writer.add_scalar('MAPE/val', validation_mape, epoch+1)


        if len(loaders[2]) > 0:
            print('Testing...')
            testing_data = valtest_reg(loaders[2], model, loss_fn, use_cuda, output_fn, labels_fn)

            testing_loss = testing_data['loss']
            testing_mae = testing_data['mae']
            testing_mse = testing_data['mse']
            testing_rmse = testing_data['rmse']
            testing_mape = testing_data['mape']


            writer.add_scalar('Loss/test', testing_loss, epoch+1)
            writer.add_scalar('MAE/test', testing_mae, epoch+1)
            writer.add_scalar('MSE/test', testing_mse, epoch+1)
            writer.add_scalar('RMSE/test', testing_rmse, epoch+1)
            writer.add_scalar('MAPE/test', testing_mape, epoch+1)

        writer.flush()

    writer.close()


@torch.inference_mode
def valtest_reg(loader, model, loss_fn, use_cuda, output_fn, labels_fn):
    model.eval()

    running_loss = 0.0

    all_outputs = None
    all_labels = None

    bar = ChargingBar()
    bar.max = len(loader)
    bar.width = 0
    spinner = Spinner()

    for step, (data, labels) in enumerate(loader):
        if use_cuda:
            data = data.cuda()
            labels = labels.cuda()

        output = model(data)

        if output_fn:
            output = output_fn(output)
        if labels_fn:
            labels = labels_fn(labels)

        loss = loss_fn(output, labels)
        running_loss += loss.item() * labels.size(0) / len(loader.dataset)
        
        if step == 0:
            all_outputs = output.detach().clone()
            all_labels = labels.detach().clone()
        else:
            all_outputs = torch.cat((all_outputs, output.detach().clone()), 0)
            all_labels = torch.cat((all_labels, labels.detach().clone()), 0)

        bar.next()
        spinner.next()

    all_outputs = all_outputs.cpu()
    all_labels = all_labels.cpu()

    loss = running_loss
    mae = torch.abs(torch.subtract(all_outputs, all_labels))
    mae = torch.mean(mae).item()
    mse = torch.square(torch.subtract(all_outputs, all_labels))
    mse = torch.mean(mse).item()
    rmse = math.sqrt(mse)
    mape = torch.abs(torch.subtract(all_outputs, all_labels))
    mape = torch.div(mape, all_labels)
    mape = torch.mean(mape).item()

    return {'loss':loss, 'mae':mae, 'mse':mse, 'rmse':rmse, 'mape':mape}