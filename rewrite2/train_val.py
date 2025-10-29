import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from progress.bar import ChargingBar
from progress.spinner import Spinner
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
import torcheval.metrics.functional as temf
from torcheval.metrics.functional import multiclass_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import os

def train_cls(loaders, model, optimizer, loss_fn, epochs, use_cuda, subbatch_count, class_names, output_fn, labels_fn, writer, transform):
    sns.set_theme(font_scale=0.4)
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")

        # ─── TRAIN ────────────────────────────────────────
        model.train()
        train_loader = loaders[0]
        all_outputs, all_labels = [], []
        running_loss = 0.0

        bar = ChargingBar("Training", max=len(train_loader), width=0)
        for step, (data, labels, _) in enumerate(train_loader):
            if use_cuda:
                data, labels = data.cuda(), labels.cuda()
            # Wrap single image type into a 3-type input if needed
            #if data.ndim == 4:
                #data = data.unsqueeze(1)  # [B, 1, C, H, W]
                #data = data.repeat(1, 3, 1, 1, 1)  # [B, 3, C, H, W]
                #data = data.permute(1, 0, 2, 3, 4)  # [3, B, C, H, W]
            if data.ndim == 4:  # [B, C, H, W]
            #Apply transform manually to each image
              batch = []
              for i in range(data.size(0)):
                  batch.append(transform(data[i]))  # output: [3, C, H, W]
                  data = torch.stack(batch, dim=1)  # [3, B, C, H, W]
            output = model(data)
            if output_fn: output = output_fn(output)
            if labels_fn: labels = labels_fn(labels)

            loss = loss_fn(output, labels)
            loss.backward()
            running_loss += loss.item() * labels.size(0) / len(train_loader.dataset)

            if (step + 1) % subbatch_count == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            all_outputs.append(output.detach().cpu())
            all_labels.append(labels.detach().cpu())
            bar.next()
        bar.finish()

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        train_loss = running_loss
        train_acc = temf.multiclass_accuracy(all_outputs, torch.argmax(all_labels, 1)).item()
        print(f"Training loss: {train_loss:.4f} | accuracy: {train_acc:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Acc/train', train_acc, epoch+1)

        cm = multiclass_confusion_matrix(all_outputs, torch.argmax(all_labels, 1), all_labels.size(1), normalize='true')
        _log_confmat(cm, class_names, writer, f'ConfMat/train', epoch+1)

        torch.save(model.state_dict(), os.path.normpath(writer.get_logdir() + '/last.pt'))

        # ─── VALIDATION ──────────────────────────────────
        print('Validating...')
        val_data = valtest_cls(loaders[1], model, loss_fn, use_cuda, output_fn, labels_fn, class_names, writer, epoch+1, stage='val')
        writer.add_scalar('Loss/val', val_data['loss'], epoch+1)
        writer.add_scalar('Acc/val', val_data['acc'], epoch+1)

        # ─── TESTING ─────────────────────────────────────
        print('Testing...')
        test_data = valtest_cls(loaders[2], model, loss_fn, use_cuda, output_fn, labels_fn, class_names, writer, epoch+1, stage='test')
        writer.add_scalar('Loss/test', test_data['loss'], epoch+1)
        writer.add_scalar('Acc/test', test_data['acc'], epoch+1)

        if test_data['loss'] < best_loss:
            best_loss = test_data['loss']
            torch.save(model.state_dict(), os.path.normpath(writer.get_logdir() + '/best-loss.pt'))

            with open(os.path.normpath(writer.get_logdir() + '/confs.csv'), 'w') as f:
                f.write(test_data['confs'])

    writer.close()
    return test_data


@torch.inference_mode()
def valtest_cls(loader, model, loss_fn, use_cuda, output_fn, labels_fn, class_names, writer, epoch, stage='test'):
    model.eval()
    all_outputs, all_labels, all_paths = [], [], []
    running_loss = 0.0

    bar = ChargingBar(f"{stage.capitalize()}", max=len(loader), width=0)
    for step, (data, labels, paths) in enumerate(loader):
        if use_cuda:
            data, labels = data.cuda(), labels.cuda()
        output = model(data)
        if output_fn: output = output_fn(output)
        if labels_fn: labels = labels_fn(labels)

        loss = loss_fn(output, labels)
        running_loss += loss.item() * labels.size(0) / len(loader.dataset)

        all_outputs.append(output.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_paths += paths
        bar.next()
    bar.finish()

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    sm = F.softmax(all_outputs, dim=1)

    loss = running_loss
    acc = temf.multiclass_accuracy(all_outputs, torch.argmax(all_labels, 1)).item()
    confmat = multiclass_confusion_matrix(all_outputs, torch.argmax(all_labels, 1), all_labels.size(1), normalize='true')

    confidences_string = 'predicted,truth,output,top,path\n'
    site_preds, site_labels = defaultdict(list), defaultdict(list)

    for i in range(sm.size(0)):
        pred = torch.argmax(sm[i]).item()
        true = torch.argmax(all_labels[i]).item()
        path = all_paths[i]
        confidences_string += f"{pred},{true},\"{sm[i]}\",{sm[i][pred].item()},{path}\n"
        site_id = path.split('_')[0]
        site_preds[site_id].append(pred)
        site_labels[site_id].append(true)

    # Per-site evaluation
    site_metrics = []
    for site_id in site_preds:
        preds = site_preds[site_id]
        trues = site_labels[site_id]
        report = classification_report(trues, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(trues, preds, labels=list(range(len(class_names))))
        site_metrics.append({
            'site': site_id,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'confusion_matrix': cm.tolist()
        })

        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        import matplotlib.pyplot as plt
        plt.close('all') 
        fig = plt.figure(dpi=300)
        sns.heatmap(df_cm, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Blues")
        plt.title(f'{stage.upper()} ConfMat - {site_id}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        writer.add_figure(f'PerSite/{stage}_ConfMat/{site_id}', fig, global_step=epoch)
        plt.close(fig)

    # Save results
    df = pd.DataFrame(site_metrics)
    df.drop(columns='confusion_matrix').to_csv(f"site_metrics_{stage}.csv", index=False)
    with open(f"site_confmats_{stage}.json", "w") as f:
        json.dump(site_metrics, f, indent=2)

    return {
        'loss': loss,
        'acc': acc,
        'confmat': confmat,
        'confs': confidences_string,
        'site_metrics': site_metrics
    }

from torcheval.metrics.functional import r2_score, mean_squared_error

def train_reg(loaders, model, optimizer, loss_fn, epochs, use_cuda, subbatch_count,
              output_fn, labels_fn, writer, transform, buckets=None, class_names=None):

    model.train()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        running_loss = 0.0
        all_preds = []
        all_targets = []

        train_loader = loaders[0]
        bar = ChargingBar("Training", max=len(train_loader), width=0)
        for step, (data, labels, _) in enumerate(train_loader):
            if use_cuda:
                data = data.cuda()
                if isinstance(labels, list):
                    labels = torch.stack(labels).float().cuda()
                else:
                    labels = labels.float().cuda()


            if data.ndim == 4:  # [B, C, H, W]
                data = torch.stack([transform(img) for img in data], dim=0).permute(1, 0, 2, 3, 4)

            preds = model(data).squeeze()
            targets = labels.squeeze()
            loss = loss_fn(preds, targets)

            loss.backward()
            running_loss += loss.item() * targets.numel()  # num elements instead of len()

            if (step + 1) % subbatch_count == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            all_preds.append(preds.detach().cpu().view(-1))
            all_targets.append(targets.detach().cpu().view(-1))

            bar.next()
        bar.finish()

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_r2 = r2_score(preds, targets).item()
        epoch_mse = mean_squared_error(preds, targets).item()

        print(f"Train Epoch {epoch+1} | Loss: {epoch_loss:.4f} | R²: {epoch_r2:.4f} | MSE: {epoch_mse:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch+1)
        writer.add_scalar('R2/train', epoch_r2, epoch+1)
        writer.add_scalar('MSE/train', epoch_mse, epoch+1)

    return {'r2': epoch_r2, 'mse': epoch_mse}



def _log_confmat(confmat, class_names, writer, tag, epoch):
    df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names)
    import matplotlib.pyplot as plt
    plt.close('all') 
    fig = plt.figure(dpi=300)
    sns.heatmap(df_cm, annot=True, fmt=".2f", vmin=0.0, vmax=1.0, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(tag)
    plt.tight_layout()
    writer.add_figure(tag, fig, global_step=epoch)
    plt.close(fig)