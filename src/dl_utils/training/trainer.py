import os
from typing import List, Callable, Union
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb

def freeze_layers(model, layers_to_freeze):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_func: Union[torch.nn.Module, Callable],
        optimizer: torch.optim.Optimizer,
        metrics: List[Callable],
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        save_per_epochs: int,
        model_path: str,
        early_stopping_patience: int = None,
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metrics = metrics
        self.scheduler = scheduler
        self.save_per_epochs = save_per_epochs
        self.model_path = model_path
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = None
        self.epochs_without_improvement = 0
        self.history = defaultdict(list)
        self.records = 0
        
        if self.model_path:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)

    def train(self, train_dl, epochs, epoch_start, valid_dl_list=None, valid_dl_names=None, valid_per_epochs=1, tracking=False):
        self.valid_per_epochs = valid_per_epochs
        
        if valid_dl_list and valid_dl_names:
            assert len(valid_dl_list) == len(valid_dl_names), "Number of validation dataloaders must match number of names"
        
        for epoch in range(epoch_start, epochs+epoch_start):
            print(f"Epoch {epoch+1}/{epochs+epoch_start}")
            self.run_epoch(epoch, train_dl, mode='train', tracking=tracking)
            if (epoch + 1) % valid_per_epochs == 0:
                if valid_dl_list:
                    early_stop = False
                    for idx, valid_dl in enumerate(valid_dl_list):
                        name = valid_dl_names[idx] if valid_dl_names else f"valid_{idx}"
                        early_stop = self.run_epoch(epoch, valid_dl, name=name, mode='valid', tracking=tracking) or early_stop
                    if early_stop:
                        print("Early stopping triggered.")
                        break 
                    
            # update records for the next epoch
            self.records += len(train_dl.dataset)
                
            if self.save_per_epochs and (epoch + 1) % self.save_per_epochs == 0:
                self.save_model(epoch + 1)

        return dict(self.history)

    def validate(self, valid_dl_list, valid_dl_names=None, tracking=False):
        for idx, valid_dl in enumerate(valid_dl_list):
            name = valid_dl_names[idx] if valid_dl_names else f"valid_{idx}"
            self.run_epoch(valid_dl, mode='valid', tracking=tracking, name=name)

    def run_epoch(self, epoch, data_loader, name=None, mode='train', tracking=False):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        data_size = len(data_loader.dataset)
        records = self.records
        
        total_loss = 0.0
        metric_values = {metric.__name__: 0.0 for metric in self.metrics}

        for batch in tqdm(data_loader):
            loss, batch_metrics = self.run_batch(batch, mode)
            
            if tracking:
                records += len(batch[0])
                # log the loss and metrics for the batch
                batch_info = {'records': records} # make sure the index 0 is the input
                batch_info.update({f"{mode}_{name}_loss" if name else f"{mode}_loss": loss})
                batch_info.update({f"{mode}_{name}_{k}" if name else f"{mode}_{k}": v for k, v in batch_metrics.items()})
                # print('logging to wandb')
                self.log_wandb(batch_info)
            
            total_loss += loss * batch[0].size(0)
            for metric_name, metric_value in batch_metrics.items():
                metric_values[metric_name] += metric_value * batch[0].size(0)

        # Calculate average loss and metrics for the entire epoch
        avg_loss = total_loss / data_size
        avg_metrics = {metric: value / data_size for metric, value in metric_values.items()}

        # Create the information dictionary with the average loss and metrics
        information = {'epoch': epoch + 1}
        information.update({f"{mode}_{name}_loss" if name else f"{mode}_loss": avg_loss})
        information.update({f"{mode}_{name}_{k}" if name else f"{mode}_{k}": v for k, v in avg_metrics.items()})
        dtype_list = []
        for key in information.keys():
            if 'epoch' in key:
                dtype_list.append('int')
            elif 'accuracy' in key:
                dtype_list.append('percent')
            else:
                dtype_list.append('float')
        
        # Print losses
        self.print_losses(information, dtype_list)

        # Update history
        for key, value in information.items():
            self.history[key].append(value)

        # Log to wandb if tracking is enabled
        if tracking:
            # print('logging to wandb')
            self.log_wandb(information)

        if mode == 'valid' and self.early_stopping_patience is not None:
            primary_metric = list(self.metrics)[0].__name__  # Assuming the first metric is the primary one for early stopping
            if self.best_metric is None or avg_metrics[primary_metric] > self.best_metric:
                self.best_metric = avg_metrics[primary_metric]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    return True  # Indicate early stopping
        return False

    def run_batch(self, batch, mode):
        inputs, labels = batch
        inputs = inputs.to(self.device).float()
        labels = labels.to(self.device)

        # Check if labels need to be converted to long
        if isinstance(self.loss_func, (nn.CrossEntropyLoss, nn.NLLLoss)) and not labels.dtype.is_floating_point:
            labels = labels.long()
        
        if mode == 'train':
            self.optimizer.zero_grad()

        with torch.set_grad_enabled(mode == 'train'):
            outputs = self.model(inputs)
            
            # Handle different types of loss functions
            if isinstance(self.loss_func, nn.Module):
                loss = self.loss_func(outputs, labels)
            elif callable(self.loss_func):
                loss = self.loss_func(outputs, labels)
            else:
                raise TypeError("loss_func must be a nn.Module or a callable")

            batch_metrics = {metric.__name__: metric(outputs, labels) for metric in self.metrics}

            if mode == 'train':
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                    
        return loss.item(), batch_metrics

    def save_model(self, epoch):
        model_to_save = self.prepare_model_for_saving(self.model)
        torch.save(model_to_save.state_dict(), f"{self.model_path}/model_epoch_{epoch}.pth")
        print(f"Model saved at epoch {epoch}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def log_wandb(self, metrics):
        # print('logging to wandb')
        # print(metrics)
        wandb.log(metrics)

    def print_losses(self, losses, dtype_list):
        printing_str = ""
        for (dtype, (k, v)) in zip(dtype_list, losses.items()):
            if k == 'epoch':
                continue
            if dtype == 'int':
                printing_str += f"{k}: {v}, "
            elif dtype == 'float':
                if isinstance(v, float) and v < 1e-3:
                    printing_str += f"{k}: {v:.4e}, "
                else:
                    printing_str += f"{k}: {v:.4f}, "
            elif dtype == 'percent':
                printing_str += f"{k}: {v*100:.2f}%, "
        print(printing_str[:-2])

    def prepare_model_for_saving(self, model):
        if isinstance(model, nn.DataParallel):
            return model.module
        if isinstance(model, nn.parallel.DistributedDataParallel):
            return model.module
        if next(model.parameters()).is_cuda:
            return model.cpu()

    def get_history(self):
        return dict(self.history)

    def plot_training_metrics(self):
        plot_info = self.history
        plot_info.pop('epoch')
        
        epochs = range(1, len(plot_info['train_loss']) + 1)
        epochs_short = range(1, len(plot_info['train_loss']) + 1, self.valid_per_epochs)
        fig, axes = plt.subplots(len(plot_info.keys()), 1, figsize=(6, 4*len(plot_info.keys())))
        for i, (metric_name, metric_values) in enumerate(plot_info.items()):
            if 'train' in metric_name:
                axes[i].plot(epochs, metric_values, 'k-')
            elif 'valid' in metric_name:
                axes[i].plot(epochs_short, metric_values, 'k-')
            axes[i].set_title(metric_name)
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel(metric_name)
        plt.tight_layout()
        plt.show()