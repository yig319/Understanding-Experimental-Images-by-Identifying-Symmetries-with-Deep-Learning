import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from typing import Callable, List
import wandb

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

class DDPTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: List[Callable],
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        save_every: int,
        model_path: str,
        early_stopping_patience: int = None,
        rank: int = 0,
        world_size: int = 1,
        use_data_parallel: bool = False
    ) -> None:
        self.model = model
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.use_data_parallel = use_data_parallel
        
        if self.world_size > 1 and not use_data_parallel:
            self.model = DDP(self.model.to(device), device_ids=[rank])
        elif use_data_parallel:
            if not isinstance(self.model, nn.DataParallel):
                raise ValueError("model must be an instance of nn.DataParallel when use_data_parallel is True")
            devices = [torch.device(f'cuda:{i}') for i in self.model.device_ids]
            if self.device not in devices:
                raise ValueError("device must match the first GPU in model.device_ids")
        else:
            self.model = self.model.to(device)
        
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metrics = metrics
        self.scheduler = scheduler
        self.save_every = save_every
        self.model_path = model_path
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = None
        self.epochs_without_improvement = 0


    def train(self, train_dl, epochs, epoch_start, valid_dl_dict={}, tracking=False):

        # ... existing checks ...
        if not isinstance(valid_dl_dict, dict):
            raise ValueError("valid_dl_dict must be a dictionary")
        if 'valid' not in valid_dl_dict:
            raise ValueError("valid_dl_dict must contain a 'valid' key for early stopping")
        
        for epoch in range(epoch_start, epochs):
            if self.rank == 0 or self.use_data_parallel:
                print(f"Epoch {epoch+1}/{epochs}")
            self.run_epoch(train_dl, dl_name='train', mode='train', tracking=tracking)

            early_stop = False
            for dl_name, dl in valid_dl_dict.items():
                result = self.run_epoch(dl, dl_name, mode='valid', tracking=tracking)
                if dl_name == 'valid':
                    early_stop = result

            if early_stop:
                if self.rank == 0 or self.use_data_parallel:
                    print("Early stopping triggered.")
                break
                
            if self.save_every and (epoch + 1) % self.save_every == 0 and (self.rank == 0 or self.use_data_parallel):
                self.save_model(epoch + 1)

    def validate(self, valid_dl, tracking=False):
        self.run_epoch(valid_dl, mode='valid', tracking=tracking)

    def run_epoch(self, data_loader, dl_name, mode='train', tracking=False):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        data_size = len(data_loader.dataset)
        total_loss = 0.0
        metric_values = {metric.__name__: 0.0 for metric in self.metrics}

        total_samples = 0
        for i, batch in enumerate(tqdm(data_loader, disable=self.rank != 0)):
            batch_size = batch[0].size(0)
            loss, batch_metrics = self.run_batch(batch, mode)
            total_loss += loss * batch_size
            total_samples += batch_size
            for metric_name, metric_value in batch_metrics.items():
                metric_values[metric_name] += metric_value * batch_size

            if tracking and (self.rank == 0 or self.use_data_parallel):
                self.log_wandb({f"{dl_name}_loss": loss, **{f"{dl_name}_{k}": v for k, v in batch_metrics.items()}})

        if self.world_size > 1:
            dist.all_reduce(torch.tensor(total_loss).to(self.device))
            dist.all_reduce(torch.tensor(total_samples).to(self.device))
            for metric_name in metric_values:
                dist.all_reduce(torch.tensor(metric_values[metric_name]).to(self.device))

        avg_loss = total_loss / total_samples
        avg_metrics = {metric: value / total_samples for metric, value in metric_values.items()}

        if self.rank == 0 or self.use_data_parallel:
            print(f"{mode.capitalize()}: Loss: {avg_loss:.4f}, " + ", ".join([f"{metric}: {value * 100:.4f}%" for metric, value in avg_metrics.items()]))

        if mode == 'valid' and self.early_stopping_patience is not None:
            primary_metric = list(self.metrics)[0].__name__
            if self.best_metric is None or avg_metrics[primary_metric] > self.best_metric:
                self.best_metric = avg_metrics[primary_metric]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    return True

        return False

    def run_batch(self, batch, mode):
        inputs = batch[0].to(self.device).float()
        labels = batch[1].to(self.device).long()

        if mode == 'train':
            self.optimizer.zero_grad()

        with torch.set_grad_enabled(mode == 'train'):
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            batch_metrics = {metric.__name__: metric(outputs, labels) for metric in self.metrics}

            if mode == 'train':
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

        # if self.rank == 0:
        #     print(f"Batch - Loss: {loss.item():.4f}, Accuracy: {batch_metrics['accuracy']:.4f}, Grad norm: {grad_norm:.4f}")

        return loss.item(), batch_metrics

    def save_model(self, epoch):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if self.rank == 0 or self.use_data_parallel:
            torch.save(self.model.module.state_dict() if isinstance(self.model, (DDP, nn.DataParallel)) else self.model.state_dict(), 
                    f"{self.model_path}/epoch_{epoch}.pth")
            print(f"Model saved at epoch {epoch}")

    def log_wandb(self, metrics):
        if self.rank == 0:
            wandb.log(metrics)

    def cleanup(self):
        if self.world_size > 1 and not self.use_data_parallel:
            dist.destroy_process_group()