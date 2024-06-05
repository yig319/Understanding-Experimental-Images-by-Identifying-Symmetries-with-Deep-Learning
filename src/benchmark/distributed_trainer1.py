import os
import shutil
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb
from typing import Callable, List

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)


class DistributedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: List[Callable],
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        
        rank: int,
        world_size: int,
        save_every: int,
        model_path: str,
        
        early_stopping_patience: int = None,
    ) -> None:
        self.model = DDP(model.to(device), device_ids=[rank], output_device=rank)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metrics = metrics
        self.scheduler = scheduler
        self.device = device
        
        self.rank = rank
        self.world_size = world_size
        self.save_every = save_every
        self.model_path = model_path
        
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = None
        self.epochs_without_improvement = 0

    def train(self, train_dl, epochs, valid_dl=None, cv_dl=None, tracking=False):
        for epoch in range(epochs):
            train_dl.sampler.set_epoch(epoch)
    
            if self.rank == 0:
                print(f"Epoch {epoch+1}/{epochs}")
            self.run_epoch(train_dl, mode='train', tracking=tracking)
            if valid_dl:
                valid_dl.sampler.set_epoch(epoch)
                early_stop = self.run_epoch(valid_dl, mode='valid', tracking=tracking)
                if cv_dl:
                    cv_dl.sampler.set_epoch(epoch)
                    _ = self.run_epoch(cv_dl, mode='valid', tracking=tracking)
                if early_stop:
                    if self.rank == 0:
                        print("Early stopping triggered.")
                    break 
                
            if self.save_every and (epoch + 1) % self.save_every == 0 and self.rank == 0:
                self.save_model(epoch + 1)

                
    def validate(self, valid_dl, tracking=False):
        self.run_epoch(valid_dl, mode='valid', tracking=tracking)

    def run_epoch(self, dataloader, mode='train', tracking=False):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        data_size = len(dataloader.dataset)
        total_loss = 0.0
        metric_values = {metric.__name__: 0.0 for metric in self.metrics}

        for i, batch in enumerate(tqdm(dataloader, disable=self.rank != 0)):
            loss, batch_metrics = self.run_batch(batch, mode)
            total_loss += loss * batch[0].size(0)
            for metric_name, metric_value in batch_metrics.items():
                metric_values[metric_name] += metric_value * batch[0].size(0)

            if tracking and self.rank == 0:
                self.log_wandb({f"{mode}_loss": loss, **batch_metrics})

        avg_loss = total_loss / data_size
        avg_metrics = {metric: value / data_size for metric, value in metric_values.items()}
        if self.rank == 0:
            print(f"{mode.capitalize()}: Loss: {avg_loss:.4f}, " + ", ".join([f"{metric}: {value * 100:.4f}%" for metric, value in avg_metrics.items()]))

        if mode == 'valid' and self.early_stopping_patience is not None:
            primary_metric = list(self.metrics)[0].__name__  # Assuming the first metric is the primary one for early stopping
            if self.best_metric is None or avg_metrics[primary_metric] > self.best_metric:
                self.best_metric = avg_metrics[primary_metric]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    return True  # Indicate early stopping

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
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

        return loss.item(), batch_metrics

    def save_model(self, epoch):
        if self.rank == 0:
            torch.save(self.model.module.state_dict(), f"{self.model_path}/model_epoch_{epoch}.pth")
            print(f"Model saved at epoch {epoch}")

    def log_wandb(self, metrics):
        if self.rank == 0:
            wandb.log(metrics)

import sys
from torch.utils.data.distributed import DistributedSampler

sys.path.append('../benchmark/')
sys.path.append('../utils/')
from build_model import resnet50_
from utils import viz_dataloader, hdf5_dataset_stack, split_train_valid, list_to_dict, viz_h5_structure
    
def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = resnet50_(in_channels=10, n_classes=17)

    NAME = 'preprocess-fft_radon_stacked'
    symmetry_classes = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm', 'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']
    label_converter = list_to_dict(symmetry_classes)
    h5_file = '../../datasets/imagenet_atom_noise_v4_rot_10m_100k_subset_transform.h5'

    train_ds = hdf5_dataset_stack(h5_file, folder='imagenet', transform=transforms.ToTensor(),
                            data_keys=['data', 'magnitude_spectrum', 'phase_spectrum', 'radon'], label_key='labels')
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_dl = DataLoader(train_ds, batch_size=600,  num_workers=1, sampler=train_sampler)

    valid_ds = hdf5_dataset_stack(h5_file, folder='imagenet', transform=transforms.ToTensor(),
                            data_keys=['data', 'magnitude_spectrum', 'phase_spectrum', 'radon'], label_key='labels')
    valid_dl = DataLoader(valid_ds, batch_size=600, num_workers=1)
    cv_ds = hdf5_dataset_stack(h5_file, folder='atom', transform=transforms.ToTensor(),
                        data_keys=['data', 'magnitude_spectrum', 'phase_spectrum', 'radon'], label_key='labels')
    cv_dl = DataLoader(cv_ds, batch_size=600, num_workers=1)

    lr = 1e-3
    epochs = 50
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=epochs, max_lr=lr, steps_per_epoch=len(train_dl))

    # outputs = model(torch.randn(2,10,256,256))
    # print(outputs.shape)

    # Create and run the distributed trainer
    # Create and run the trainer
    trainer = DistributedTrainer(model=model, loss_func=loss_func, optimizer=optimizer, metrics=[accuracy], scheduler=scheduler,
                                 device=rank, rank=rank, world_size=world_size, save_every=1, model_path='./')
    trainer.train(train_dl, epochs, valid_dl=valid_dl, cv_dl=cv_dl, tracking=False)

    torch.distributed.destroy_process_group()

if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

    # Clean up the downloaded MNIST dataset and saved model
    # # shutil.rmtree('./data', ignore_errors=True)
    # for epoch in range(0, 11, 5):  # Assuming epochs=10 and save_every=5
    #     model_file = f'./model_epoch_{epoch}.pth'
    #     if os.path.exists(model_file):
    #         os.remove(model_file)
    # print("Data and saved models are cleaned up.")
