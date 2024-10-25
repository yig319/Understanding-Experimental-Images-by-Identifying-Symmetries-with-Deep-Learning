import os
import time
from tqdm import tqdm
import wandb
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def resnet50_yichen(in_channels, n_classes, pretrained=False):
    model = models.resnet50(pretrained=pretrained)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(nn.BatchNorm1d(2048),
                            nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features = 2048, out_features=512, bias=False),
                            nn.ReLU(inplace=True),

                            nn.BatchNorm1d(512),
                            nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features = 512, out_features=64, bias=False),
                            nn.ReLU(inplace=True),
                            
                            nn.BatchNorm1d(64),
                            nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features=64, out_features=n_classes, bias=True)
                            )
    return model

def split_train_valid(dataset, train_ratio, seed=42):
    imagenet_size = len(dataset)
    train_size = int(train_ratio * imagenet_size)
    valid_size = imagenet_size - train_size
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, valid_size], 
                                                       generator=torch.Generator().manual_seed(seed))
    return train_ds, valid_ds

def list_to_dict(lst):
    dictionary = {}
    for index, item in enumerate(lst):
        dictionary[index] = item
    return dictionary

class hdf5_dataset(Dataset):
    
    def __init__(self, file_path, folder='train', transform=None, data_key='data', label_key='labels'):
        self.file_path = file_path
        self.folder = folder
        self.transform = transform
        self.hf = None
        self.data_key = data_key
        self.label_key = label_key

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            self.len = len(f[self.folder]['labels'])
        return self.len
    
    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.file_path, 'r')
            
        image = np.array(self.hf[self.folder][self.data_key][idx])
        labels = np.array(self.hf[self.folder][self.label_key][idx])
        
        if self.transform:
            image = self.transform(image)
        return image, labels


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer_ddp:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        test_dl: DataLoader,
        loss_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        gpu_id: torch.device,
        save_every: int,
        model_path: str,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.model_path = model_path

    def _run_batch(self, source, targets, mode='train'):
        if mode == 'train':
            self.optimizer.zero_grad()
            
        output = self.model(source)
        loss = self.loss_func(output, targets)
        batch_loss = loss.item()
        
        if mode == 'train':
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
                
        return batch_loss, output

    def _run_epoch(self, epoch, mode='train', tracking=False):
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        if mode == 'train':
            dataloder = self.train_dl
            self.model.train()
        if mode == 'valid':
            dataloder = self.valid_dl
            self.model.eval()
        if mode == 'test':
            dataloder = self.test_dl
            self.model.eval()
            
        b_sz = len(next(iter(dataloder))[0])
        # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(dataloder)}")
        dataloder.sampler.set_epoch(epoch)
        for inputs, targets in tqdm(dataloder):
            inputs = inputs.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            batch_loss, output = self._run_batch(inputs, targets)
            
            # Compute the total loss for the batch and add it to train_loss
            epoch_loss += batch_loss * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(output.data, 1)
            correct_counts = predictions.eq(targets.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            epoch_acc += acc.item() * inputs.size(0)
            
        # Find average training loss and training accuracy
        avg_loss = epoch_loss/len(dataloder.dataset) 
        avg_acc = epoch_acc/float(len(dataloder.dataset))

        print(f"Epoch {epoch}: {mode} Loss: {avg_loss:.4f}, {mode} Accuracy: {avg_acc*100:.2f}%")

        if tracking:
            self._wandb_log(epoch, avg_loss, avg_acc, mode)
            

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"{self.model_path}-checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
        
    def _wandb_log(self, epoch, avg_loss, avg_acc, mode):
        # record the epoch loss and accuracy:            
        wandb.log({"epoch": epoch,
                    f"{mode}_loss": avg_loss, 
                    f"{mode}_acc": avg_acc}) 

    def train(self, max_epochs: int, tracking: bool=False):
        if tracking:   
            wandb.watch(self.model, log_freq=100)
        
        if self.model_path and not os.path.isdir(self.model_path): os.mkdir(self.model_path)
        for epoch in range(max_epochs):
            self._run_epoch(epoch, mode='train', tracking=tracking)
            if self.gpu_id == 0 and epoch % self.save_every == 0 and self.model_path:
                self._save_checkpoint(epoch)
                
            if self.valid_dl != None:
                self._run_epoch(epoch, mode='valid', tracking=tracking)
            if self.test_dl != None:
                self._run_epoch(epoch, mode='test', tracking=tracking)
                
    def valid(self, tracking: bool=False):
        self._run_epoch(0, mode='valid', tracking=tracking)
        

def prepare_dataloader(batch_size):
    
    train_ds = hdf5_dataset('../../datasets/imagenet_atom_noise_v4_rot_10m_100k_subset.h5', folder='imagenet', transform=transforms.ToTensor())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4,
                          sampler=DistributedSampler(train_ds))
    return train_dl, None, None

def load_train_objs(total_epochs, lr, train_dl):
    model = resnet50_yichen(in_channels=3, n_classes=17)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=total_epochs, max_lr=lr, steps_per_epoch=len(train_dl))
    return model, loss_func, optimizer, scheduler


def main(rank: int, world_size: int, total_epochs: int, batch_size: int, lr: float=1e-3, tracking: bool=False):
    ddp_setup(rank, world_size)
    train_dl, valid_dl, test_dl = prepare_dataloader(batch_size)
    model, loss_func, optimizer, scheduler = load_train_objs(total_epochs, lr, train_dl)
    
    trainer = Trainer_ddp(model, train_dl, valid_dl, test_dl, loss_func, optimizer, scheduler,
                          rank, save_every=10, model_path=None)
        
    trainer.train(total_epochs, tracking=tracking)
    destroy_process_group()
    
    
if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='simple distributed training job')
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    # args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    save_every = 10
    total_epochs = 50
    batch_size = 64
    mp.spawn(main, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size)