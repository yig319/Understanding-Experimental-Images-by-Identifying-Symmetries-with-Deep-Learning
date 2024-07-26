import os
import math
import time
import operator
import datetime
import h5py
from tqdm import tqdm
import wandb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_epochs(rank, world_size, model, loss_func, optimizer, device, train_dl, valid_dl_list, valid_name_list,
                 epochs, start=0, scheduler=None, valid_every_epochs=1, model_dir=None, tracking=False):
    
    setup(rank, world_size)
    
    if isinstance(valid_every_epochs, int) and valid_dl_list != []:
        valid_every_epochs = [valid_every_epochs]*len(valid_dl_list)
    elif isinstance(valid_every_epochs, list) and valid_dl_list == []:
        if len(valid_every_epochs) != len(valid_dl_list):
            raise ValueError("The length of valid_every_epochs should match the number of cross-validation datasets.")
    
    if model_dir and rank == 0 and not os.path.isdir(model_dir): 
        os.mkdir(model_dir)

    history = {'epoch':[], 'train_loss':[], 'valid_loss':[], 'train_acc':[], 'valid_acc':[]}
    if valid_dl_list != []:
        for cv_name in valid_name_list:
            history[cv_name+'_loss'] = []
            history[cv_name+'_acc'] = []
    
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    if tracking and rank == 0:   
        wandb.watch(ddp_model, log_freq=100)
        
    for epoch_idx in range(start, epochs+start):
        if rank == 0:
            print(f"Epoch: {epoch_idx+1}/{epochs+start}")

        train_dl.sampler.set_epoch(epoch)
        avg_train_loss, avg_train_acc = train(ddp_model, loss_func, optimizer, device, train_dl, 
                              scheduler=scheduler, tracking=tracking and rank == 0)
        
        if rank == 0:
            print(f"Training: Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc*100:.4f}%.")
        
        metadata = {'epoch': epoch_idx, 'train_loss': avg_train_loss, 'train_acc': avg_train_acc}

        if valid_dl_list != []:
            for i, (cv_dl, cv_name) in enumerate(zip(valid_dl_list, valid_name_list)):
                if (epoch_idx+1) % valid_every_epochs[i] == 0:
                    avg_cv_loss, avg_cv_acc = valid(ddp_model, loss_func, device, cv_dl, task_label=cv_name, tracking=tracking and rank == 0)
                    metadata[cv_name+'_loss'] = avg_cv_loss
                    metadata[cv_name+'_acc'] = avg_cv_acc
                    if rank == 0:
                        print(f"{cv_name}: Loss: {avg_cv_loss:.4f}, Accuracy: {avg_cv_acc*100:.4f}%.")

        if tracking and rank == 0:   
            wandb.log(metadata)
                
        if model_dir != None and rank == 0:
            torch.save(ddp_model.module.cpu(), os.path.join(model_dir, f'epoch-{epoch_idx+1}.pt'))
        dist.barrier()  # Ensure all processes have finished before moving to the next epoch

        if rank == 0:
            for key, value in metadata.items():
                history[key].append(value)
                
    cleanup()
    return history if rank == 0 else None

def train(model, loss_func, optimizer, device, train_dl, scheduler=None, tracking=False):
    train_data_size = len(train_dl.dataset)
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for i, batch in enumerate(train_dl):
        inputs = batch[0].to(device).float()
        labels = batch[1].to(device).long()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_func(outputs, labels) 

        train_loss += loss.item() * inputs.size(0)
        
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        train_acc += acc.item() * inputs.size(0)
            
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if tracking:   
            wandb.log({"train_loss": loss.item(), "train_acc": acc.item()}) 

    avg_train_loss = train_loss/train_data_size 
    avg_train_acc = train_acc/float(train_data_size)

    return avg_train_loss, avg_train_acc

def valid(model, loss_func, device, valid_dl, task_label='valid', tracking=False):
    valid_data_size = len(valid_dl.dataset)
    valid_loss = 0.0
    valid_acc = 0.0
    
    model.eval()

    with torch.no_grad():
        for j, batch in enumerate(valid_dl):
            inputs = batch[0].float().to(device)
            labels = batch[1].long().to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels) 
            
            valid_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            valid_acc += acc.item() * inputs.size(0)  
            
            if tracking: 
                wandb.log({f"{task_label}_loss": loss.item(), f"{task_label}_acc": acc.item()})
                    
    avg_valid_loss = valid_loss/valid_data_size 
    avg_valid_acc = valid_acc/float(valid_data_size)

    return avg_valid_loss, avg_valid_acc

# def main():
#     world_size = torch.cuda.device_count()
#     mp.spawn(train_epochs, args=(world_size, model, loss_func, optimizer, device, train_dl, valid_dl_list, valid_name_list,
#                                  epochs, start, scheduler, valid_every_epochs, model_dir, tracking), nprocs=world_size, join=True)

# if __name__ == "__main__":
#     main()