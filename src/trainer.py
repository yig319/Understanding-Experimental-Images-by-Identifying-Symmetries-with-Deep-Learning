import os
import time
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset

class Trainer:
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

    def _run_epoch(self, epoch, dataloder, mode='train', tracking=False):
                
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        if mode == 'train':
            self.model.train()
        if mode == 'valid':
            self.model.eval()
        if mode == 'test':
            self.model.eval()
            
        b_sz = len(next(iter(dataloder))[0])
        # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(dataloder)}")
        # dataloder.sampler.set_epoch(epoch)
        for inputs, targets in tqdm(dataloder):
            inputs = inputs.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            batch_loss, output = self._run_batch(inputs, targets, mode=mode)
            
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

        print(f"{mode} Loss: {avg_loss:.4f}, {mode} Accuracy: {avg_acc*100:.2f}%")

        if tracking:
            self._wandb_log(epoch, avg_loss, avg_acc, mode)
            

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"{self.model_path}epoch_{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
        
    def _wandb_log(self, epoch, avg_loss, avg_acc, mode):
        # record the epoch loss and accuracy:            
        wandb.log({"epoch": epoch,
                    f"{mode}_loss": avg_loss, 
                    f"{mode}_acc": avg_acc}) 

    def train(self, max_epochs: int, tracking: bool=False, start_epoch: int=1):
        
        if tracking:   
            wandb.watch(self.model, log_freq=100)
        
        if self.model_path and not os.path.isdir(self.model_path): os.mkdir(self.model_path)
        for epoch in range(start_epoch, max_epochs+start_epoch):
            
            print(f"Epoch: {epoch}/{max_epochs+start_epoch-1}:")

            self._run_epoch(epoch, self.train_dl, mode='train', tracking=tracking)
                
            if self.valid_dl != None:
                self._run_epoch(epoch, self.valid_dl, mode='valid', tracking=tracking)
            if self.test_dl != None:
                self._run_epoch(epoch, self.test_dl, mode='test', tracking=tracking)
                
            # if self.gpu_id == 0 and epoch % self.save_every == 0 and self.model_path:
                # self._save_checkpoint(epoch)
                
    def valid(self, tracking: bool=False):
        self._run_epoch(0, mode='valid', tracking=tracking)