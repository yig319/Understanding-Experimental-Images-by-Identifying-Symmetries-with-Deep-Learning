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


def train_epochs(model, loss_func, optimizer, device, train_dl, valid_dl_list, valid_name_list,
                 epochs, start=0, scheduler=None, valid_every_epochs=1, model_dir=None, tracking=False):
    
    if isinstance(valid_every_epochs, int) and valid_dl_list != []:
        valid_every_epochs = [valid_every_epochs]*len(valid_dl_list)
        
    elif isinstance(valid_every_epochs, list) and valid_dl_list == []:
        if len(valid_every_epochs) != len(valid_dl_list):
            raise ValueError("The length of valid_every_epochs should match the number of cross-validation datasets.")
    
    # make directory for the model
    if model_dir and not os.path.isdir(model_dir): 
        os.mkdir(model_dir)

    history = {'epoch':[], 'train_loss':[], 'valid_loss':[], 'train_acc':[], 'valid_acc':[]}
    if valid_dl_list != []:
        for cv_name in valid_name_list:
            history[cv_name+'_loss'] = []
            history[cv_name+'_acc'] = []
    
    if tracking:   
        wandb.watch(model, log_freq=100)
        
    for epoch_idx in range(start, epochs+start):
                
        print("Epoch: {}/{}".format(epoch_idx+1, epochs+start))
        
        avg_train_loss, avg_train_acc = train(model, loss_func, optimizer, device, train_dl, 
                              scheduler=scheduler, tracking=tracking)
        print(f"Training: Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc*100:.4f}%.")
        
        metadata = {'epoch': epoch_idx, 'train_loss': avg_train_loss, 'train_acc': avg_train_acc}

        if valid_dl_list != []:
            for i, (cv_dl, cv_name) in enumerate(zip(valid_dl_list, valid_name_list)):
                if (epoch_idx+1) % valid_every_epochs[i] == 0: # the first element is for the validation set
                    avg_cv_loss, avg_cv_acc = valid(model, loss_func, device, cv_dl, task_label=cv_name, tracking=tracking)
                    metadata[cv_name+'_loss'] = avg_cv_loss
                    metadata[cv_name+'_acc'] = avg_cv_acc
                    print(f"{cv_name}: Loss: {avg_cv_loss:.4f}, Accuracy: {avg_cv_acc*100:.4f}%.")

        if tracking:   
            # record the epoch loss and accuracy:            
            wandb.log(metadata)
                
        if model_dir != None:
            torch.save(model.cpu(), os.path.join(model_dir, 'epoch-{}.pt'.format(epoch_idx+1)))

        # record the epoch loss and accuracy:
        for key, value in metadata.items():
            history[key].append(value)
                
    return history


def train(model, loss_func, optimizer, device, train_dl, scheduler=None, tracking=False):

    train_data_size = len(train_dl.dataset)
    start_time = time.time()

    # Set to training mode
    model.train()

    # Loss and Accuracy within the epoch
    train_loss = 0.0
    train_acc = 0.0

    for i, batch in enumerate(tqdm(train_dl)):
        inputs = batch[0].to(device).float()
        labels = batch[1].to(device).long()
        model = model.to(device)

        # Clean existing gradients
        optimizer.zero_grad()

        # Forward pass - compute outputs on input data using the model
        outputs = model(inputs)

        # Compute loss
        loss = loss_func(outputs, labels) 

        # Compute the total loss for the batch and add it to train_loss
        train_loss += loss.item() * inputs.size(0)
        
        # Compute the accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))

        # Convert correct_counts to float and then compute the mean
        acc = torch.mean(correct_counts.type(torch.FloatTensor))

        # Compute total accuracy in the whole batch and add to train_acc
        train_acc += acc.item() * inputs.size(0)
            
        # Backpropagate the gradients
        loss.backward()

        # Update the parameters
        optimizer.step()
        if scheduler:
            scheduler.step()

        if tracking:   
            wandb.log({ "train_loss": loss.item(), 
                        "train_acc": acc.item()}) 

    # Find average training loss and training accuracy
    avg_train_loss = train_loss/train_data_size 
    avg_train_acc = train_acc/float(train_data_size)

    return avg_train_loss, avg_train_acc


def valid(model, loss_func, device, valid_dl, task_label='valid', tracking=False):

    valid_data_size = len(valid_dl.dataset)

    # Loss and Accuracy within the epoch
    valid_loss = 0.0
    valid_acc = 0.0
    
    start_time = time.time()
        
    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        
        for j, batch in enumerate(tqdm(valid_dl)):
            
            inputs = batch[0].float().to(device)
            labels = batch[1].long().to(device)

            model = model.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_func(outputs, labels) 
            
            # Compute the total loss for the batch and add it to valid_loss
            valid_loss += loss.item() * inputs.size(0)
            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            valid_acc += acc.item() * inputs.size(0)  
            
            if tracking: 
                wandb.log({ task_label+"_loss": loss.item(), 
                            task_label+"_acc": acc.item()})
                    
    # Find average training loss and training accuracy
    avg_valid_loss = valid_loss/valid_data_size 
    avg_valid_acc = valid_acc/float(valid_data_size)

    return avg_valid_loss, avg_valid_acc