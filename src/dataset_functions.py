import glob
import random
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from torch.utils.data import Dataset
import h5py
from matplotlib import pyplot as plt
from visualization_functions import show_images

def copy_h5_group(file, copy_from, copy_to, batch_size):
    with h5py.File(file, 'a') as h5:
        # print(h5.keys())
        print(f'Current group {copy_from}\' contains datasets: {list(h5[copy_from].keys())}')
        creaet_group = h5.create_group(copy_to)
        for name in list(h5[copy_from].keys()):
            size = h5[copy_from][name].shape
            dtype = h5[copy_from][name].dtype
            print(f'copying dataset {name} with shape {size} and data type {dtype}')
            create_data = creaet_group.create_dataset(name, shape=size, dtype=dtype)
            for i_start in range(0, size[0], batch_size):
                i_end = np.min((i_start+batch_size, size[0]))
                # print(f'copying index from {i_start} to {i_end}...')
                create_data[i_start:i_end] = np.array(h5[copy_from][name][i_start:i_end])

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

def viz_dataloader(dl, n=8, hist_bins=True, title=None, label_converter=None):
    batch = next(iter(dl))
    if len(batch[0]) < n: 
        raise ValueError("n is smaller than batch size, increase n")
    inputs = batch[0][:n]
    labels = list(batch[1][:n].numpy())
    if label_converter:
        for i in range(len(labels)):
            labels[i] = label_converter[labels[i]]
    show_images(torch.permute(inputs, [0,2,3,1]).cpu().numpy(), labels=labels, title=title, hist_bins=hist_bins)            


class hdf5_dataset(Dataset):
    
    def __init__(self, file_path, folder='train', transform=None, classes=[]):
        self.file_path = file_path
        self.folder = folder
        self.transform = transform
        self.hf = None

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            self.len = len(f[self.folder]['labels'])
        return self.len
    
    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.file_path, 'r')
            
        image = np.array(self.hf[self.folder]['data'][idx])
        labels = np.array(self.hf[self.folder]['labels'][idx])
        
        if self.transform:
            image = self.transform(image)
        return image, labels


class hdf5_dataset_hierarchy(Dataset):
    
    def __init__(self, file_path, folder='train', transform=None, classes=[]):
        self.file_path = file_path
        self.folder = folder
        self.transform = transform
        self.hf = None

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            self.len = len(f[self.folder]['labels'])
        return self.len
    
    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.file_path, 'r')
        
        image = np.array(self.hf[self.folder]['data'][idx])
        labels = np.array(self.hf[self.folder]['labels'][idx])
        labels_l1 = np.array(self.hf[self.folder]['l1_labels'][idx])
        
        if self.transform:
            image = self.transform(image)
        return image, labels_l1, labels