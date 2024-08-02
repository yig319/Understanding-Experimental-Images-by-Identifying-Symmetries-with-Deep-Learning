import os
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import sys
sys.path.append('../../src/benchmark/')
sys.path.append('../../src/utils/')
from build_model import xcit_small
from train_functions import train_epochs
from utils import split_train_valid, list_to_dict, viz_dataloader, hdf5_dataset

def setup(rank, world_size, gpu_indices):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    gpu = gpu_indices[rank]
    torch.cuda.set_device(gpu)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return gpu

def cleanup():
    dist.destroy_process_group()

def prepare_dataloader(dataset, batch_size, num_workers, shuffle=True):
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )

def run(rank, world_size, gpu_indices):
    gpu = setup(rank, world_size, gpu_indices)

    symmetry_classes = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm', 'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']
    label_converter = list_to_dict(symmetry_classes)

    bs = 64 // world_size  # Adjust batch size for multiple GPUs
    num_workers = 3 // world_size  # Adjust number of workers

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])


    # imagenet
    imagenet_ds = hdf5_dataset('../../datasets/imagenet_v5_rot_10m.h5', folder='train', transform=transform)
    train_ds, valid_ds = split_train_valid(imagenet_ds, 0.8)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    # atom
    atom_ds = hdf5_dataset('../../datasets/atom_v5_rot_200k.h5', folder='test', transform=transform)
    atom_dl = DataLoader(atom_ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    # noise
    noise_ds = hdf5_dataset('../../datasets/imagenet_atom_noise_v4_rot_10m_100k_subset.h5', folder='noise', transform=transform)
    noise_dl = DataLoader(noise_ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    model = xcit_small(3, 17)
    device = torch.device(f'cuda:{gpu}')
    model = model.to(device)
    model = DDP(model, device_ids=[gpu])


    NAME = '07282024-benchmark-XCiT-v4_10m'
    device = torch.device(f'cuda:{rank}')
    lr = 1e-3
    start = 0
    epochs = 20

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=epochs, max_lr=lr, steps_per_epoch=len(train_dl))

    history = train_epochs(model, loss_func, optimizer, device, train_dl, 
                           valid_dl_list=[valid_dl, atom_dl, noise_dl], 
                           valid_name_list=['validation', 'atom', 'noise'],
                           epochs=epochs, start=start, scheduler=scheduler, 
                           valid_every_epochs=5, model_dir=f'../../saved_models/{NAME}/', 
                           tracking=False)

    cleanup()
