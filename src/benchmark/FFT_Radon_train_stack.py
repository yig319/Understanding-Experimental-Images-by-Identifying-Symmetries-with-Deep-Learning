
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

import sys
sys.path.append('../utils/')
from build_model import resnet50_
from utils import viz_dataloader, hdf5_dataset_stack, split_train_valid, list_to_dict, viz_h5_structure
from distributed_trainer import accuracy, DistributedTrainer

def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Example dataset and dataloader
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # train_dl = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)

    lr = 1e-3
    epochs = 50
    batch_size = 128

    h5_file = '../../datasets/imagenet_atom_noise_v4_rot_10m_100k_subset_transform.h5'
    train_ds = hdf5_dataset_stack(h5_file, folder='imagenet', transform=transforms.ToTensor(),
                                  data_keys=['data', 'magnitude_spectrum', 'phase_spectrum', 'radon'], label_key='labels')
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=1, sampler=train_sampler)

    valid_ds = hdf5_dataset_stack(h5_file, folder='atom', transform=transforms.ToTensor(),
                            data_keys=['data', 'magnitude_spectrum', 'phase_spectrum', 'radon'], label_key='labels')
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=1, sampler=valid_sampler)

    cv_ds = hdf5_dataset_stack(h5_file, folder='atom', transform=transforms.ToTensor(),
                        data_keys=['data', 'magnitude_spectrum', 'phase_spectrum', 'radon'], label_key='labels')
    cv_sampler = DistributedSampler(cv_ds, num_replicas=world_size, rank=rank)
    cv_dl = DataLoader(cv_ds, batch_size=batch_size, num_workers=1, sampler=cv_sampler)

    model = resnet50_(in_channels=10, n_classes=17)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=epochs, max_lr=lr, steps_per_epoch=len(train_dl))
    metrics = [accuracy]

    trainer = DistributedTrainer(model=model, loss_func=loss_func, optimizer=optimizer, metrics=[accuracy], scheduler=scheduler,
                                 device=rank, rank=rank, world_size=world_size, save_every=1, model_path='./')
    trainer.train(train_dl, epochs=epochs, valid_dl=valid_dl, cv_dl=cv_dl)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)