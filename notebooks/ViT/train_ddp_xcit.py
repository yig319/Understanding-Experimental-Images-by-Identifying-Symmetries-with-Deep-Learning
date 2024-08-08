import os
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms

# Import your custom modules
import sys
sys.path.append('../../src/benchmark/')
sys.path.append('../../src/utils/')
from build_model import xcit_small
from trainer_ddp import DDPTrainer
from utils import split_train_valid, list_to_dict, viz_dataloader, hdf5_dataset

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

def setup(rank, world_size, gpu_ids):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, world_size, gpu_ids, num_workers, batch_size, learning_rate, num_epochs, tracking=False):
    setup(rank, world_size, gpu_ids)
    device = torch.device(f"cuda:{rank}")

    # load data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # imagenet_ds = hdf5_dataset('../../datasets/imagenet_atom_noise_v4_rot_10m_100k_subset.h5', folder='imagenet', transform=transform)
    imagenet_ds = hdf5_dataset('../../datasets/imagenet_v5_rot_10m.h5', folder='train', transform=transform)
    train_ds, valid_ds = split_train_valid(imagenet_ds, 0.8)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
  
    # noise
    noise_ds = hdf5_dataset('../../datasets/imagenet_atom_noise_v4_rot_10m_100k_subset.h5', folder='noise', transform=transform)
    # _, noise_ds = split_train_valid(noise_ds, 0.01)
    noise_sampler = DistributedSampler(noise_ds, num_replicas=world_size, rank=rank)
    noise_dl = DataLoader(noise_ds, batch_size=batch_size, sampler=noise_sampler, num_workers=num_workers)
        
    # atom
    # atom_ds = hdf5_dataset('../../datasets/imagenet_atom_noise_v4_rot_10m_100k_subset.h5', folder='atom', transform=transform)
    atom_ds = hdf5_dataset('../../datasets/atom_v5_rot_200k.h5', folder='test', transform=transform)
    atom_sampler = DistributedSampler(atom_ds, num_replicas=world_size, rank=rank)
    atom_dl = DataLoader(atom_ds, batch_size=batch_size, sampler=atom_sampler, num_workers=num_workers)

    # define model and metrics
    NAME = '07302024-benchmark-XCiT-v4_10m-DDP'
    model = xcit_small(3, 17).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=num_epochs, max_lr=learning_rate, steps_per_epoch=len(train_dl))
    metrics = [accuracy]
    save_every = 1
    model_path = f'../../saved_models/{NAME}/'
    
    if rank == 0:
        config = {
            'dataset': '10 million datasets',
            'loss_func': 'CrossEntropyLoss', # nn.MSELoss()
            'optimizer': 'Adam',
            'scheduler': 'OneCycleLR',
        }
        wandb.login()
        proj_name = 'Understanding-Experimental-Images-by-Identifying-Symmetries-with-Deep-Learning'
        wandb.init(project=proj_name, entity='yig319', name=NAME, id=NAME, save_code=True, config=config)
        config = wandb.config

    # training scripts
    trainer = DDPTrainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        metrics=metrics,
        scheduler=scheduler,
        device=device,
        save_every=save_every,
        model_path=model_path,
        rank=rank,
        world_size=world_size,
        use_data_parallel=False
    )
    
    valid_dl_dict = {'valid': valid_dl, 'atom': atom_dl, 'noise': noise_dl}
    trainer.train(train_dl, num_epochs, valid_dl_dict=valid_dl_dict, tracking=tracking)
    trainer.cleanup()
    
    cleanup()

def main():
    mp.spawn(
        train_model,
        args=(WORLD_SIZE, GPU_IDS, NUM_WORKERS, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, TRACKING),
        nprocs=WORLD_SIZE,
        join=True
    )

# Global variables
GPU_IDS = [2, 3, 4, 5, 6, 7, 8]
WORLD_SIZE = len(GPU_IDS)
NUM_WORKERS = 4 # Adjust as needed
BATCH_SIZE = 180
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
TRACKING = True

if __name__ == "__main__":
    main()