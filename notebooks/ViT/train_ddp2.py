import os
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
from train_functions import train_epochs
from utils import split_train_valid, list_to_dict, viz_dataloader, hdf5_dataset

# Global variables
GPU_IDS = [6, 7, 8]
WORLD_SIZE = len(GPU_IDS)
NUM_WORKERS = 6  # Adjust as needed
BATCH_SIZE = 300
LEARNING_RATE = 1e-3
START_EPOCH = 0
NUM_EPOCHS = 1

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, GPU_IDS))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    
    model = xcit_small(3, 17).to(device)
    model = DDP(model, device_ids=[rank])
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    imagenet_ds = hdf5_dataset('../../datasets/imagenet_v5_rot_10m.h5', folder='train', transform=transform)
    train_ds, valid_ds = split_train_valid(imagenet_ds, 0.8)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=NUM_WORKERS)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=NUM_EPOCHS, max_lr=LEARNING_RATE, steps_per_epoch=len(train_dl))
    
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_dl):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0 and rank == 0:
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_dl)}, Loss: {loss.item():.4f}')
        
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in valid_dl:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_loss += loss_func(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        if rank == 0:
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss/len(train_dl):.4f}, '
                  f'Valid Loss: {valid_loss/len(valid_dl):.4f}, '
                  f'Valid Accuracy: {100.*correct/total:.2f}%')
    
    cleanup()

def main():
    mp.spawn(train_model, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

if __name__ == "__main__":
    main()