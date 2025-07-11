import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import wandb
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from m3util.viz.layout import layout_fig

from dl_utils.analysis.confusion_matrix import show_multiple_cm, confusion_matrix
from dl_utils.utils.utils import list_to_dict, sort_tasks_by_size, find_last_epoch_file
from dl_utils.utils.dataset import viz_dataloader, split_train_valid, hdf5_dataset
from dl_utils.training.build_model import resnet50_, xcit_small, fpn_resnet50_classification, densenet161_
from dl_utils.training.trainer import Trainer, accuracy


def viz_4confusion_matrix(cm_files, title, symmetry_classes, filename=None):
    
    cm_list = []
    subtitles = []
    dataset_list = ['imagenet', 'imagenet', 'atom', 'noise']
    task_list = ['train', 'valid', 'cross_validation', 'cross_validation']
    for dataset, task in zip(dataset_list, task_list):

        matching_paths = [path for path in cm_files if f'{dataset}_{task}' in path]
        if matching_paths == 0 or len(matching_paths) > 1:
            raise ValueError(f'Invalid number of matching paths found: {matching_paths}')
        else:
            path = matching_paths[0]
        cm = np.load(path)
        cm_list.append(cm)
        subtitles.append(f'{dataset}-{task}')

    fig, axes = layout_fig(len(cm_list), 2, figsize=(10, 8))
    show_multiple_cm(cm_list, subtitles, fig=fig, axes=axes, suptitle=title, classes=symmetry_classes, font_size=4)

    if filename:
        plt.savefig(f'{filename}.png', dpi=600)
        plt.savefig(f'{filename}.svg', dpi=600)
    plt.show()
    
    return cm_list


def generate_confusion_matrix_batch_full_val_set_add_noise(model, model_path_list, ds_size_list, ds_path_info, running_specs, noise_std=0.1):
    
    '''
    Example inputs:
    
    task_orders = ['1k', '10k', '100k', '500k', '1m', '2m', '5m', '10m']
    ds_size_list = [1000, 10000, 100000, 500000, 1000000, 2000000, 5000000, 10000000]
    dir_path_list = sort_tasks_by_size(glob.glob('../../../models/ResNet50/*'), task_orders)
    ds_path_info = {'imagenet': '../../../datasets/imagenet_v5_rot_10m_fix_vector.h5',
                'noise': '../../../datasets/noise_v5_rot_1m_fix_vector.h5',
                'atom': '../../../datasets/atom_v5_rot_1m_fix_vector.h5'
                }
    running_specs = {'batch_size': 2800, 
                    'num_workers': 12, 
                    'save_path': '../../../results/ResNet50/',
                    'device_ids': [0],
    }
    
    '''
     # model 
    device = torch.device('cuda:{}'.format(running_specs['device_ids'][0]))

    stats = {}
    print(f'Running on {device}')
    for dir_path, ds_size in zip(model_path_list, ds_size_list):
        
        stats_running = {}
        
        task = os.path.basename(dir_path)
        file = find_last_epoch_file(glob.glob(f'{dir_path}/*'))
        # model = xcit_small(in_channels=3, n_classes=17)
        print(f'Loading model from {file}')
        model.load_state_dict(torch.load(file, weights_only=True))
        if len(running_specs['device_ids']) > 1:
            model = torch.nn.DataParallel(model, device_ids=running_specs['device_ids'])
        else:
            model = model.to(device)
            
        model.eval()

        # symmetry classes
        symmetry_classes = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm', 'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']
        label_converter = list_to_dict(symmetry_classes)


        import torchvision.transforms as T
        class AddGaussianNoise(object):
            def __init__(self, mean=0.0, std=0.1, clamp=True):
                self.mean = mean
                self.std = std
                self.clamp = clamp

            def __call__(self, tensor):
                noisy = tensor + torch.randn_like(tensor) * self.std + self.mean
                if self.clamp:
                    noisy = torch.clamp(noisy, 0.0, 1.0)
                return noisy

            def __repr__(self):
                return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, clamp={self.clamp})"

        transform = T.Compose([
            T.ToTensor(),
            AddGaussianNoise(mean=0.0, std=noise_std),
        ])
        symmetry_classes = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm', 'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']
        label_converter = list_to_dict(symmetry_classes)
        
        # noise
        noise_ds = hdf5_dataset(ds_path_info['noise'], folder='noise', transform=transform)
        noise_dl = DataLoader(noise_ds, batch_size=running_specs['batch_size'], shuffle=False, num_workers=running_specs['num_workers'])
        viz_dataloader(noise_dl, label_converter=label_converter, title='noise_dl')


        # atom
        atom_ds = hdf5_dataset(ds_path_info['atom'], folder='atom', transform=transform)
        atom_dl = DataLoader(atom_ds, batch_size=running_specs['batch_size'], shuffle=False, num_workers=running_specs['num_workers'])
        viz_dataloader(atom_dl, label_converter=label_converter, title='atom_dl')
        
        print(f'Confusion Matrix for {task}:')
        
        cm, accuracy = confusion_matrix(model, atom_dl, symmetry_classes, device, n_batches='all')
        np.save(f'../../../results/XCiT/{task}-atom_cross_validation_cm_full_val_add_noise_{noise_std}.npy', cm)
        stats_running[f'{task}-atom_cv'] = accuracy

        cm, accuracy = confusion_matrix(model, noise_dl, symmetry_classes, device, n_batches='all')
        np.save(f'../../../results/XCiT/{task}-noise_cross_validation_cm_full_val_add_noise_{noise_std}.npy', cm)
        stats_running[f'{task}-noise_cv'] = accuracy
        
        stats[f'{task}'] = stats_running
    return stats


def generate_confusion_matrix_batch_full_val_set(model, model_path_list, ds_size_list, ds_path_info, running_specs, subset_size=None):
    
    '''
    Example inputs:
    
    task_orders = ['1k', '10k', '100k', '500k', '1m', '2m', '5m', '10m']
    ds_size_list = [1000, 10000, 100000, 500000, 1000000, 2000000, 5000000, 10000000]
    dir_path_list = sort_tasks_by_size(glob.glob('../../../models/ResNet50/*'), task_orders)
    ds_path_info = {'imagenet': '../../../datasets/imagenet_v5_rot_10m_fix_vector.h5',
                'noise': '../../../datasets/noise_v5_rot_1m_fix_vector.h5',
                'atom': '../../../datasets/atom_v5_rot_1m_fix_vector.h5'
                }
    running_specs = {'batch_size': 2800, 
                    'num_workers': 12, 
                    'save_path': '../../../results/ResNet50/',
                    'device_ids': [0],
    }
    
    '''
     # model 
    device = torch.device('cuda:{}'.format(running_specs['device_ids'][0]))

    stats = {}
    print(f'Running on {device}')
    for dir_path, ds_size in zip(model_path_list, ds_size_list):
        
        stats_running = {}
        
        task = os.path.basename(dir_path)
        file = find_last_epoch_file(glob.glob(f'{dir_path}/*'))
        # model = xcit_small(in_channels=3, n_classes=17)
        print(f'Loading model from {file}')
        model.load_state_dict(torch.load(file, weights_only=True))
        if len(running_specs['device_ids']) > 1:
            model = torch.nn.DataParallel(model, device_ids=running_specs['device_ids'])
        else:
            model = model.to(device)
            
        model.eval()

        # symmetry classes
        symmetry_classes = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm', 'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']
        label_converter = list_to_dict(symmetry_classes)

        # imagenet
        # imagenet_ds = hdf5_dataset(ds_path_info['imagenet'], folder='imagenet', transform=transforms.ToTensor())
        # ratio = ds_size * (1/0.8) / len(imagenet_ds)
        # imagenet_ds, _ = split_train_valid(imagenet_ds, ratio, seed=42)
        # train_ds, valid_ds = split_train_valid(imagenet_ds, 0.8, seed=42) 
        # train_dl = DataLoader(train_ds, batch_size=running_specs['batch_size'], shuffle=True, num_workers=running_specs['num_workers'])
        # valid_dl = DataLoader(valid_ds, batch_size=running_specs['batch_size'], shuffle=False, num_workers=running_specs['num_workers'])

        # noise
        noise_ds = hdf5_dataset(ds_path_info['noise'], folder='noise', transform=transforms.ToTensor())
        if subset_size is not None:
            ratio = np.min((subset_size / len(noise_ds), 1))
            noise_ds, rest_ds = split_train_valid(noise_ds, ratio, seed=42)
        noise_dl = DataLoader(noise_ds, batch_size=running_specs['batch_size'], shuffle=False, num_workers=running_specs['num_workers'])
        
        # atom
        atom_ds = hdf5_dataset(ds_path_info['atom'], folder='atom', transform=transforms.ToTensor())
        if subset_size is not None:
            ratio = np.min((subset_size / len(atom_ds), 1))
            atom_ds, rest_ds = split_train_valid(atom_ds, ratio, seed=42)
        atom_dl = DataLoader(atom_ds, batch_size=running_specs['batch_size'], shuffle=False, num_workers=running_specs['num_workers'])
        
        print(f'Confusion Matrix for {task}:')
        
        # cm, accuracy = confusion_matrix(model, train_dl, symmetry_classes, device, n_batches='all')
        # np.save(f'../../../results/XCiT/{task}-imagenet_train_cm.npy', cm)
        # # plot_cm(cm, symmetry_classes, title='ResNet50-ImageNet-Train', cm_style='simple', fig_style='printing', font_size=4)
        # stats_running[f'{task}-train'] = accuracy

        # cm, accuracy = confusion_matrix(model, valid_dl, symmetry_classes, device, n_batches='all')
        # np.save(f'../../../results/XCiT/{task}-imagenet_valid_cm.npy', cm)
        # # plot_cm(cm, symmetry_classes, title='ResNet50-ImageNet-Train', cm_style='simple', fig_style='printing', font_size=4)
        # stats_running[f'{task}-valid'] = accuracy

        cm, accuracy = confusion_matrix(model, atom_dl, symmetry_classes, device, n_batches='all')
        np.save(f'../../../results/XCiT/{task}-atom_cross_validation_cm_full_val.npy', cm)
        # plot_cm(cm, symmetry_classes, title='ResNet50-Atom-Cross_Validation', cm_style='simple', fig_style='printing', font_size=4)
        stats_running[f'{task}-atom_cv'] = accuracy

        cm, accuracy = confusion_matrix(model, noise_dl, symmetry_classes, device, n_batches='all')
        np.save(f'../../../results/XCiT/{task}-noise_cross_validation_cm_full_val.npy', cm)
        # plot_cm(cm, symmetry_classes, title='ResNet50-Atom-Cross_Validation', cm_style='simple', fig_style='printing', font_size=4)
        stats_running[f'{task}-noise_cv'] = accuracy
        
        stats[f'{task}'] = stats_running
    return stats


def generate_confusion_matrix_batch(model, model_path_list, ds_size_list, ds_path_info, running_specs):
    
    '''
    Example inputs:
    
    task_orders = ['1k', '10k', '100k', '500k', '1m', '2m', '5m', '10m']
    ds_size_list = [1000, 10000, 100000, 500000, 1000000, 2000000, 5000000, 10000000]
    dir_path_list = sort_tasks_by_size(glob.glob('../../../models/ResNet50/*'), task_orders)
    ds_path_info = {'imagenet': '../../../datasets/imagenet_v5_rot_10m_fix_vector.h5',
                'noise': '../../../datasets/noise_v5_rot_1m_fix_vector.h5',
                'atom': '../../../datasets/atom_v5_rot_1m_fix_vector.h5'
                }
    running_specs = {'batch_size': 2800, 
                    'num_workers': 12, 
                    'save_path': '../../../results/ResNet50/',
                    'device_ids': [0],
    }
    
    '''
     # model 
    device = torch.device('cuda:{}'.format(running_specs['device_ids'][0]))

    stats = {}
    print(f'Running on {device}')
    for dir_path, ds_size in zip(model_path_list, ds_size_list):
        
        stats_running = {}
        
        task = os.path.basename(dir_path)
        file = find_last_epoch_file(glob.glob(f'{dir_path}/*'))
        # model = xcit_small(in_channels=3, n_classes=17)
        print(f'Loading model from {file}')
        model.load_state_dict(torch.load(file, weights_only=True))
        if len(running_specs['device_ids']) > 1:
            model = torch.nn.DataParallel(model, device_ids=running_specs['device_ids'])
        else:
            model = model.to(device)
            
        model.eval()

        # symmetry classes
        symmetry_classes = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm', 'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']
        label_converter = list_to_dict(symmetry_classes)

        # imagenet
        imagenet_ds = hdf5_dataset(ds_path_info['imagenet'], folder='imagenet', transform=transforms.ToTensor())
        ratio = ds_size * (1/0.8) / len(imagenet_ds)
        imagenet_ds, _ = split_train_valid(imagenet_ds, ratio, seed=42)
        train_ds, valid_ds = split_train_valid(imagenet_ds, 0.8, seed=42) 
        train_dl = DataLoader(train_ds, batch_size=running_specs['batch_size'], shuffle=True, num_workers=running_specs['num_workers'])
        valid_dl = DataLoader(valid_ds, batch_size=running_specs['batch_size'], shuffle=False, num_workers=running_specs['num_workers'])

        # noise
        noise_ds = hdf5_dataset(ds_path_info['noise'], folder='noise', transform=transforms.ToTensor())
        ratio = np.min((ds_size / len(noise_ds), 1)) # avoid larger than 1
        noise_ds, rest_ds = split_train_valid(noise_ds, ratio, seed=42)
        noise_dl = DataLoader(noise_ds, batch_size=running_specs['batch_size'], shuffle=False, num_workers=running_specs['num_workers'])
        
        # atom
        atom_ds = hdf5_dataset(ds_path_info['atom'], folder='atom', transform=transforms.ToTensor())
        ratio = np.min((ds_size / len(atom_ds), 1)) # avoid larger than 1
        atom_ds, rest_ds = split_train_valid(atom_ds, ratio, seed=42)
        atom_dl = DataLoader(atom_ds, batch_size=running_specs['batch_size'], shuffle=False, num_workers=running_specs['num_workers'])
        
        print(f'Confusion Matrix for {task}:')
        
        cm, accuracy = confusion_matrix(model, train_dl, symmetry_classes, device, n_batches='all')
        np.save(f'../../../results/XCiT/{task}-imagenet_train_cm.npy', cm)
        # plot_cm(cm, symmetry_classes, title='ResNet50-ImageNet-Train', cm_style='simple', fig_style='printing', font_size=4)
        stats_running[f'{task}-train'] = accuracy

        cm, accuracy = confusion_matrix(model, valid_dl, symmetry_classes, device, n_batches='all')
        np.save(f'../../../results/XCiT/{task}-imagenet_valid_cm.npy', cm)
        # plot_cm(cm, symmetry_classes, title='ResNet50-ImageNet-Train', cm_style='simple', fig_style='printing', font_size=4)
        stats_running[f'{task}-valid'] = accuracy

        cm, accuracy = confusion_matrix(model, atom_dl, symmetry_classes, device, n_batches='all')
        np.save(f'../../../results/XCiT/{task}-atom_cross_validation_cm.npy', cm)
        # plot_cm(cm, symmetry_classes, title='ResNet50-Atom-Cross_Validation', cm_style='simple', fig_style='printing', font_size=4)
        stats_running[f'{task}-atom_cv'] = accuracy

        cm, accuracy = confusion_matrix(model, noise_dl, symmetry_classes, device, n_batches='all')
        np.save(f'../../../results/XCiT/{task}-noise_cross_validation_cm.npy', cm)
        # plot_cm(cm, symmetry_classes, title='ResNet50-Atom-Cross_Validation', cm_style='simple', fig_style='printing', font_size=4)
        stats_running[f'{task}-noise_cv'] = accuracy
        
        stats[f'{task}'] = stats_running
    return stats
    


def benchmark_task(task_name, model, training_specs, ds_path_info, wandb_specs={}):

    # symmetry classes
    symmetry_classes = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm', 'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']
    label_converter = list_to_dict(symmetry_classes)

    # imagenet
    imagenet_ds = hdf5_dataset(ds_path_info['imagenet'], folder='imagenet', transform=transforms.ToTensor())
    ratio = training_specs['ds_size'] * (1/0.8) / len(imagenet_ds)
    imagenet_ds, _ = split_train_valid(imagenet_ds, ratio, seed=42)
    train_ds, valid_ds = split_train_valid(imagenet_ds, 0.8, seed=42) 
    train_dl = DataLoader(train_ds, batch_size=training_specs['batch_size'], shuffle=True, num_workers=training_specs['num_workers'])
    valid_dl = DataLoader(valid_ds, batch_size=training_specs['batch_size'], shuffle=False, num_workers=training_specs['num_workers'])

    # noise
    noise_ds = hdf5_dataset(ds_path_info['noise'], folder='noise', transform=transforms.ToTensor())
    ratio = np.min((training_specs['ds_size'] / len(noise_ds), 1)) # avoid larger than 1
    noise_ds, rest_ds = split_train_valid(noise_ds, ratio, seed=42)
    noise_dl = DataLoader(noise_ds, batch_size=training_specs['batch_size'], shuffle=False, num_workers=training_specs['num_workers'])
    
    # atom
    atom_ds = hdf5_dataset(ds_path_info['atom'], folder='atom', transform=transforms.ToTensor())
    ratio = np.min((training_specs['ds_size'] / len(atom_ds), 1)) # avoid larger than 1
    atom_ds, rest_ds = split_train_valid(atom_ds, ratio, seed=42)
    atom_dl = DataLoader(atom_ds, batch_size=training_specs['batch_size'], shuffle=False, num_workers=training_specs['num_workers'])
    
    # visualization
    if ds_path_info['viz_dataloader']:
        viz_dataloader(train_dl, label_converter=label_converter, title='imagenet - train')
        viz_dataloader(valid_dl, label_converter=label_converter, title='imagenet - valid')
        viz_dataloader(noise_dl, label_converter=label_converter, title='noise - cv')
        viz_dataloader(atom_dl, label_converter=label_converter, title='atom - cv')

    # model 
    device = torch.device('cuda:{}'.format(training_specs['device_ids'][0]))
    if len(training_specs['device_ids']) > 1:
        model = torch.nn.DataParallel(model, device_ids=training_specs['device_ids'])
    else:
        model = model.to(device)
        
    # wandb
    NAME = task_name + '-dstaset_size=' + str(training_specs['ds_size'])
    if wandb_specs != {}:
        wandb.login()
        wandb.init(project=wandb_specs['project'], entity=wandb_specs['entity'], name=NAME, id=NAME, group=wandb_specs['group'], save_code=wandb_specs['save_code'], config=wandb_specs['config'], resume=wandb_specs['resume'])
        training_specs['wandb_record'] = True
    else:
        training_specs['wandb_record'] = False
        
    # training
    lr = training_specs['learning_rate']
    if 'epoch_start' in training_specs:
        epoch_start = training_specs['epoch_start']
    else:
        epoch_start = 0
    epochs = training_specs['training_image_count'] // len(train_ds) - epoch_start # training epochs based on the number of images in the dataset 
    valid_per_epochs = int(np.max((1, epochs / training_specs['validation_times']))) # validation times based on the number of epochs, and at least 1
    early_stopping_patience = np.max((5, valid_per_epochs+2)) # early stopping patience based on the number of validation times, and at least 2
    efficient_print = training_specs['efficient_print']
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=epochs, max_lr=lr, steps_per_epoch=len(train_dl))
    metrics = [accuracy]  # You can add more metrics if needed
    if training_specs['folder_name'] == 'default':
        folder_name = NAME
    else:
        folder_name = training_specs['folder_name']
    
    trainer = Trainer(model=model, loss_func=loss_func, optimizer=optimizer, metrics=metrics, scheduler=scheduler, device=device, save_per_epochs=valid_per_epochs, model_path=training_specs['model_path']+folder_name+'/', early_stopping_patience=early_stopping_patience, efficient_print=efficient_print) # 

    history = trainer.train(train_dl=train_dl, epochs=epochs, epoch_start=epoch_start, valid_per_epochs=valid_per_epochs, valid_dl_list=[valid_dl, noise_dl, atom_dl], valid_dl_names=['', 'noise', 'atom'], wandb_record=training_specs['wandb_record'])
    wandb.finish()
    
    print(history)
    trainer.plot_training_metrics()
    
    return model, history
    