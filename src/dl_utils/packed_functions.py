import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import wandb
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from m3util.viz.layout import layout_fig

from dl_utils.analysis.case_analysis import generate_prediction_example
from dl_utils.analysis.confusion_matrix import show_multiple_cm
from dl_utils.analysis.attention_map import AttentionMapVisualizer
from dl_utils.utils.utils import list_to_dict, sort_tasks_by_size, viz_h5_structure, find_symm_index_in_hdf5, fetch_img_metadata
from dl_utils.utils.dataset import viz_dataloader, split_train_valid, hdf5_dataset
from dl_utils.training.trainer import Trainer, accuracy



def generate_attention_maps(model, ds_path, confusion_pair, layers, task_name, model_type, device, filename=None, viz=True):
    
    ds_path = f'../../../Wallpaper_Group_Symmetry_Dataset/datasets/imagenet_v5/imagenet_v5_rot_10k_fix_vector_a100_0.h5'
    group = 'imagenet'
    with h5py.File(ds_path) as h5:
        index = find_symm_index_in_hdf5(h5, symm_str='p4', group=group, index_start=1000, index_end=None)
        input_tensor, label, label_str, ts, va, vb, VA, VB = fetch_img_metadata(h5, group=group, index=index)
        input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)
        label_pred = model(input_tensor).argmax().item()
        label_pred_str = symmetry_classes[label_pred]
        if label_pred_str != 'p4':
            raise ValueError(f'Invalid prediction: {label_pred_str}')
    
    attention_map_resized_list, overlay_attention_map_list = [], []
    visualizer = AttentionMapVisualizer(device=device)

    layers = ['layer4', 'layer3', 'layer2', 'layer1']
    for i, layer in enumerate(layers):
        if model_type == 'ResNet50':
            input_image_np, attention_map_resized = visualizer.generate_cnn_attention_map(model, input_tensor, layer_name=layer)
        elif model_type == 'XCiT':
            input_image_np, attention_map_resized = visualizer.generate_transformer_attention_map(model, input_tensor, attention_layer_idx=layer)
        attention_map_resized_list.append(attention_map_resized)
        overlay_attention_map_list.append(overlay_attention_map)
            
    if viz:
        fig, axes = layout_fig(len(layers)*3, 3, figsize=(8, len(layers)*2.8), subplot_style='subplots', layout='tight')

        for i, layer in enumerate(layers):
            overlay_attention_map = visualizer.visualize_attention_map(input_image_np, attention_map_resized, keyword=layer, fig=fig, axes=axes[i*3:i*3+3])
            
        plt.suptitle(f"{task_name}", fontsize=10)
        if filename:
            plt.savefig(f'{filename}.png', dpi=600)
            plt.savefig(f'{filename}.svg', dpi=600)
        
        plt.show()
        
    return input_image_np, overlay_attention_map_list, overlay_attention_map_list


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
    epoch_start = 0
    epochs = training_specs['training_image_count'] // len(train_ds) # training epochs based on the number of images in the dataset 
    valid_per_epochs = epochs / training_specs['validation_times'] # validation times based on the number of epochs
    efficient_print = training_specs['efficient_print']
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=epochs, max_lr=lr, steps_per_epoch=len(train_dl))
    metrics = [accuracy]  # You can add more metrics if needed
    trainer = Trainer(model=model, loss_func=loss_func, optimizer=optimizer, metrics=metrics, scheduler=scheduler, device=device, save_per_epochs=valid_per_epochs, model_path=training_specs['model_path']+NAME+'/', early_stopping_patience=valid_per_epochs, efficient_print=efficient_print)

    history = trainer.train(train_dl=train_dl, epochs=epochs, epoch_start=epoch_start, valid_per_epochs=valid_per_epochs, valid_dl_list=[valid_dl, noise_dl, atom_dl], valid_dl_names=['', 'noise', 'atom'], wandb_record=training_specs['wandb_record'])
    wandb.finish()
    
    trainer.plot_training_metrics()
    
    return model, history
    