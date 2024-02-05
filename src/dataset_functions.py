import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from matplotlib import pyplot as plt
import random
import shutil
from viz import show_images
import fnmatch

def parse_gitignore(gitignore_path):
    """
    Parses the .gitignore file and returns a list of patterns to ignore.

    Parameters:
    - gitignore_path (str): Path to the .gitignore file.

    Returns:
    - List[str]: A list of patterns to ignore.
    """
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as file:
            for line in file.readlines():
                # Strip whitespace and ignore comments
                pattern = line.strip()
                if pattern and not pattern.startswith('#'):
                    patterns.append(pattern)
    return patterns

def should_ignore(path, patterns):
    """
    Determines if a given path matches any of the ignore patterns.

    Parameters:
    - path (str): The path to check.
    - patterns (List[str]): The list of patterns to check against.

    Returns:
    - bool: True if the path should be ignored, False otherwise.
    """
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False

def copy_directory_structure(src, dest, gitignore_path=None):
    """
    Copies the directory structure (folders and subfolders) from source (src) to destination (dest),
    ignoring directories that match patterns in a .gitignore file.

    Parameters:
    - src (str): Source directory path.
    - dest (str): Destination directory path.
    - gitignore_path (str): Path to the .gitignore file. Optional.
    """
    ignore_patterns = []
    if gitignore_path:
        ignore_patterns = parse_gitignore(gitignore_path)
    
    # Ensure the source directory exists
    if not os.path.exists(src):
        print(f"Source directory {src} does not exist.")
        return
    
    # Ensure the destination directory exists; if not, create it
    os.makedirs(dest, exist_ok=True)
    
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d), ignore_patterns)]
        for dir_ in dirs:
            src_dir_path = os.path.join(root, dir_)
            dest_dir_path = src_dir_path.replace(src, dest, 1)
            os.makedirs(dest_dir_path, exist_ok=True)
            print(f"Directory created: {dest_dir_path}")



def verify_image_vector(ax, image, ts, va, vb): 
    ts = np.array(ts)[1], np.array(ts)[0]
    va = np.array(va)[1], np.array(va)[0]
    vb = np.array(vb)[1], np.array(vb)[0]

    ax.imshow(image)
    
    ax.set_ylabel('Y-axis')
    ax.set_xlabel('X-axis')
    
    array = np.array([[ts[0], ts[1], va[0], va[1]]])
    X, Y, U, V = zip(*array)
    ax.quiver(X, Y, U, V,color='b', angles='xy', scale_units='xy', scale=1, linewidth=0.3)
    
    array = np.array([[ts[0], ts[1], vb[0], vb[1]]])
    X, Y, U, V = zip(*array)
    ax.quiver(X, Y, U, V,color='b', angles='xy', scale_units='xy', scale=1, linewidth=0.3)

    plt.draw()


def verify_image_in_hdf5_file(ds_path, n_list, group, data_key='data', viz=True):

    symmetry_dict = {'p1': 0, 'p2': 1, 'pm': 2, 'pg': 3, 'cm': 4, 'pmm': 5, 'pmg': 6, 'pgg': 7, 'cmm': 8, 
                        'p4': 9, 'p4m': 10, 'p4g': 11, 'p3': 12, 'p3m1': 13, 'p31m': 14, 'p6': 15, 'p6m': 16}

    symmetry_inv_dict = {v: k for k, v in symmetry_dict.items()}

    with h5py.File(ds_path, 'r') as h5:
        if viz:
            print('Total number of images in the dataset: ', len(h5[group][data_key]))
        if isinstance(n_list, int):
           n_list = random.choices(range(len(h5[group][data_key])), k=n_list)
        n_list = np.sort(n_list)
        if viz:
            print('Randomly selected images: ', n_list)
        imgs = np.array(h5[group][data_key][n_list])
        unit_cells = np.array(h5[group]['unit_cell'][n_list])
        labels = np.array(h5[group]['labels'][n_list])
        ts_list = np.array(h5[group]['translation_start_point'][n_list])
        va_list = np.array(h5[group]['primitive_uc_vector_a'][n_list])
        vb_list = np.array(h5[group]['primitive_uc_vector_b'][n_list])
        VA_list = np.array(h5[group]['translation_uc_vector_a'][n_list])
        VB_list = np.array(h5[group]['translation_uc_vector_b'][n_list])

    if viz:
        for i in range(len(n_list)):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            verify_image_vector(axes[0], unit_cells[i], ts_list[i], va_list[i], vb_list[i])
            verify_image_vector(axes[1], imgs[i], ts_list[i], va_list[i], vb_list[i])
            verify_image_vector(axes[2], imgs[i], ts_list[i], VA_list[i], VB_list[i])
            plt.title(symmetry_inv_dict[labels[i]])
            plt.show()
    return imgs, unit_cells, labels, ts_list, va_list, vb_list, VA_list, VB_list


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

def viz_dataloader(dl, n=8, hist_bins=None, title=None, label_converter=None):
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