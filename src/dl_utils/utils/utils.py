import os
import re
import numpy as np
from tqdm import tqdm
import torch
import h5py
import fnmatch
import random

def find_symm_index_in_hdf5(h5, symm_str, group, index_start=0, index_end=None):
    symmetry_classes = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm', 'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']
    symm_dict = {i:symmetry_classes[i] for i in range(len(symmetry_classes))}

    if index_end is None:
        index_end = len(h5[group]['labels'])
    for i in range(index_start, index_end):
        if symm_str == symm_dict[h5[group]['labels'][i]]:
            return i
    return None

def fetch_img_metadata(h5, group, index, data_str='data', label_str='labels', ts_str='translation_start_point', va_str='primitive_uc_vector_a', vb_str='primitive_uc_vector_b', VA_str='translation_uc_vector_a', VB_str='translation_uc_vector_b'):
    
    symmetry_classes = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm', 'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']
    label_converter = list_to_dict(symmetry_classes)

    img, label, ts = h5[group][data_str][index], h5[group][label_str][index], h5[group][ts_str][index]
    va, vb = h5[group][va_str][index], h5[group][vb_str][index]
    VA, VB = h5[group][VA_str][index], h5[group][VB_str][index]
    label_str = label_converter[label]
    return img, label, label_str, ts, va, vb, VA, VB
    
    
def viz_h5_structure(h5_object, indent=''):
    """
    Print the structure of an HDF5 file.

    Parameters:
    - h5file: HDF5 file object or HDF5 group object.
    - indent: String used for indentation to represent the hierarchy level.
    """
    for key in h5_object.keys():
        item = h5_object[key]
        if isinstance(item, h5py.Group):
            print(f"{indent}'Group': {key}")
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}'Dataset': {key}; Shape: {item.shape}; dtype: {item.dtype}")
        if isinstance(item, h5py.Group):
            viz_h5_structure(item, indent + '  ')


def get_random_batch_indices(dataset, batch_size):
    """
    Generates a random batch of indices from an HDF5 dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample from.
        batch_size (int): Number of samples in the batch.

    Returns:
        list: A list of randomly selected indices.
    """
    dataset_size = len(dataset)
    return random.sample(range(dataset_size), batch_size)

def sort_tasks_by_size(file_paths, size_order):
    # Mapping from size label to sort index
    size_to_index = {label: i for i, label in enumerate(size_order)}

    def extract_size_label(file_path):
        # Ensure we match suffix like '-10k', '-5m' at the end of the filename
        match = re.search(r'-([\d]+[km])$', file_path)
        return match.group(1) if match else None

    # Sort using precomputed index from size_order
    return sorted(
        file_paths,
        key=lambda x: size_to_index.get(extract_size_label(x), float('inf'))
    )

    
def find_last_epoch_file(files):
    # Updated function to extract the epoch number from filenames like:
    # epoch_22-valid:loss=0.04,acc=0.99.pth
    def extract_epoch_number(file_path):
        match = re.search(r'epoch_(\d+)', file_path)
        if match:
            return int(match.group(1))
        return -1  # Return -1 if no epoch number found

    max_epoch_number = -1
    max_file_path = None

    for file_path in files:
        epoch_number = extract_epoch_number(file_path)
        if epoch_number > max_epoch_number:
            max_epoch_number = epoch_number
            max_file_path = file_path

    if not max_file_path:
        print("No valid epoch file found.")
        
    return max_file_path

def numpy_to_tensor(image):
    # Ensure the image is in the correct format (H, W, C) and type
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 2:  # If it's a grayscale image
        image = np.expand_dims(image, axis=-1)
    # Convert to (C, H, W) format
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float() / 255.0



### tool functions:

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


def copy_h5_data(source_file_path, target_file_path, batch_size=1024):
    with h5py.File(source_file_path, 'r') as source_file, h5py.File(target_file_path, 'w') as target_file:
        def copy_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Handle dataset
                if obj.dtype.kind == 'U':  # Check if the dataset contains Unicode strings
                    dtype = h5py.special_dtype(vlen=str)  # Use variable-length string dtype
                else:
                    dtype = obj.dtype

                target_dataset = target_file.create_dataset(name, shape=obj.shape, dtype=dtype, chunks=obj.chunks, compression=obj.compression)
                
                # Copy data in batches
                print(f'Copying dataset {name} with shape {obj.shape} and dtype {dtype}')
                for i in tqdm(range(0, obj.shape[0], batch_size)):
                    end_index = min(i + batch_size, obj.shape[0])
                    target_dataset[i:end_index] = obj[i:end_index]

            elif isinstance(obj, h5py.Group):
                # Handle group
                target_group = target_file.create_group(name)
                # Copy attributes
                for attr_name, attr_value in obj.attrs.items():
                    target_group.attrs[attr_name] = attr_value

        # Walk through all items in the source file and copy them
        source_file.visititems(copy_item)


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def NormalizeData_batch(data):
    min_vals = torch.min(data, dim=(1, 2, 3), keepdim=True).values
    max_vals = torch.max(data, dim=(1, 2, 3), keepdim=True).values
    return (data - min_vals) / (max_vals - min_vals)


def list_to_dict(lst, inverse=False):
    dictionary = {}
    if inverse:
        for index, item in enumerate(lst):
            dictionary[item] = index
    else:
        for index, item in enumerate(lst):
            dictionary[index] = item
    return dictionary


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
