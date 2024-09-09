import glob
import random
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import h5py
from matplotlib import pyplot as plt
import shutil
from viz import show_images
import fnmatch
import json # For dealing with metadata
from datafed.CommandLib import API

### packed visualization functions
def viz_dataloader(dl, n=8, title=None, hist_bins=None, show_colorbar=False, label_converter=None, stacked=False):
    batch = next(iter(dl))
    if len(batch[0]) < n: 
        raise ValueError("n is smaller than batch size, increase n")
 
    if stacked:
        for i in range(0, batch[0][:n].shape[1], 3):
            inputs = batch[0][:n][:, i:i+3]
            labels = list(batch[1][:n].numpy())
            if label_converter:
                for i in range(len(labels)):
                    labels[i] = label_converter[labels[i]]
            show_images(torch.permute(inputs, [0,2,3,1]).cpu().numpy(), labels=labels, title=title, hist_bins=hist_bins, show_colorbar=show_colorbar)

    else:
        inputs = batch[0][:n]
        labels = list(batch[1][:n].numpy())
        if label_converter:
            for i in range(len(labels)):
                labels[i] = label_converter[labels[i]]
        show_images(torch.permute(inputs, [0,2,3,1]).cpu().numpy(), labels=labels, title=title, hist_bins=hist_bins, show_colorbar=show_colorbar) 


def verify_image_vector(ax, image, ts, va, vb): 
    ts = np.array(ts)[1], np.array(ts)[0]
    va = np.array(va)[1], np.array(va)[0]
    vb = np.array(vb)[1], np.array(vb)[0]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        show = True
    else:
        show = False
        
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

    if show:
        plt.show()


def verify_image_in_hdf5_file_symmetry_operation(ds_path, n_list, group, viz=True):

    symmetry_dict = {'p1': 0, 'p2': 1, 'pm': 2, 'pg': 3, 'cm': 4, 'pmm': 5, 'pmg': 6, 'pgg': 7, 'cmm': 8, 
                     'p4': 9, 'p4m': 10, 'p4g': 11, 'p3': 12, 'p3m1': 13, 'p31m': 14, 'p6': 15, 'p6m': 16}
    symmetry_inv_dict = {v: k for k, v in symmetry_dict.items()}

    with h5py.File(ds_path, 'r') as h5:
        if viz:
            print('Total number of images in the dataset: ', len(h5[group]['data']))
        if isinstance(n_list, int):
           n_list = random.choices(range(len(h5[group]['data'])), k=n_list)
        n_list = np.sort(n_list)
        if viz:
            print('Randomly selected images: ', n_list)
        imgs = np.array(h5[group]['data'][n_list])
        labels = np.array(h5[group]['labels'][n_list])

    symmetry_dict = {'rotation_2f':0, 'rotation_3f':1, 'rotation_4f':2, 'mirror':3, 'glide':4}
    symmetry_inv_dict = {v: k for k, v in symmetry_dict.items()}
    labels = [symmetry_inv_dict[l] for l in labels]
    
    if viz:
        img_per_row = np.min((len(imgs), 4))
        show_images(imgs, labels=labels, img_per_row=img_per_row, img_height=4, show_colorbar=True)
    return imgs

def verify_image_in_hdf5_file(ds_path, n_list, group, viz=True):

    symmetry_dict = {'p1': 0, 'p2': 1, 'pm': 2, 'pg': 3, 'cm': 4, 'pmm': 5, 'pmg': 6, 'pgg': 7, 'cmm': 8, 
                        'p4': 9, 'p4m': 10, 'p4g': 11, 'p3': 12, 'p3m1': 13, 'p31m': 14, 'p6': 15, 'p6m': 16}

    symmetry_inv_dict = {v: k for k, v in symmetry_dict.items()}

    
    with h5py.File(ds_path, 'r') as h5:
        if viz:
            print('Total number of images in the dataset: ', len(h5[group]['data']))
        if isinstance(n_list, int):
           n_list = random.choices(range(len(h5[group]['data'])), k=n_list)
        n_list = np.sort(n_list)
        if viz:
            print('Randomly selected images: ', n_list)
        imgs = np.array(h5[group]['data'][n_list])
        # unit_cells = np.array(h5[group]['unit_cell'][n_list])
        labels = np.array(h5[group]['labels'][n_list])
        ts_list = np.array(h5[group]['translation_start_point'][n_list])
        va_list = np.array(h5[group]['primitive_uc_vector_a'][n_list])
        vb_list = np.array(h5[group]['primitive_uc_vector_b'][n_list])
        VA_list = np.array(h5[group]['translation_uc_vector_a'][n_list])
        VB_list = np.array(h5[group]['translation_uc_vector_b'][n_list])
    
    labels_str = [symmetry_inv_dict[l] for l in labels]
    metadata = {'ts': ts_list, 'va': va_list, 'vb': vb_list, 'VA': VA_list, 'VB': VB_list}
    if viz:
        for i in range(len(n_list)):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # verify_image_vector(axes[0], unit_cells[i], ts_list[i], va_list[i], vb_list[i])
            verify_image_vector(axes[0], imgs[i], ts_list[i], va_list[i], vb_list[i])
            verify_image_vector(axes[1], imgs[i], ts_list[i], VA_list[i], VB_list[i])
            plt.title(labels_str[i])
            plt.show()
        
    return imgs, labels_str, metadata

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



def numpy_to_tensor(image):
    # Ensure the image is in the correct format (H, W, C) and type
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 2:  # If it's a grayscale image
        image = np.expand_dims(image, axis=-1)
    # Convert to (C, H, W) format
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float() / 255.0


# ### h5py datasets for training
# class hdf5_dataset(Dataset):
    
#     def __init__(self, file_path, folder='train', transform=None, data_key='data', label_key='labels'):
#         self.file_path = file_path
#         self.folder = folder
#         self.transform = transform
#         self.hf = None
#         self.data_key = data_key
#         self.label_key = label_key

#     def __len__(self):
#         with h5py.File(self.file_path, 'r') as f:
#             self.len = len(f[self.folder]['labels'])
#         return self.len
    
#     # def __getitem__(self, idx):
#     #     if self.hf is None:
#     #         self.hf = h5py.File(self.file_path, 'r')
            
#     #     image = np.array(self.hf[self.folder][self.data_key][idx])
#     #     label = np.array(self.hf[self.folder][self.label_key][idx])
        
#     #     if self.transform:
#     #         image = self.transform(image)
#         # return image, label
    

#     def __getitem__(self, idx):
#         if self.hf is None:
#             self.hf = h5py.File(self.file_path, 'r')
            
#         image = np.array(self.hf[self.folder][self.data_key][idx])
#         label = np.array(self.hf[self.folder][self.label_key][idx])
        
#         # Convert numpy array to PIL Image
#         if image.dtype != np.uint8:
#             image = (image * 255).astype(np.uint8)
#         if image.ndim == 2:  # If it's a grayscale image
#             image = Image.fromarray(image, mode='L')
#         else:
#             image = Image.fromarray(image)
        
#         if self.transform:
#             image = self.transform(image)
        
#         return image, torch.tensor(label)


class hdf5_dataset(Dataset):
    
    def __init__(self, file_path, folder=None, transform=None, data_key='data', label_key='labels'):
        self.file_path = file_path
        self.folder = folder
        self.transform = transform
        self.hf = None
        self.data_key = data_key
        self.label_key = label_key

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            if self.folder:
                self.len = len(f[self.folder][self.label_key])
            else:
                self.len = len(f[self.label_key])
        return self.len
    
    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.file_path, 'r')
        
        if self.folder:
            image = np.array(self.hf[self.folder][self.data_key][idx])
            label = np.array(self.hf[self.folder][self.label_key][idx])
        else:
            image = np.array(self.hf[self.data_key][idx])
            label = np.array(self.hf[self.label_key][idx])
        
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if image.ndim == 2:  # If it's a grayscale image
            image = Image.fromarray(image, mode='L')
        else:
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)
    
class hdf5_dataset_stack(Dataset):
    
    def __init__(self, file_path, folder='train', transform=None, data_keys=['data'], label_key='labels'):
        self.file_path = file_path
        self.folder = folder
        self.transform = transform
        self.hf = None
        self.data_keys = data_keys
        self.label_key = label_key

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            self.len = len(f[self.folder]['labels'])
        return self.len
    
    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.file_path, 'r')

        images = [np.array(self.hf[self.folder][key][idx]) for key in self.data_keys]
        for i in range(len(images)):
            if len(images[i].shape)==2:
                images[i] = np.expand_dims(images[i], axis=-1)
        image = np.concatenate(images, axis=2)
        label = np.array(self.hf[self.folder][self.label_key][idx])
        
        if self.transform:
            image = self.transform(image)
        return image, label
    

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



### tool functions:

def split_train_valid(dataset, train_ratio, seed=42):
    all_size = len(dataset)
    train_size = int(train_ratio * all_size)
    valid_size = all_size - train_size
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, valid_size], 
                                                       generator=torch.Generator().manual_seed(seed))
    return train_ds, valid_ds



def detect_blank_images(h5_file, dataset_name, group_name=None):
    with h5py.File(h5_file, 'r') as f:
        if group_name:
            dataset = f[group_name][dataset_name]
        else:
            dataset = f[dataset_name]
        total_images = dataset.shape[0]
        blank_image_index = []

        for i in tqdm(range(total_images)):
            image = dataset[i]
            if np.min(image) == np.max(image):
                blank_image_index.append(i)
                # print(i)
                # plt.imshow(image)
                # plt.show()
                
    print(f"Blank images are: {blank_image_index}")
    return blank_image_index
        


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


def datafed_create_collection(collection_name, parent_id=None):
    df_api = API()
    coll_resp = df_api.collectionCreate(collection_name, parent_id=parent_id)
    return coll_resp

def visualize_datafed_collection(collection_id, max_count=100):
    df_api = API()
    item_list = []
    for item in list(df_api.collectionItemsList(collection_id, count=max_count)[0].item):
        print(item)
        item_list.append(item)
    return item_list

def datafed_upload(file_path, parent_id, metadata=None, wait=True):
    df_api = API()

    file_name = os.path.basename(file_path)
    dc_resp = df_api.dataCreate(file_name, metadata=json.dumps(metadata), parent_id=parent_id)
    rec_id = dc_resp[0].data[0].id
    put_resp = df_api.dataPut(rec_id, file_path, wait=wait)
    print(put_resp)
    
def datafed_download(file_path, file_id, wait=True):
    df_api = API()
    get_resp = df_api.dataGet([file_id], # currently only accepts a list of IDs / aliases
                              file_path, # directory where data should be downloaded
                              orig_fname=True, # do not name file by its original name
                              wait=wait, # Wait until Globus transfer completes
    )
    print(get_resp)

def datafed_update_record(record_id, metadata):
    df_api = API()
    du_resp = df_api.dataUpdate(record_id,
                                metadata=json.dumps(metadata),
                                metadata_set=True,
                                )
    print(du_resp)