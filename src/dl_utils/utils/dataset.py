# import modules
import h5py
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
from m3util.viz.layout import layout_fig


### packed visualization functions
def viz_dataloader(dl, n=8, title=None, hist_bins=None, show_colorbar=False, label_converter=None, stacked=False):
    batch = next(iter(dl))
    if len(batch[0]) < n: 
        raise ValueError("n is smaller than batch size, increase n")
    
    if stacked:
        for i in range(0, batch[0][:n].shape[1], 3):
            inputs = batch[0][:n][:, i:i+3]
            labels = list(batch[1][:n].numpy())
            imgs = torch.permute(inputs, [0,2,3,1]).cpu().numpy()
            
            if label_converter:
                for i in range(len(labels)):
                    labels[i] = label_converter[labels[i]]
            fig, axes = layout_fig(n, 8, figsize=(8, 2))
            for i in range(n):
                axes[i].imshow(imgs[i])
                axes[i].set_title(labels[i])
                axes[i].axis('off')
            plt.show()
    else:
        inputs = batch[0][:n]
        labels = list(batch[1][:n].numpy())
        imgs = torch.permute(inputs, [0,2,3,1]).cpu().numpy()

        if label_converter:
            for i in range(len(labels)):
                labels[i] = label_converter[labels[i]]
                
        fig, axes = layout_fig(n, 8, figsize=(8, 2))
        for i in range(n):
            axes[i].imshow(imgs[i])
            axes[i].set_title(labels[i])
            axes[i].axis('off')
        plt.show()


# def split_train_valid(dataset, train_ratio, seed=42):
#     all_size = len(dataset)
#     train_size = int(train_ratio * all_size)
#     valid_size = all_size - train_size
#     train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(seed))
#     return train_ds, valid_ds

class SubsetWithAttributes(torch.utils.data.Subset):
    """A custom Subset class that inherits attributes and methods from the original dataset."""
    
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.__dict__.update(dataset.__dict__)  # Inherit attributes
    
    def list_metrics(self):
        """List available metric keys in the HDF5 dataset."""
        return self.dataset.list_metrics()
    
    def get_metric(self, idx, metric_name):
        """Retrieve a specific metric for a given index, adjusted for subset indices."""
        original_idx = self.indices[idx]  # Map subset index to original dataset index
        return self.dataset.get_metric(original_idx, metric_name)


def split_train_valid(dataset, train_ratio=0.8, seed=42):
    """
    Splits a dataset into training and validation subsets while inheriting all properties.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset.
        train_ratio (float): Fraction of data to use for training.
        seed (int): Random seed for reproducibility.

    Returns:
        train_ds, valid_ds (SubsetWithAttributes): Subsets with inherited attributes.
    """
    all_size = len(dataset)
    train_size = int(train_ratio * all_size)
    valid_size = all_size - train_size

    train_indices, valid_indices = random_split(
        range(all_size), [train_size, valid_size], generator=torch.Generator().manual_seed(seed)
    )

    # Use the custom Subset class to retain all dataset attributes and methods
    train_ds = SubsetWithAttributes(dataset, train_indices)
    valid_ds = SubsetWithAttributes(dataset, valid_indices)

    return train_ds, valid_ds


class hdf5_dataset(Dataset):
    
    def __init__(self, file_path, folder=None, transform=None, data_key='data', label_key='labels', classes=[]):
        self.file_path = file_path
        self.folder = folder
        self.transform = transform
        self.hf = None
        self.data_key = data_key
        self.label_key = label_key
        self.classes = classes

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
    
    def list_metrics(self):
        """List all available metric keys in the HDF5 dataset."""
        with h5py.File(self.file_path, 'r') as f:
            if self.folder:
                keys = list(f[self.folder].keys())  # Get all keys inside the folder
            else:
                keys = list(f.keys())  # Get all keys at root level

        # Exclude known data and label keys
        metrics = [key for key in keys if key not in {self.data_key, self.label_key}]
        return metrics

    def get_metric(self, idx, metric_name):
        """Retrieve a specific metric for a given index."""
        if self.hf is None:
            self.hf = h5py.File(self.file_path, 'r')
        
        if self.folder:
            if metric_name in self.hf[self.folder]:
                metric_value = self.hf[self.folder][metric_name][idx]
            else:
                raise KeyError(f"Metric '{metric_name}' not found in folder '{self.folder}'.")
        else:
            if metric_name in self.hf:
                metric_value = self.hf[metric_name][idx]
            else:
                raise KeyError(f"Metric '{metric_name}' not found in the dataset.")
        
        return metric_value
    
    
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