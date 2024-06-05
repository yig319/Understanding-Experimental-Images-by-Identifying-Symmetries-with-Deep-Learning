import os
import shutil
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import radon, rescale
import sys
sys.path.append('../utils/')
from viz import show_images
from NormalizeData import NormalizeData
    
def log_scale(fft_real):
    fft_real_log = np.log(np.clip(fft_real, 1e-5,None))
    fft_real_log[fft_real_log == float('-inf')] = 0
    return NormalizeData(fft_real_log)

def fft_transform(image, return_type='numpy'):
    ms_list, ps_list = [], []
    for channel in range(image.shape[2]):  # Assuming the image has three channels: R, G, B
        img_fft = np.fft.fft2(image[:,:,channel])
        img_fft_shift = np.fft.fftshift(img_fft)
        
        ms = NormalizeData(np.log1p(np.abs(img_fft_shift)))
        ms_list.append(ms)
        ps = NormalizeData(np.angle(img_fft_shift))
        ps_list.append(np.angle(img_fft_shift))
        
    if return_type == 'numpy':
        ms = np.stack(ms_list, axis=-1)
        ps = np.stack(ps_list, axis=-1)
        ms = np.nan_to_num(ms, copy=False)
        ps = np.nan_to_num(ps, copy=False)
        return NormalizeData(ms), NormalizeData(ps)
        # return ms, ps
    elif return_type == 'list':
        for channel in range(image.shape[2]): 
            ms_list[channel] = NormalizeData(ms_list[channel])
            ps_list[channel] = NormalizeData(ps_list[channel])
        return ms_list, ps_list
        
    # image = torch.tensor(image)
    # out = torch.clone(image)
    # img_fft = torch.fft.fft2(image, dim=(0,1))
    # img_shift = torch.fft.fftshift(img_fft)
    # out = np.log(np.abs(img_shift))
    # out[out==-np.inf] = 0
    # out = scale(out)
    # return out

def radon_transform(image):
    
    image_gray = rgb2gray(image)
    # Define the range of angles
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)

    # Perform the Radon transform
    sinogram = radon(image_gray, theta=theta, circle=True)
    return (sinogram/255).astype(np.float32)

    # img = np.copy(image)
    # if not isinstance(image, np.ndarray):
    #     img = img.numpy()
    # if len(img.shape) == 3:
    #     if img.shape[0] == 3: # shape: 3,256,256 
    #         img = np.swapaxes(img, 0, 2)
    #         img = np.swapaxes(img, 0, 1)
    #     img = color.rgb2gray(img)
   
    # theta = np.linspace(0., 180., len(image), endpoint=False)
    # sinogram = radon(img, theta=theta, circle=True)
    # return scale(sinogram)
#     return np.expand_dims(sinogram, axis=0)

def fft_radon_examples(img, title, axes=None, save_path = None):

    labels = ['raw image', 'fft_magnitude-0', 'fft_magnitude-1', 'fft_magnitude-2', 'fft_phase-0', 'fft_phase-1', 'fft_phase-2', 'radon']

    fft_magnitude, fft_phase = fft_transform(img, return_type='numpy')
    radon = radon_transform(img)
    imgs_show = [img, fft_magnitude[:,:,0], fft_magnitude[:,:,1], fft_magnitude[:,:,2], 
                fft_phase[:,:,0], fft_phase[:,:,1], fft_phase[:,:,2], radon]
    # for a in imgs_show:
        # print(np.min(a), np.max(a))
    if save_path != None:
        save_path = f'{save_path}-fft_radon_example_images-{title}'
        
    show_images(imgs_show, labels, img_per_row=8, title=title, img_height=0.5, hist_bins=20, 
                show_colorbar=True, axes=axes, save_path=save_path)


# def fft_radon_example_images(file, save_path = None):
#     outputs = []
#     labels = []    
    
#     img = plt.imread(file)[:,:,:3]
#     outputs.append(img)
#     labels.append('original')
    
#     ms, ps = fft_transform(img, return_type='list')
#     outputs += ms
#     outputs += ps
#     labels += ['fft_magnitude-0', 'fft_magnitude-1', 'fft_magnitude-2', 'fft_phase-0', 'fft_phase-1', 'fft_phase-2']
    
#     sinogram = radon_transform(img)
#     outputs.append(sinogram)
#     labels.append('radon')
    
#     title = file.split('/')[-1].split('.')[0].split('-')[-1]
#     if save_path != None:
#         save_path = f'{save_path}-fft_radon_example_images-{title}'
#     show_images(outputs, labels, img_per_row=8, title=title, show_colorbar=True, img_height=0.5, 
#                 save_path=save_path)
