import os
import shutil
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import radon, rescale

def scale(x):
    if x.min() < 0:
        return (x - x.min()) / (x.max() - x.min())
    else:
        return x/(x.max() - x.min())
    
def log_scale(fft_real):
    fft_real_log = np.log(np.clip(fft_real, 1e-5,None))
    fft_real_log[fft_real_log == float('-inf')] = 0
    return scale(fft_real_log)

def fft_transform(image, return_type='numpy'):
    ms_list, ps_list = [], []
    for channel in range(image.shape[2]):  # Assuming the image has three channels: R, G, B
        img_fft = np.fft.fft2(image[:,:,channel])
        img_fft_shift = np.fft.fftshift(img_fft)
        
        ms = scale(np.log1p(np.abs(img_fft_shift)))
        ms_list.append(ms)
        ps = scale(np.angle(img_fft_shift))
        ps_list.append(np.angle(img_fft_shift))
        
    if return_type == 'numpy':
        ms = np.stack(ms_list, axis=-1)
        ps = np.stack(ps_list, axis=-1)
        return scale(ms), scale(ps)
    elif return_type == 'list':
        for channel in range(image.shape[2]): 
            ms_list[channel] = scale(ms_list[channel])
            ps_list[channel] = scale(ps_list[channel])
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



