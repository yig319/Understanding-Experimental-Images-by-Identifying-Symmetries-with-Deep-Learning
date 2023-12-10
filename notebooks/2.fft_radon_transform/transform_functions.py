import os
import shutil
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage.transform import radon

def scale(x):
    if x.min() < 0:
        return (x - x.min()) / (x.max() - x.min())
    else:
        return x/(x.max() - x.min())
    
def log_scale(fft_real):
    fft_real_log = np.log(np.clip(fft_real, 1e-5,None))
    fft_real_log[fft_real_log == float('-inf')] = 0
    return scale(fft_real_log)

def fft_transform(image):
    image = torch.tensor(image)
    out = torch.clone(image)
    img_fft = torch.fft.fft2(image, dim=(0,1))
    img_shift = torch.fft.fftshift(img_fft)
    out = np.log(np.abs(img_shift))
    out[out==-np.inf] = 0
    out = scale(out)  
    return out

def radon_transform(image):
    img = np.copy(image)
    if str(type(img)) != "<class 'numpy.ndarray'>":
        img = img.numpy()
    if len(img.shape) == 3:
        if img.shape[0] == 3: # shape: 3,256,256 
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 0, 1)
        img = color.rgb2gray(img)
   
    theta = np.linspace(0., 180., len(image), endpoint=False)
    sinogram = radon(img, theta=theta, circle=True)
    return scale(sinogram)
#     return np.expand_dims(sinogram, axis=0)