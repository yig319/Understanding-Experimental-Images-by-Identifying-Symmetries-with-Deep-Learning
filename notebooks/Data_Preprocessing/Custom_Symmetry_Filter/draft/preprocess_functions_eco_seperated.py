import os
import math
import time
import operator

import numpy as np
from scipy.ndimage import rotate
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torchvision


class symmetry_filter(nn.Module):

    def __init__(self, device=torch.device("cpu"), n_weight=1, r_folds=2, threshold=0, noise_level=0, 
                 kernel_size=4, stride=1, padding=0, same=False):

        super(symmetry_filter, self).__init__()

#         self.k = _pair(kernel_size)        # kernel size has to be n*n:
        self.k = kernel_size        # kernel size has to be n*n:

        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same
        self.device = device
        self.n_weight = n_weight
        self.r_folds = r_folds
        self.threshold = threshold
        self.noise_level = noise_level

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k - self.stride[0], 0)
            else:
                ph = max(self.k - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k - self.stride[1], 0)
            else:
                pw = max(self.k - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding    

    def uniform_rand(self, r1, r2, shape):
        return (r1 - r2) * torch.rand(shape) + r2

    def scalar(self, img):
        img -= img.min()
        img /= img.max()
        return img

    def remove_blank(self, x, sliding):
        diff = torch.max(sliding.flatten(start_dim=4), dim=4)[0] - torch.min(sliding.flatten(start_dim=4), dim=4)[0]
        x = torch.where(diff > self.threshold, x, torch.zeros(x.shape).to(self.device).double())
        return x

    def add_noise(self, tensor):
        noise = (self.uniform_rand(0,1,tensor.shape)*self.noise_level).to(self.device)
        return self.scalar(tensor+noise)
    
    
    
    def set_weight(self):
        # set a random matrix for weights without rotation
        weight = self.uniform_rand(0,2,[1, self.channels, self.channels, self.k, self.k])

        return weight.double().to(self.device)

    def std_forward(self, x):

        # crop the edge of square matrix, leave a round-shape matrix with edges to be zero.
        
        v_all = []
        for w in range(self.n_weight):
            values = []
            self.weight = self.set_weight()

            for n in range(self.r_folds):
                self.weight = torch.tensor(rotate(self.weight.cpu().float(), angle=360//self.r_folds, axes=(3,4), reshape=False)).to(self.device).double()

            for n in range(self.r_folds):
                values.append(torch.sum( torch.einsum("ijmnpq,tkjpq->ikmnpq", x, self.weight), 
                                        dim=(4,5)) )    
                self.weight = torch.tensor(rotate(self.weight.cpu().float(), angle=360//self.r_folds, axes=(3,4), reshape=False)).to(self.device).double()

            values = torch.std(torch.stack(values, dim=0), dim=0)
            v_all.append(values)   
            
        x = torch.mean(1-torch.stack(v_all, dim=0), dim=0)

        return x
    
    
    def std_forward_mirror(self, x):
        
        self.weight = self.set_weight()

        values = []
        values.append( torch.sum(torch.einsum("ijmnpq,tkjpq->tikmnpq", x, self.weight), 
                                    dim=(5,6)).unsqueeze(-1) )    

        self.weight = torch.flip(self.weight, dims=[4])
        values.append( torch.sum(torch.einsum("ijmnpq,tkjpq->tikmnpq", x, self.weight), 
                                    dim=(5,6)).unsqueeze(-1) )  
        
        x = torch.cat(values, dim=5)
        x = torch.mean(1-torch.std(x, dim=5), dim=0)
        return x
    
    
    def std_forward_glide(self, x):
        self.weight = self.set_weight()

        x = [ torch.sum(torch.einsum("ijmnpq,tkjpq->tikmnpq", x[:,:,:,:,:self.k//2, :], self.weight[:,:,:,:self.k//2, :]), 
                                    dim=(5,6)).unsqueeze(-1), 
              torch.sum(torch.einsum("ijmnpq,tkjpq->tikmnpq", x[:,:,:,:,-self.k//2:, :], self.weight[:,:,:,-self.k//2:, :]), 
                                    dim=(5,6)).unsqueeze(-1) ]
        
        x = torch.cat(x, dim=5)
        x = torch.mean(1-torch.std(x, dim=5), dim=0)
        return x

    
    def forward(self, x):

        x = F.pad(x, self._padding(x), mode='reflect')
        
        if self.noise_level != 0:
            x = self.add_noise(x)
            
        # create sliding windows for input batch
        x = x.unfold(2, self.k, self.stride[0]).unfold(3, self.k, self.stride[1]).double()

        # x.shape: batch_size, channels, H, W, h, w
        self.channels = x.shape[1]

        if self.threshold != 0:
            sliding = x.clone()
            if self.r_folds == 'mirror':
                x = self.std_forward_mirror(x)
            elif self.r_folds == 'glide':
                x = self.std_forward_glide(x)
            else:
                x = self.std_forward(x)
            x = self.remove_blank(x, sliding)
            
        else:
            if self.r_folds == 'mirror':
                x = self.std_forward_mirror(x)
            elif self.r_folds == 'glide':
                x = self.std_forward_glide(x)
            else:
                if self.r_folds == 2:
                    x_ = self.std_forward(x)
                    self.r_folds = 4
                    _x = self.std_forward(x)
                    self.r_folds = 6
                    x = self.std_forward(x)

                    x = torch.std(torch.stack((x, _x, x_), dim=0), dim=0)
                    
                    
                elif self.r_folds == 3:
                    x_ = self.std_forward(x)
                    self.r_folds = 6
                    x = self.std_forward(x)
                    print(x.shape)
                    x = torch.std(torch.stack((x, x_), dim=0), dim=0)
                    
                else:
                    x = self.std_forward(x)
        return self.scalar(x)
    
    
def preprocess_dataset(kernel_size, symmetries, path_from, path_to, device, func):

    for folder in os.listdir(path_from):

        if not os.path.isdir(path_to): os.mkdir(path_to)
        if not os.path.isdir(path_to + folder): os.mkdir(path_to + folder)
        print(folder)

        for symmetry in symmetries:
            if symmetry != 'info.csv':
                if not os.path.isdir(path_to + folder + '/' + symmetry): os.mkdir(path_to + folder + '/' + symmetry)
                print(symmetry)

                for i, file in enumerate(os.listdir(path_from + folder + '/' + symmetry)):
                    img = plt.imread(path_from + folder + '/' + symmetry + '/' + file)[:,:,:3]
                    img_ = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)

                    result = func(torch.tensor(img_).unsqueeze(0).to(device))

                    result = np.swapaxes(np.swapaxes(result[0].cpu().numpy(), 0, 2), 1, 0)

                    if i%2000 == 0:
                        print(path_to + folder + '/' + symmetry + '/' + file)


                        fig = plt.figure(figsize=(12, 5))

                        ax1 = fig.add_subplot(121)
                        im1 = ax1.imshow(img)
                        fig.colorbar(im1)

                        ax2 = fig.add_subplot(122)
                        im2 = ax2.imshow(result)
                        fig.colorbar(im2)
                        plt.show()

                    plt.imsave(path_to + folder + '/' + symmetry + '/' + file, result)
        #             print(save_to + folder + '/' + symmetry + '/' + file)

        
def print_file_number(path):
    l= os.listdir(path)
    print(path)

    for folder in l:
        print(folder)
        ls = os.listdir(path+folder)
        if 'info.csv' in ls: ls.remove('info.csv')
        
        for i, s in enumerate(ls):
            print('-', i, s, len(os.listdir(path+folder+'/'+s)))