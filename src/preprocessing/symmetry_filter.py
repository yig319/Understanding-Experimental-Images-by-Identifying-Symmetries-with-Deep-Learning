import os
import random
import glob
import numpy as np
from scipy.ndimage import rotate
from IPython.display import display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class SymmetryFilter(nn.Module):

    def __init__(self, r_folds=2, n_weight=1, threshold=0, noise_level=0, 
                 kernel_size=4, stride=1, padding=0, same=False, device=torch.device("cpu")):

        super(SymmetryFilter, self).__init__()

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
        x = x.to(self.device)
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
                    x = torch.std(torch.stack((x, x_), dim=0), dim=0)
                    
                else:
                    x = self.std_forward(x)
        return self.scalar(x)

def symmetry_preprocess_example(file, setting_dict, show=False):

    r_folds = setting_dict['r_folds']
    kernel_size = setting_dict['kernel_size']

    threshold = setting_dict['threshold']
    noise_level = setting_dict['noise_level']
    n_weight = setting_dict['n_weight']
    
    device = setting_dict['device']
    
    
    img = plt.imread(file)[:,:,:3]
    f = SymmetryFilter(r_folds=r_folds, n_weight=n_weight, threshold=threshold,
                        noise_level=noise_level, kernel_size=kernel_size, device=device)
    result = f(torch.permute(torch.tensor(img).to(torch.float32), (2,0,1)).unsqueeze(0))


    add_up = F.pad(result, [(img.shape[0]-result.shape[2])//2, (img.shape[0]-result.shape[2])-(img.shape[0]-result.shape[2])//2,
                            (img.shape[1]-result.shape[3])//2, (img.shape[1]-result.shape[3])-(img.shape[1]-result.shape[3])//2])
    add_up = torch.permute(add_up.squeeze(), (1,2,0)).detach().cpu()
    img_add_up = scalar(torch.tensor(img) + add_up).numpy()
    img_preprocessed = torch.permute(result.squeeze(), (1,2,0)).detach().cpu().numpy()

    if show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        im0 = axes[0].imshow(img)
        axes[0].title.set_text('original image')
        axes[0].title.set_size(20)
        im1 = axes[1].imshow(img_add_up)
        axes[1].title.set_text('Add up')
        axes[1].title.set_size(20)
        im2 = axes[2].imshow(img_preprocessed)
        axes[2].title.set_text('preprocessed with '+ str(setting_dict['r_folds'])+' folds\n')
        axes[2].title.set_size(20)
        plt.title(os.path.split(os.path.split(file)[0])[-1])
        plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(15, 3))
        im0 = axes[0].hist(img.flatten())
        axes[0].title.set_text('original image '+os.path.split(os.path.split(file)[0])[-1])
        im1 = axes[1].hist(img_add_up.flatten())
        axes[1].title.set_text('Add up')
        im2 = axes[2].hist(img_preprocessed.flatten())
        axes[2].title.set_text('preprocessed with '+ str(setting_dict['r_folds'])+' folds\n')
        plt.show()

    return img, img_add_up, img_preprocessed
        
def scalar(img):
    img -= img.min()
    img /= img.max()
    return img