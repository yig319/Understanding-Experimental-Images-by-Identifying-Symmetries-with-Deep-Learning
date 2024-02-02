import matplotlib.pyplot as plt
import numpy as np
# import torch

def trim_axes(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def show_images(images, labels=None, img_per_row=8, img_height=1, label_size=12, title=None, show_colorbar=False, 
                clim=3, cmap='viridis', scale_0_1=False, hist_bins=None, show_axis=False, save_path=None):
    
    '''
    Plots multiple images in grid.
    
    images
    labels: labels for every images;
    img_per_row: number of images to show per row;
    img_height: height of image in axes;
    show_colorbar: show colorbar;
    clim: int or list of int, value of standard deviation of colorbar range;
    cmap: colormap;
    scale_0_1: scale image to 0~1;
    hist_bins: number of bins for histogram;
    show_axis: show axis
    '''
    
    assert type(images) == list or type(images) == np.ndarray, "do not use torch.tensor for hist"
    if type(clim) == list:
        assert len(images) == len(clim), "length of clims is not matched with number of images"

    def scale(x):
        if x.min() < 0:
            return (x - x.min()) / (x.max() - x.min())
        else:
            return x/(x.max() - x.min())
    
    h = images[0].shape[1] // images[0].shape[0]*img_height + 1
    if not labels:
        labels = range(len(images))
        
    n = 1
    if hist_bins: n +=1
        
    fig, axes = plt.subplots(n*len(images)//img_per_row+1*int(len(images)%img_per_row>0), img_per_row, 
                             figsize=(16, n*h*len(images)//img_per_row+1), constrained_layout=True)
    trim_axes(axes, len(images))

    for i, img in enumerate(images):
        
#         if torch.is_tensor(x_tensor):
#             if img.requires_grad: img = img.detach()
#             img = img.numpy()
            
        if scale_0_1: img = scale(img)
        
        if len(images) <= img_per_row and not hist_bins:
            index = i%img_per_row
        else:
            index = (i//img_per_row)*n, i%img_per_row

        axes[index].set_title (labels[i], fontsize=label_size)
        im = axes[index].imshow(img, cmap=cmap)

        if show_colorbar:
            m, s = np.mean(img), np.std(img) 
            if type(clim) == list:
                im.set_clim(m-clim[i]*s, m+clim[i]*s) 
            else:
                im.set_clim(m-clim*s, m+clim*s) 

            fig.colorbar(im, ax=axes[index])
            
        if show_axis:
            axes[index].tick_params(axis="x",direction="in", top=True)
            axes[index].tick_params(axis="y",direction="in", right=True)
        else:
            axes[index].axis('off')

        if hist_bins:
            index_hist = (i//img_per_row)*n+1, i%img_per_row
            h = axes[index_hist].hist(img.flatten(), bins=hist_bins)

        if title:
            fig.suptitle(title, fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.tight_layout()
    plt.show()
    
    
def show_plots(ys, xs=None, labels=None, ys_fit=None, img_per_row=4, subplot_height=3, ylim=None):
#     matplotlib.rcParams.update(matplotlib.rcParamsDefault)
#     plt.rcParams.update(mpl.rcParamsDefault)

    if type(labels) == type(None): labels = range(len(ys))
    
    if type(xs) == type(None):
        xs = []
        for y in ys:
            xs.append(np.linspace(0, len(y), len(y)+1))            
        
    fig, axes = plt.subplots(len(ys)//img_per_row+1*int(len(ys)%img_per_row>0), img_per_row, 
                             figsize=(16, subplot_height*len(ys)//img_per_row+1))    
    trim_axes(axes, len(ys))
    
    for i in range(len(ys)):
        
        if len(ys) <= img_per_row:
            index = i%img_per_row
        else:
            index = (i//img_per_row), i%img_per_row

        axes[index].title.set_text(labels[i])
        
        im = axes[index].plot(xs[i], ys[i], marker='.')
        
        if type(ys_fit) != type(None):
            im = axes[index].plot(xs[i], ys_fit[i])
        
        if type(ylim) != type(None):
            axes[index].set_ylim([ylim[0], ylim[1]])

    fig.tight_layout()
    plt.show()
