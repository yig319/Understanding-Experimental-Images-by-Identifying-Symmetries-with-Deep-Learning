import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
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
        plt.savefig(save_path+'.svg', dpi=300)
        plt.savefig(save_path+'.png', dpi=300)
    # plt.tight_layout()
    plt.show()
    
    
def labelfigs(ax, number=None, style="wb",
              loc="tl", string_add="", size=8,
              text_pos="center", inset_fraction=(0.15, 0.15), **kwargs):

    # initializes an empty string
    text = ""

    # Sets up various color options
    formatting_key = {
        "wb": dict(color="w", linewidth=.75),
        "b": dict(color="k", linewidth=0),
        "w": dict(color="w", linewidth=0),
    }

    # Stores the selected option
    formatting = formatting_key[style]

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_inset = (xlim[1] - xlim[0]) * inset_fraction[1]
    y_inset = (ylim[1] - ylim[0]) * inset_fraction[0]

    if loc == 'tl':
        x, y = xlim[0] + x_inset, ylim[1] - y_inset
    elif loc == 'tr':
        x, y = xlim[1] - x_inset, ylim[1] - y_inset
    elif loc == 'bl':
        x, y = xlim[0] + x_inset, ylim[0] + y_inset
    elif loc == 'br':
        x, y = xlim[1] - x_inset, ylim[0] + y_inset
    elif loc == 'ct':
        x, y = (xlim[0] + xlim[1]) / 2, ylim[1] - y_inset
    elif loc == 'cb':
        x, y = (xlim[0] + xlim[1]) / 2, ylim[0] + y_inset
    else:
        raise ValueError(
            "Invalid position. Choose from 'tl', 'tr', 'bl', 'br', 'ct', or 'cb'.")

    text += string_add

    if number is not None:
        text += number_to_letters(number)

    text_ = ax.text(x, y, text, va='center', ha='center',
                      path_effects=[patheffects.withStroke(
                      linewidth=formatting["linewidth"], foreground="k")],
                      color=formatting["color"], size=size, **kwargs
                      )

    text_.set_zorder(np.inf)

    
def number_to_letters(num):
    letters = ''
    while num >= 0:
        num, remainder = divmod(num, 26)
        letters = chr(97 + remainder) + letters
        num -= 1  # decrease num by 1 because we have processed the current digit
    return letters