import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
# import torch
from NormalizeData import NormalizeData

def create_axes_grid(n_plots, n_per_row, plot_height, n_rows=None, figsize='auto'):
    """
    Create a grid of axes.

    Args:
        n_plots: Number of plots.
        n_per_row: Number of plots per row.
        plot_height: Height of each plot.
        n_rows: Number of rows. If None, it is calculated from n_plots and n_per_row.
        
    Returns:
        axes: Axes object.
    """
    
    if figsize == 'auto':
        figsize = (16, plot_height*n_plots//n_per_row+1)
    elif isinstance(figsize, tuple):
        pass
    elif figsize != None:
        raise ValueError("figsize must be a tuple or 'auto'")
    
    fig, axes = plt.subplots(n_plots//n_per_row+1*int(n_plots%n_per_row>0), n_per_row, figsize=figsize)
    trim_axes(axes, n_plots)
    return fig, axes


def trim_axes(axs, N):

    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    # axs = axs.flatten()
    # for ax in axs[N:]:
    #     ax.remove()
    # return axs[:N]
    for i in range(N, len(axs.flatten())):
        axs.flatten()[i].remove()
    return axs.flatten()[:N]


def show_images(images, labels=None, img_per_row=8, img_height=1, label_size=12, title=None, show_colorbar=False, 
                clim=3, cmap='viridis', scale_range=False, hist_bins=None, hist_range=None, show_axis=False, axes=None, save_path=None):
    
    '''
    Plots multiple images in grid.
    
    images
    labels: labels for every images;
    img_per_row: number of images to show per row;
    img_height: height of image in axes;
    show_colorbar: show colorbar;
    clim: int or list of int, value of standard deviation of colorbar range;
    cmap: colormap;
    scale_range: scale image to a range, default is False, if True, scale to 0-1, if a tuple, scale to the range;
    hist_bins: number of bins for histogram;
    show_axis: show axis
    '''
    
    assert type(images) == list or type(images) == np.ndarray, "do not use torch.tensor for hist"
    if type(clim) == list:
        assert len(images) == len(clim), "length of clims is not matched with number of images"

    h = images[0].shape[1] // images[0].shape[0]*img_height + 1
    if not labels:
        labels = range(len(images))
        
    if isinstance(axes, type(None)):
        if hist_bins: # add a row for histogram
            fig, axes = create_axes_grid(len(images)*2, img_per_row, img_height*2, n_rows=None, figsize='auto')
        else:
            fig, axes = create_axes_grid(len(images), img_per_row, img_height, n_rows=None, figsize='auto')
        
    axes = axes.flatten()

    for i, img in enumerate(images):

        if hist_bins:
            index = i + (i//img_per_row)*img_per_row
        else:
            index = i
            
        if isinstance(scale_range, bool): 
            if scale_range: img = NormalizeData(img)
                    
        axes[index].set_title(labels[i], fontsize=label_size)
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
            index_hist = index+img_per_row
            # index_hist = index*2+1
            if not isinstance(hist_range, type(None)):
                if not isinstance(hist_range[0], type(None)):
                    img_hist = img[img>hist_range[0]].flatten()
                if not isinstance(hist_range[1], type(None)):
                    img_hist = img[img<hist_range[1]].flatten()
            else:
                img_hist = img.flatten()
            h = axes[index_hist].hist(img_hist, bins=hist_bins)

        if title:
            fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    
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