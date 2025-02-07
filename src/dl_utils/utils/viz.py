# import modules
import numpy as np
import matplotlib.pyplot as plt
from dl_utils.utils.utils import NormalizeData
from m3util.viz.layout import layout_fig


def show_images(images, labels=None, img_per_row=8, img_height=1, label_size=12, title=None, show_colorbar=False, clim=3, cmap='viridis', scale_range=False, hist_bins=None, hist_range=None, show_axis=False, axes=None, save_path=None):
    
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
            num_graph = len(images)*2 
            figsize = ( 3*img_per_row, img_height*2 * int(np.ceil(num_graph/img_per_row)) )
            fig, axes = layout_fig(num_graph, img_per_row, figsize=figsize, layout='tight')
        else:
            num_graph = len(images)
            figsize = ( 3*img_per_row, img_height*2 * int(np.ceil(num_graph/img_per_row)) )
            fig, axes = layout_fig(num_graph, img_per_row, figsize=figsize, layout='tight')
            # fig, axes = create_axes_grid(len(images), img_per_row, img_height, n_rows=None, figsize='auto')
        
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

