# import modules
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
from dl_utils.utils.utils import NormalizeData
from m3util.viz.layout import layout_fig


def verify_image_vector(ax, image, ts, va, vb, shade_alpha=0.3, shade_color='gray'):
    """
    Visualizes an image with two vectors and shades the region they define.

    Parameters:
        ax (matplotlib axis): Axis to plot on.
        image (ndarray): The image to display.
        ts (tuple): Starting point (y, x).
        va (tuple): First vector (dy, dx).
        vb (tuple): Second vector (dy, dx).
        shade_alpha (float): Transparency of the shaded region (0=transparent, 1=solid).
        shade_color (str): Color of the shaded region.
    """
    # Convert (y, x) format to (x, y) for plotting
    ts = np.array([ts[1], ts[0]])  # Convert to (x, y)
    va = np.array([va[1], va[0]])  # Convert to (dx, dy)
    vb = np.array([vb[1], vb[0]])  # Convert to (dx, dy)

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        show = True
    else:
        show = False

    # Display the image
    ax.imshow(image, cmap='viridis')

    # Set axis labels
    ax.set_ylabel('Y-axis')
    ax.set_xlabel('X-axis')

    if shade_alpha != 0 and shade_color is not None:
        # Compute the four corners of the parallelogram
        p1 = ts
        p2 = ts + va
        p3 = ts + vb
        p4 = ts + va + vb

        # Draw the shaded region
        ax.fill([p1[0], p2[0], p4[0], p3[0]], 
                [p1[1], p2[1], p4[1], p3[1]], 
                color=shade_color, alpha=shade_alpha, label="Shaded Region")

    # Draw quiver arrows for vectors
    ax.quiver(ts[0], ts[1], va[0], va[1], color='b', angles='xy', scale_units='xy', scale=1, linewidth=0.5)
    ax.quiver(ts[0], ts[1], vb[0], vb[1], color='b', angles='xy', scale_units='xy', scale=1, linewidth=0.5)

    # Display legend
    ax.legend()

    # Draw plot
    plt.draw()

    if show:
        plt.show()



def verify_image_in_hdf5_file(ds_path, n_list, group, keys={}, viz=True):
    if keys == {}:
        keys = {
            'data': 'data',
            'labels': 'labels',
            'ts': 'translation_start_point',
            'va': 'primitive_uc_vector_a',
            'vb': 'primitive_uc_vector_b',
            'VA': 'translation_uc_vector_a',
            'VB': 'translation_uc_vector_b'
        }
        
    symmetry_dict = {'p1': 0, 'p2': 1, 'pm': 2, 'pg': 3, 'cm': 4, 'pmm': 5, 'pmg': 6, 'pgg': 7, 'cmm': 8, 
                        'p4': 9, 'p4m': 10, 'p4g': 11, 'p3': 12, 'p3m1': 13, 'p31m': 14, 'p6': 15, 'p6m': 16}

    symmetry_inv_dict = {v: k for k, v in symmetry_dict.items()}

    
    with h5py.File(ds_path, 'r') as h5:
        if viz:
            print('Total number of images in the dataset: ', len(h5[group]['data']))
        if isinstance(n_list, int):
           n_list = random.choices(range(len(h5[group]['data'])), k=n_list)
        n_list = np.sort(n_list)
        if viz:
            print('Randomly selected images: ', n_list)
        imgs = np.array(h5[group][keys['data']][n_list])
        # unit_cells = np.array(h5[group]['unit_cell'][n_list])
        labels = np.array(h5[group][keys['labels']][n_list])
        ts_list = np.array(h5[group][keys['ts']][n_list])
        va_list = np.array(h5[group][keys['va']][n_list])
        vb_list = np.array(h5[group][keys['vb']][n_list])
        VA_list = np.array(h5[group][keys['VA']][n_list])
        VB_list = np.array(h5[group][keys['VB']][n_list])
    
    labels_str = [symmetry_inv_dict[l] for l in labels]
    metadata = {'ts': ts_list, 'va': va_list, 'vb': vb_list, 'VA': VA_list, 'VB': VB_list}
    if viz:
        for i in range(len(n_list)):
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
            # verify_image_vector(axes[0], unit_cells[i], ts_list[i], va_list[i], vb_list[i])
            verify_image_vector(axes[0], imgs[i], ts_list[i], va_list[i], vb_list[i])
            verify_image_vector(axes[1], imgs[i], ts_list[i], VA_list[i], VB_list[i])
            plt.title(labels_str[i])
            plt.show()
        
    return imgs, labels_str, metadata


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

