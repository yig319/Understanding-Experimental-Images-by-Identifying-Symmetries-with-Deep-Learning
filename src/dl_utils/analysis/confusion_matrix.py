import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.metrics import ConfusionMatrixDisplay
from m3util.viz.text import labelfigs
from m3util.viz.layout import layout_fig

def confusion_matrix(model, dataloader, classes, device, n_batches=1):
    """Computes the confusion matrix for classification."""
    
    model.eval()
    cm = torch.zeros(len(classes), len(classes))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device) 
            labels = labels.to(device)
            model = model.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

    cm = np.array(cm)

    print('Sum for true labels:')
    true_counts = np.expand_dims(np.sum(cm, axis=1), 0)
    display(pd.DataFrame(true_counts, columns=classes))

    wrong, right = 0, 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                right += cm[i, j]
            else:
                wrong += cm[i, j]
    
    accuracy = right / (right + wrong)
    print(f'Accuracy for these batches: {accuracy * 100:.2f}%')
    return cm.astype(np.int32), accuracy


def show_multiple_cm(cm_list, subtitles, fig=None, axes=None, classes=None, suptitle=None, file_path=None, **kwargs):
    """
    Plots multiple confusion matrices from files, either in a summary grid or individually.

    Parameters:
        files (list): List of file paths containing confusion matrices.
        keywords (list): List of keywords used to sort and label the files.
        title_head (str, optional): Prefix for the title of each plot.
        file_path (str, optional): Base path to save the plots.
        **kwargs: Additional arguments passed to `plot_cm`.
    """
    if axes is None:
        # Create a grid layout for summary mode
        num_graph = len(cm_list)
        img_per_row = 2
        
        figsize = (3 * img_per_row, 4 * 2 * int(np.ceil(num_graph / img_per_row)))
        fig, axes = layout_fig(num_graph, img_per_row, figsize=figsize)
        # fig, axes = plt.subplots(int(np.ceil(num_graph / img_per_row)), img_per_row, figsize=figsize)
        # axes = axes.flatten()  # Flatten the axes array for easy iteration

    
    for i, (ax, cm, subtitle) in enumerate(zip(axes, cm_list, subtitles)):
        plot_cm(
            cm=cm,
            classes=classes,
            title=f"{subtitle}" if subtitle else None,
            ax=ax,
            show=False,  # Do not show individual plots
            **kwargs
        )
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
        
        
def plot_cm(
    cm, 
    classes, 
    title=None, 
    file_path=None, 
    ax=None, 
    cm_style='simple', 
    fig_index=None, 
    font_size=6, 
    figsize=None, 
    cmap='Blues', 
    xlabel='Predicted', 
    ylabel='Actual', 
    rotate_xticks=45, 
    rotate_yticks=0, 
    grid_lines=False, 
    show=True, 
    save_formats=None, 
    **kwargs
):
    """
    Plots a confusion matrix with customizable options.

    Parameters:
        cm (np.ndarray): Confusion matrix.
        classes (list): List of class labels.
        title (str, optional): Title of the plot.
        file_path (str, optional): Path to save the plot.
        ax (matplotlib.axes.Axes, optional): Axis to plot on.
        cm_style (str): Style of the confusion matrix ('simple' or 'with_axis').
        fig_index (int, optional): Index to label the figure.
        font_size (int, optional): Font size for annotations.
        figsize (tuple, optional): Figure size.
        cmap (str): Colormap for the heatmap.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        rotate_xticks (int): Rotation angle for x-axis tick labels.
        rotate_yticks (int): Rotation angle for y-axis tick labels.
        grid_lines (bool): Whether to show grid lines.
        show (bool): Whether to display the plot.
        save_formats (list, optional): List of formats to save the plot (e.g., ['png', 'svg']).
        **kwargs: Additional arguments passed to the plotting function.
    """
    # Validate inputs
    if not isinstance(cm, np.ndarray):
        raise ValueError("`cm` must be a numpy array.")
    if len(classes) != cm.shape[0]:
        raise ValueError("Length of `classes` must match the dimensions of `cm`.")

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize if figsize else (8, 6))
    else:
        show = False

    # Plot confusion matrix based on style
    if cm_style == 'simple':
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=cmap, ax=ax, **kwargs)
        
        if font_size:
            for labels in disp.text_.ravel():
                value = float(labels.get_text())  # Convert text to float
                labels.set_text(f"{int(value):d}")  # Force integer formatting
                labels.set_fontsize(font_size)
                
        if fig_index is not None:
            labelfigs(ax, number=fig_index, style='b', loc='tr', inset_fraction=(0.12, 0.12))
        
        ax.set_xticklabels(labels=classes, rotation=rotate_xticks)
        ax.set_yticklabels(labels=classes, rotation=rotate_yticks)
        ax.tick_params(axis='both', which='both', length=0)  # Hide ticks but keep labels

    elif cm_style == 'with_axis':
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        df_cm.index.name = ylabel
        df_cm.columns.name = xlabel

        sns.heatmap(
            df_cm, 
            annot=True, 
            square=True, 
            cmap=cmap, 
            fmt='d', 
            ax=ax, 
            cbar_kws={'label': 'Number of Images'}, 
            **kwargs
        )
        ax.tick_params(axis='both', which='both', length=0)  # Hide ticks but keep labels

        if grid_lines:
            ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    
    # Set title
    if title:
        ax.set_title(title)

    # Save plot if file_path is provided
    if file_path and save_formats:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        for fmt in save_formats:
            plt.savefig(f"{file_path}.{fmt}", dpi=600)

    # Show plot
    if show:
        plt.tight_layout()
        plt.show()
