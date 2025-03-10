import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from m3util.viz.layout import layout_fig

def sort_history_files_by_size(files):
    def extract_size(file_path):
        """Extracts and converts dataset size to a numerical value for sorting."""
        match = re.search(r"size-(\d+)([km])", file_path)
        if match:
            num = int(match.group(1))  # Extract number
            unit = match.group(2)  # Extract unit ('k' or 'm')
            return num * (1_000 if unit == 'k' else 1_000_000)  # Convert to actual size
        return 0  # Default if no match found
    
    def extract_task_name(file_path):
        """Extracts model name and dataset size in a formatted way."""
        match = re.search(r"(\w+)-dataset.*?-([\d]+[mk])", file_path)
        if match:
            return f"{match.group(1)}-{match.group(2)}"  # Example: "resnet50-10m"
        return "unknown"

    # Sort the files by dataset size
    sorted_files = sorted(files, key=extract_size)

    # Create a dictionary with extracted task names as keys
    sorted_dict = {extract_task_name(file): file for file in sorted_files}

    return sorted_dict

def plot_training_history(log_file, task_name, x_axis='epoch', printing=None, filename=None):
    
    # Extract using regex the model name and dataset size in a hardcoded format
    match = re.search(r"(\w+)-dataset.*?-([\d]+m)", log_file)
    if match:
        extracted_text = f"{match.group(1)}-{match.group(2)}"
        task_name = extracted_text  # Output: resnet50-10m
    
    # Load the data
    df = pd.read_csv(log_file)

    loss_names = ['train_loss', 'valid_loss', 'valid_noise_loss', 'valid_atom_loss']
    acc_names = ['train_accuracy', 'valid_accuracy', 'valid_noise_accuracy', 'valid_atom_accuracy']

    # Convert x_axis to numeric and sort by epoch
    df[x_axis] = pd.to_numeric(df[x_axis], errors='coerce')
    df = df.sort_values(by=x_axis)

    # Melt DataFrame to long format for Seaborn
    df_loss = df.melt(id_vars=[x_axis], value_vars=loss_names, var_name='Loss Type', value_name='Loss Value')
    df_acc = df.melt(id_vars=[x_axis], value_vars=acc_names, var_name='Accuracy Type', value_name='Accuracy Value')

    fig, axes = layout_fig(2, 1, figsize=(6, 4), layout='tight')
    sns.scatterplot(data=df_loss, x=x_axis, y='Loss Value', hue='Loss Type', style='Loss Type', ax=axes[0], s=15, edgecolor='none', alpha=0.8, legend=False)
    sns.lineplot(data=df_loss, x=x_axis, y='Loss Value', hue='Loss Type', style='Loss Type', ax=axes[0], linewidth=1, alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc='best')

    sns.scatterplot(data=df_acc, x=x_axis, y='Accuracy Value', hue='Accuracy Type', style='Accuracy Type', ax=axes[1], s=15, edgecolor='none', alpha=0.8, legend=False)
    sns.lineplot(data=df_acc, x=x_axis, y='Accuracy Value', hue='Accuracy Type', style='Accuracy Type', ax=axes[1], linewidth=1, alpha=0.8)
    axes[1].set_xlabel(x_axis)
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc='best')
    
    if printing is not None and filename is not None:
        printing.savefig(fig, filename)
    
    plt.suptitle(f"{task_name}", fontsize=10)
    plt.show()