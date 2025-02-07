import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from m3util.viz.layout import layout_fig
from dl_utils.analysis.case_analysis import show_prediction_example
from dl_utils.analysis.confusion_matrix import show_multiple_cm
from dl_utils.analysis.attention_map import AttentionMapVisualizer


def plot_attention_map(model, dataloder, copnfusion_pair, layers, task_name, model_type, device, filename=None):
    image = np.zeros((256, 256, 3))
    while np.sum(image) <= 10000:
        image, label, top_predictions, probs = show_prediction_example(model, dataloder, copnfusion_pair[0], copnfusion_pair[1], dataloder.dataset.classes, device, k=5, batch_limit=None, viz=False)

    visualizer = AttentionMapVisualizer(device=device)
    input_tensor = torch.tensor(image).unsqueeze(0)  # Example input tensor (N, C, H, W)
    stats = f"True: {label} | Predicted: " + f", ".join(f"{pred}({prob.item() * 100:.2f}%)" for prob, pred in zip(probs, top_predictions))

    layers = ['layer4', 'layer3', 'layer2', 'layer1']
    fig, axes = layout_fig(len(layers)*3, 3, figsize=(8, len(layers)*2.8), subplot_style='subplots', layout='tight')
    for i, layer in enumerate(layers):
        if model_type == 'ResNet50':
            input_image_np, attention_map_resized = visualizer.generate_cnn_attention_map(model, input_tensor, layer_name=layer)
        elif model_type == 'XCiT':
            input_image_np, attention_map_resized = visualizer.generate_transformer_attention_map(model, input_tensor, attention_layer_idx=layer)
            
        visualizer.visualize_attention_map(input_image_np, attention_map_resized, keyword=layer, fig=fig, axes=axes[i*3:i*3+3])
        
    plt.suptitle(f"{task_name}: {stats}", fontsize=10)
    if filename:
        plt.savefig(f'{filename}.png', dpi=600)
        plt.savefig(f'{filename}.svg', dpi=600)
    
    plt.show()
    
    return input_image_np, attention_map_resized


def viz_4confusion_matrix(cm_files, title, symmetry_classes, filename=None):
    
    cm_list = []
    subtitles = []
    dataset_list = ['imagenet', 'imagenet', 'atom', 'noise']
    task_list = ['train', 'valid', 'cross_validation', 'cross_validation']
    for dataset, task in zip(dataset_list, task_list):

        matching_paths = [path for path in cm_files if f'{dataset}_{task}' in path]
        if matching_paths == 0 or len(matching_paths) > 1:
            raise ValueError(f'Invalid number of matching paths found: {matching_paths}')
        else:
            path = matching_paths[0]
        cm = np.load(path)
        cm_list.append(cm)
        subtitles.append(f'{dataset}-{task}')

    fig, axes = layout_fig(len(cm_list), 2, figsize=(10, 8))
    show_multiple_cm(cm_list, subtitles, fig=fig, axes=axes, suptitle=title, classes=symmetry_classes, font_size=4)

    if filename:
        plt.savefig(f'{filename}.png', dpi=600)
        plt.savefig(f'{filename}.svg', dpi=600)
    plt.show()
    
    return cm_list