# import modules
import os
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
# import sys
# sys.path.append('../utils/')
from viz import labelfigs, create_axes_grid
from style import set_style

def freeze_layers(model, layers_to_freeze):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False


def show_cm(files, keywords, summary=False, title_head=None, file_path=None, **kwargs):
    symmetry_classes = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm', 'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']

    def sort_key(file, sorting_keys):
        for index, key in enumerate(sorting_keys):
            if key in file:
                return index
        return len(sorting_keys)

    sorted_files = sorted(files, key=lambda file: sort_key(file, keywords))

    if summary:
        # if len(files) != 4:
        #     print('Summary requires 4 files for current layout.')
        #     return

        fig, axes = create_axes_grid(n_plots=len(files), n_per_row=2, plot_height=4, n_rows=None, figsize='auto')

        # fig, axes = plt.subplots(2, 2, figsize=(6.5, 5))
        for i, (ax, file) in enumerate(zip(axes.flatten(), sorted_files)):
            cm = np.load(file)
            plot_cm(cm, symmetry_classes, title=None, ax=ax, fig_index=i, **kwargs)
        plt.tight_layout()
        plt.savefig(f'{file_path}.png', dpi=600)
        plt.savefig(f'{file_path}.svg', dpi=300)
        plt.show()

    else:
        for file, group in zip(sorted_files, keywords):
            cm = np.load(file)
            if title_head:
                title = title_head + group
            else:
                title = None

            if file_path:
                save_file = file_path + title
            else:   
                save_file = None
            plot_cm(cm, symmetry_classes, title=title, file_path=save_file, **kwargs)
        

# imagenet 
def confusion_matrix(model, dataloader, classes, device, n_batches=1):
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
            if i == j: right+=cm[i,j]
            if i != j: wrong+=cm[i,j]
    print(f'Accuracy for these batches: {right/(right+wrong)*100}%')
    return cm.astype(np.int32)


def plot_cm(cm, classes, title=None, file_path=None, ax=None, cm_style='simple', fig_index=None, font_size=None, figsize=None, fig_style='notebook'):

    if isinstance(ax, type(None)):
        show = True
        set_style(fig_style)
        if figsize != None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1)
    else:
        show = False
        
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'Arial'

    
    if cm_style == 'simple':
        # print(ax)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp = disp.plot(cmap=plt.cm.Blues, ax=ax)
        
        if font_size != None:
            for labels in disp.text_.ravel():
                labels.set_fontsize(font_size)
        if fig_index != None:
            labelfigs(ax, number=fig_index, style='b', loc='tr', inset_fraction=(0.12, 0.12))
        
        ax.set_xticklabels(labels=classes, rotation=45)
    
    if cm_style== 'with_axis':
        df_cm = pd.DataFrame(cm)
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'

        res = sns.heatmap(df_cm, annot=True, square=True, cmap='Blues',
                         xticklabels = classes, yticklabels=classes, fmt='g', 
                         ax=ax, cbar_kws={'label': 'Number of Images'})

        res.axhline(y = 0, color = 'k', linewidth = 1)
        res.axhline(y = 16.98, color = 'k', linewidth = 1)
        res.axvline(x = 0, color = 'k', linewidth = 1)
        res.axvline(x = 16.98, color = 'k', linewidth = 1)
        
    # ax.set_xticks([])
    # ax.set_yticks([])
        
    if title != None:
        ax.set_title(title)
        
    if show:
        if file_path:
            if not os.path.isdir(os.path.dirname(file_path)):
                os.mkdir(os.path.dirname(file_path))
                
            plt.savefig(file_path+'.png', dpi=600)
            plt.savefig(file_path+'.svg', dpi=300)

        plt.show()
    
    
def parameter_analysis(classes, info_file, visualize=False, print_lens=False):
    if print_lens: print(len(classes))
        
    result = pd.DataFrame()
    for symmetry in classes:
        info_pd = pd.read_csv(info_file)

        data_1 = info_pd[info_pd['symmetry']==symmetry]
        data_2 = pd.DataFrame(data_1, columns=['radius_a','radius_b','unit_w','unit_h','repeat_w','repeat_h','angle_r','rot_angle'])

        values = pd.DataFrame({symmetry:data_2.mean()})
        if print_lens: print(f'{symmetry}: {len(data_1)} images.')
        result = pd.concat((result, values), axis=1)
    
    result = result.transpose()
    if visualize: 
        result.plot(kind='line', figsize=(12,10), xticks=np.arange(0,17))
    return result


def prediction_vs_actual(model, ds, dl, device, num_images=6):
    
    def ax_imshow(inp, ax, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        ax.imshow(inp)

    model.eval()
    fig = plt.figure(figsize=(10, num_images//3*3))

    with torch.no_grad():
        inputs, labels = next(iter(dl))
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
    class_names = ds.classes

    for i in range(num_images):
    
        r = np.random.randint(0, inputs.size()[0])
        ax = fig.add_subplot(num_images//3, 3, i+1)
        ax.axis('off')
        ax.set_title(f'predicted: {class_names[preds[r]]}, actual:{class_names[labels[r]]}')
        ax_imshow(inputs.cpu().data[r], ax) 
        
    plt.show()


def most_confused(model, batch, t, p, classes, device, k=5):

    model.eval()    
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    model = model.to(device)

    t_n, p_n = classes.index(t), classes.index(p)
    data = data[torch.where(labels==t_n)]
    labels = labels[torch.where(labels==t_n)]
    pred_label = 0
    
    while pred_label != p:
        i = np.random.randint(0, data.shape[0])
        image, label = data[i], labels[i]

        image_nor = torch.clone(image)
        inp = image_nor.unsqueeze(0).to(device)
        output = model(inp)
        pred_label = torch.nn.Softmax(dim=1)(output).argmax(dim=1, keepdim=True)
    
    probs, values_ = torch.nn.Softmax(dim=1)(output).topk(k)
        
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.title(label)
    plt.imshow(image[:3].reshape(256,256,3))
    plt.subplot(1, 2, 2)
    plt.title(label)
    plt.imshow(image[3])
    plt.show()
    
    print(f'top {k} wrong precitions are: ')
    for prob, symmetry in zip(probs[0], classes):
          print(symmetry,':', round(prob.item()*100,6),'%')
            
            