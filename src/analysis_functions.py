# import modules
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.metrics import ConfusionMatrixDisplay

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
    print('Accuracy for these batches:', right/(right+wrong))
    return cm.astype(np.int32)

def plot_cm(cm, classes, title=None, file_path=None, style='simple'):

    if style == 'simple':
        fig = plt.figure(figsize=(12,10))
        ax = fig.subplots(1,1)
        ax.set_title('Confusion Matrix of'+title)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp = disp.plot(cmap=plt.cm.Blues, ax=ax)

    if style== 'with_axis':
        df_cm = pd.DataFrame(cm)
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'

        fig = plt.figure(figsize=(12,10))
        ax = fig.subplots(1,1)
        if title:
            ax.set_title('Confusion Matrix of'+title)

        res = sns.heatmap(df_cm, annot=True, square=True, cmap='Blues',
                         xticklabels = classes, yticklabels=classes, fmt='g', 
                         ax=ax, cbar_kws={'label': 'Number of Images'})

        res.axhline(y = 0, color = 'k', 
                    linewidth = 1)
        res.axhline(y = 16.98, color = 'k',
                    linewidth = 1)
        res.axvline(x = 0, color = 'k',
                    linewidth = 1)
        res.axvline(x = 16.98, color = 'k',
                    linewidth = 1)
    
    if file_path: 
        plt.savefig(file_path, dpi=300)
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
            
            