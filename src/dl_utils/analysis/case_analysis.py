import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from m3util.viz.layout import layout_fig
from dl_utils.utils.utils import get_random_batch_indices
from dl_utils.utils.utils import find_symm_index_in_hdf5, fetch_img_metadata

def prediction_vs_actual(model, dataloader, device, num_images=6):
    
    def ax_imshow(inp, ax, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        ax.imshow(inp)

    model.eval()
    model = model.to(device)
    
    fig = plt.figure(figsize=(10, num_images//3*3))

    with torch.no_grad():
        inputs, labels = next(iter(dataloader))
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
    class_names = dataloader.dataset.classes
    fig, axes = layout_fig(num_images, 3, figsize=(10, num_images//3*3), layout='tight')

    for i in range(num_images):
        r = np.random.randint(0, inputs.size()[0])
        # ax_imshow(inputs.cpu().data[r], axes[i]) 
        axes[i].imshow(inputs.cpu().data[r].permute(1, 2, 0))
        axes[i].axis('off')
        axes[i].set_title(f'predicted: {class_names[preds[r]]}, actual:{class_names[labels[r]]}')
        
    plt.show()


def most_confused_pairs(cm, classes, top_n=5):
    """
    Identifies the top misclassified class pairs from a confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix.
        dataset (Dataset): Dataset with class names.
        top_n (int): Number of most confused pairs to return.

    Returns:
        list of tuples: [(Most confused class 1, Most confused class 2, count), ...]
    """
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)  # Remove correct predictions from confusion matrix

    # Sort by highest misclassification
    misclassified_pairs = np.argsort(cm_no_diag.ravel())[::-1]  # Flatten and sort

    confused_list = []
    added_pairs = set()

    for idx in misclassified_pairs:
        row = idx // len(classes)  # True class
        col = idx % len(classes)   # Predicted class

        if row != col and (row, col) not in added_pairs:
            most_confused_class_1 = classes[row]
            most_confused_class_2 = classes[col]
            count = cm_no_diag[row, col]

            confused_list.append((most_confused_class_1, most_confused_class_2, count))
            added_pairs.add((row, col))  # Ensure uniqueness
            if len(confused_list) >= top_n:
                break

    if confused_list:
        strings = ["Top confused class pairs: \n"]
        strings = strings+[f"{i+1}. {pair[0]} → {pair[1]} ({pair[2]}); " for (i, pair) in enumerate(confused_list)]
        # print("\nTop confused class pairs:")
        # for i, pair in enumerate(confused_list):
        #     print(f"{i}. {pair[0]} -> {pair[1]} ({pair[2]} times)", end="; ")
        print("".join(strings))
    else:
        print("No misclassification found.")

    return confused_list


# def show_prediction_example(model, dataloader, t, p, classes, device, k=5, batch_limit=100, viz=True):
#     """
#     Finds and visualizes the most confused cases between two classes efficiently.
    
#     Args:
#         model (torch.nn.Module): Trained model.
#         batch (tuple): (data, labels) batch from dataloader.
#         t (str): True class name.
#         p (str): Predicted class name.
#         classes (list): List of class names.
#         device (str): Device ("cuda" or "cpu").
#         k (int): Number of top wrong predictions to display.
    
#     Returns:
#         None
#     """
#     model.eval()
#     model = model.to(device)
#     dataset = dataloader.dataset

#     # use custom data retrieval process to load data and metrics
#     for i_loop in range(len(dataloader)):
#         idx_list = get_random_batch_indices(dataset=dataset, batch_size=dataloader.batch_size)
#         batch = [dataset[idx] for idx in idx_list]  # Retrieve batch using the indices
#         data, labels = zip(*batch)  # Separate images and labels into two lists
        
        
#     # for i_loop, (data, labels) in enumerate(dataloader):
#     #     data, labels = data.to(device), labels.to(device)

#         # Convert class names to indices
#         t_n, p_n = classes.index(t), classes.index(p)

#         # Select only samples where the true label is `t_n`
#         mask = labels == t_n
#         data_t, labels_t = data[mask], labels[mask]

#         # Forward pass all filtered samples at once
#         with torch.no_grad():
#             outputs = model(data_t)
#             preds = torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1)

#         # Find all misclassified samples where prediction == `p_n`
#         misclassified_mask = preds == p_n
#         misclassified_data = data_t[misclassified_mask]

#         if len(misclassified_data) == 0:
#             # print(f"No misclassified samples found for {t} → {p}")
#             continue

#         # Randomly select one misclassified sample to visualize
#         i = np.random.randint(0, len(misclassified_data))
#         image = misclassified_data[i].cpu().numpy()

#         # Compute top-k wrong predictions
#         output_probs = torch.nn.functional.softmax(outputs[misclassified_mask], dim=1)
#         probs, top_classes = output_probs[i].topk(k)
#         # print(f"Top {k} incorrect predictions: " + ", ".join(f"{classes[cls_idx]}: {prob.item() * 100:.2f}%" for prob, cls_idx in zip(probs, top_classes)))
#         label = classes[labels_t[misclassified_mask][i]]
#         top_predictions = [classes[cls_idx] for cls_idx in top_classes]
        
#         if viz:
#             title = f"True: {t} | Predicted: {p}\n"
#             title += f", ".join(f"{classes[cls_idx]}: {prob.item() * 100:.2f}%" for prob, cls_idx in zip(probs, top_classes))

#             # Visualization
#             plt.figure(figsize=(5, 4))
#             plt.imshow(image[:3].transpose(1, 2, 0))  # Convert to (H, W, C)
#             plt.title(title)
#             plt.axis("off")
#             plt.show()
            
#         i_loop += 1
#         if batch_limit is not None and i_loop >= batch_limit:
#             break
#         # Print misclassification probabilities
#         return image, label, top_predictions, probs
    
#     raise ValueError(f"Batch limit of {batch_limit} reached without finding misclassified samples.")

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt

def generate_prediction_example(model, data_source, t, p, classes, device, k=5, batch_limit=100, group=None, viz=True):
    """
    Finds and visualizes the most confused cases between two classes efficiently.
    Supports both PyTorch dataloader and HDF5 dataset as input.

    Args:
        model (torch.nn.Module): Trained model.
        data_source (str or torch.utils.data.DataLoader): Path to HDF5 file or PyTorch dataloader.
        t (str): True class name.
        p (str): Predicted class name.
        classes (list): List of class names.
        device (str): Device ("cuda" or "cpu").
        k (int): Number of top wrong predictions to display.
        batch_limit (int): Max batches to process.
        group (str): Group name when using HDF5 dataset.
        viz (bool): Whether to visualize the misclassified sample.

    Returns:
        tuple: (image, label, top_predictions, probs) or raises ValueError if no misclassified sample is found.
    """
    model.eval()
    model = model.to(device)
    
    t_n, p_n = classes.index(t), classes.index(p)
    
    def process_sample(data, label):
        """Runs inference and checks if the sample is misclassified as `p`."""
        data = data.to(device).unsqueeze(0)  # Ensure batch dimension
        label = classes.index(label) if isinstance(label, str) else label
        
        with torch.no_grad():
            outputs = model(data)
            preds = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_classes = preds.topk(k)
        
        predicted_class = top_classes[0, 0].item()
        
        if predicted_class == p_n:
            image = data.cpu().numpy()[0]
            top_predictions = [classes[idx.item()] for idx in top_classes[0]]
            return image, label, top_predictions, top_probs[0]

        return None  # No misclassification found

    if isinstance(data_source, str):  # HDF5 file path
        if group is None:
            raise ValueError("Group name must be provided when loading data from HDF5.")

        with h5py.File(data_source, 'r') as h5:
            for _ in range(batch_limit):
                start_index = np.random.randint(0, len(h5[group]['labels']))
                index = find_symm_index_in_hdf5(h5, symm_str=t, group=group, index_start=start_index, index_end=None)
                img, label, label_str, ts, va, vb, VA, VB  = fetch_img_metadata(h5, group=group, index=index)

                data = torch.tensor(img / 255., dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
                result = process_sample(data, label_str)

                if result:
                    image, label, top_predictions, probs = result
                    image = image[:3].transpose(1, 2, 0)
                    break
            else:
                raise ValueError(f"No misclassified samples found after {batch_limit} attempts.")
            metadata = { 'ts': ts, 'va': va, 'vb': vb, 'VA': VA, 'VB': VB }



    else:  # PyTorch DataLoader
        dataloader = data_source
        
        for i_loop, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            
            # Filter samples with true label `t`
            mask = labels == t_n
            data_t, labels_t = data[mask], labels[mask]

            if len(data_t) == 0:
                continue  # No relevant samples in this batch

            # Forward pass all filtered samples at once
            with torch.no_grad():
                outputs = model(data_t)
                preds = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_classes = preds.topk(k)

            # Find misclassified samples (t → p)
            misclassified_mask = top_classes[:, 0] == p_n
            misclassified_data = data_t[misclassified_mask]
            misclassified_labels = labels_t[misclassified_mask]
            misclassified_top_probs = top_probs[misclassified_mask]
            misclassified_top_classes = top_classes[misclassified_mask]

            if len(misclassified_data) == 0:
                continue  # No misclassifications in this batch

            # Select a random misclassified sample
            i = np.random.randint(0, len(misclassified_data))
            image = misclassified_data[i].cpu().permute(1,2,0).numpy()
            label = classes[misclassified_labels[i].item()]
            top_predictions = [classes[idx.item()] for idx in misclassified_top_classes[i]]
            probs = misclassified_top_probs[i]
            break
        else:
            raise ValueError(f"No misclassified samples found after {batch_limit} batches.")
        metadata = {}
        
    # Visualization
    if viz:
        title = f"True: {t} | Predicted: {p}\n"
        title += ", ".join(f"{cls}: {prob.item() * 100:.2f}%" for cls, prob in zip(top_predictions, probs))
        plt.figure(figsize=(5, 4))
        plt.imshow(image)  # Convert (C, H, W) to (H, W, C)
        plt.title(title)
        plt.axis("off")
        plt.show()
    
    return image, label, top_predictions, probs, metadata

