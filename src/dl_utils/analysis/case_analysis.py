import torch
import numpy as np
import matplotlib.pyplot as plt
from m3util.viz.layout import layout_fig
from dl_utils.utils.utils import get_random_batch_indices


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


def show_prediction_example(model, dataloader, t, p, classes, device, k=5, batch_limit=100, viz=True):
    """
    Finds and visualizes the most confused cases between two classes efficiently.
    
    Args:
        model (torch.nn.Module): Trained model.
        batch (tuple): (data, labels) batch from dataloader.
        t (str): True class name.
        p (str): Predicted class name.
        classes (list): List of class names.
        device (str): Device ("cuda" or "cpu").
        k (int): Number of top wrong predictions to display.
    
    Returns:
        None
    """
    model.eval()
    model = model.to(device)
    dataset = dataloader.dataset

    # use custom data retrieval process to load data and metrics
    for i_loop in range(len(dataloader)):
        idx_list = get_random_batch_indices(dataset=dataset, batch_size=dataloader.batch_size)
        batch = [dataset[idx] for idx in idx_list]  # Retrieve batch using the indices
        data, labels = zip(*batch)  # Separate images and labels into two lists
        
        
    # for i_loop, (data, labels) in enumerate(dataloader):
    #     data, labels = data.to(device), labels.to(device)

        # Convert class names to indices
        t_n, p_n = classes.index(t), classes.index(p)

        # Select only samples where the true label is `t_n`
        mask = labels == t_n
        data_t, labels_t = data[mask], labels[mask]

        # Forward pass all filtered samples at once
        with torch.no_grad():
            outputs = model(data_t)
            preds = torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1)

        # Find all misclassified samples where prediction == `p_n`
        misclassified_mask = preds == p_n
        misclassified_data = data_t[misclassified_mask]

        if len(misclassified_data) == 0:
            # print(f"No misclassified samples found for {t} → {p}")
            continue

        # Randomly select one misclassified sample to visualize
        i = np.random.randint(0, len(misclassified_data))
        image = misclassified_data[i].cpu().numpy()

        # Compute top-k wrong predictions
        output_probs = torch.nn.functional.softmax(outputs[misclassified_mask], dim=1)
        probs, top_classes = output_probs[i].topk(k)
        # print(f"Top {k} incorrect predictions: " + ", ".join(f"{classes[cls_idx]}: {prob.item() * 100:.2f}%" for prob, cls_idx in zip(probs, top_classes)))
        label = classes[labels_t[misclassified_mask][i]]
        top_predictions = [classes[cls_idx] for cls_idx in top_classes]
        
        if viz:
            title = f"True: {t} | Predicted: {p}\n"
            title += f", ".join(f"{classes[cls_idx]}: {prob.item() * 100:.2f}%" for prob, cls_idx in zip(probs, top_classes))

            # Visualization
            plt.figure(figsize=(5, 4))
            plt.imshow(image[:3].transpose(1, 2, 0))  # Convert to (H, W, C)
            plt.title(title)
            plt.axis("off")
            plt.show()
            
        i_loop += 1
        if batch_limit is not None and i_loop >= batch_limit:
            break
        # Print misclassification probabilities
        return image, label, top_predictions, probs
    
    raise ValueError(f"Batch limit of {batch_limit} reached without finding misclassified samples.")