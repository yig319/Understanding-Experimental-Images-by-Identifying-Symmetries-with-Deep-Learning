import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
import itertools

def compute_loss_landscape(
    model, 
    dataloader, 
    direction1,
    direction2,
    criterion=CrossEntropyLoss(), 
    grid_size=20, 
    use_batch=True, 
    subset_ratio=1.0, 
    device="cuda"
):
    """
    Computes the loss landscape of a deep learning model with tqdm tracking.

    Args:
        model (torch.nn.Module): The neural network model (e.g., ResNet50).
        dataloader (DataLoader): The dataset's DataLoader object.
        criterion (torch.nn.Module): The loss function (default: CrossEntropyLoss).
        direction1 (torch.Tensor): The first perturbation direction.
        direction2 (torch.Tensor): The second perturbation direction.
        # direction_method (str): The method to compute the perturbation directions (default: "random", options: "random", "hessian").
        grid_size (int): The resolution of the grid (e.g., 20x20).
        use_batch (bool): If True, uses only a single batch for loss computation.
        subset_ratio (float): Fraction of the dataset to use (1.0 = full dataset, 0.1 = 10%).
        device (str): Compute device ("cuda" or "cpu").

    Returns:
        tuple: (X, Y, loss_values) where X, Y are the meshgrid coordinates and loss_values is the computed loss matrix.
    """
    
    # Move model to correct device
    model = model.to(device)
    model.eval()

    # Select data source (full dataset or single batch)
    if use_batch:
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(device), targets.to(device)
    else:
        if subset_ratio < 1.0:
            dataset_size = int(len(dataloader.dataset) * subset_ratio)
            subset_indices = torch.randperm(len(dataloader.dataset))[:dataset_size]
            subset = Subset(dataloader.dataset, subset_indices)
            dataloader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)
    
    # Flatten model parameters
    reference_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    
    # # Generate two random perturbation directions
    # num_params = reference_params.size(0)
    # direction1, direction2 = torch.randn(num_params), torch.randn(num_params)
    
    # # Normalize directions
    # direction1 /= torch.norm(direction1)
    # direction2 /= torch.norm(direction2)

    # Define grid space
    alpha_range = np.linspace(-1, 1, grid_size)
    beta_range = np.linspace(-1, 1, grid_size)
    loss_values = np.zeros((grid_size, grid_size))

    # First-level progress bar (alpha & beta combined)
    for i, j in tqdm(itertools.product(range(grid_size), range(grid_size)), total=grid_size * grid_size, desc="Computing Loss Landscape"):

        alpha, beta = alpha_range[i], beta_range[j]

        # Perturb model parameters
        perturbed_params = reference_params.cpu() + alpha * direction1 + beta * direction2
        torch.nn.utils.vector_to_parameters(perturbed_params.to(device), model.parameters())

        if use_batch:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_values[i, j] = loss.item()
        else:
            total_loss, total_samples = 0.0, 0

            # Second-level tqdm for dataloader tracking
            for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                total_loss += loss.item() * batch_inputs.size(0)
                total_samples += batch_inputs.size(0)

            loss_values[i, j] = total_loss / total_samples

    # Restore original model parameters
    torch.nn.utils.vector_to_parameters(reference_params, model.parameters())

    # Create meshgrid for plotting
    X, Y = np.meshgrid(alpha_range, beta_range)

    return X, Y, loss_values


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button

# def visualize_loss_landscape_interactive(X, Y, loss_values, colormap="viridis"):
#     """
#     Interactive 3D visualization of a precomputed loss landscape.

#     Args:
#         X, Y (numpy.ndarray): Meshgrid coordinates.
#         loss_values (numpy.ndarray): Computed loss values.
#         colormap (str): Initial colormap to use (default: "viridis").
#     """

#     # Initialize figure
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")

#     # Initial plot
#     surf = ax.plot_surface(X, Y, loss_values, cmap=colormap, edgecolor="k", alpha=0.8)
#     contour = ax.contourf(X, Y, loss_values, zdir="z", offset=np.min(loss_values), cmap=colormap, alpha=0.7)

#     # Set labels and initial view
#     ax.set_xlabel("Direction 1 (alpha)")
#     ax.set_ylabel("Direction 2 (beta)")
#     ax.set_zlabel("Loss")
#     ax.set_title("Interactive Loss Landscape")
#     ax.view_init(elev=30, azim=45)

#     # Define interactive sliders for elevation and azimuth
#     axcolor = "lightgoldenrodyellow"
#     ax_elev = plt.axes([0.15, 0.02, 0.3, 0.03], facecolor=axcolor)
#     ax_azim = plt.axes([0.55, 0.02, 0.3, 0.03], facecolor=axcolor)

#     slider_elev = Slider(ax_elev, "Elevation", 0, 90, valinit=30)
#     slider_azim = Slider(ax_azim, "Azimuth", 0, 360, valinit=45)

#     # Function to update the view angle
#     def update(val):
#         ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
#         fig.canvas.draw_idle()

#     slider_elev.on_changed(update)
#     slider_azim.on_changed(update)

#     # Dropdown menu to change colormap dynamically
#     cmap_options = ["viridis", "coolwarm", "plasma", "inferno", "cividis"]
#     ax_button = plt.axes([0.85, 0.85, 0.1, 0.05])
#     button = Button(ax_button, "Change Colormap", color="lightgray")

#     def change_colormap(event):
#         new_cmap = cmap_options.pop(0)  # Rotate colormap
#         cmap_options.append(new_cmap)
#         surf.set_cmap(new_cmap)
#         for c in contour.collections:
#             c.set_cmap(new_cmap)
#         fig.canvas.draw_idle()

#     button.on_clicked(change_colormap)

#     # Activate interactive mode
#     plt.ion()
#     plt.show()
import plotly.graph_objects as go
import numpy as np

def visualize_loss_landscape_interactive_plotly(X, Y, loss_values, colormap="Viridis"):
    """
    Interactive 3D visualization of a precomputed loss landscape using Plotly.

    Args:
        X, Y (numpy.ndarray): Meshgrid coordinates.
        loss_values (numpy.ndarray): Computed loss values.
        colormap (str): Initial colormap to use (default: "Viridis").
    """

    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(z=loss_values, x=X, y=Y, colorscale=colormap)])

    # Update layout for better visualization
    fig.update_layout(
        title="Interactive Loss Landscape",
        scene=dict(
            xaxis_title="Direction 1 (alpha)",
            yaxis_title="Direction 2 (beta)",
            zaxis_title="Loss",
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),  # Initial view angle
    ),
        width=800,  # Adjust width (default ~700)
        height=600,  # Adjust height (default ~450)
        
        updatemenus=[
            # Dropdown for colormap selection
            dict(
                buttons=[
                    dict(
                        args=[{"colorscale": c}],
                        label=c.capitalize(),
                        method="restyle"
                    ) for c in ["Viridis", "Coolwarm", "Plasma", "Inferno", "Cividis"]
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.17,
                xanchor="left",
                y=1.15,
                yanchor="top",
            ),
            # Reset view button
            dict(
                buttons=[
                    dict(
                        args=[dict(camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)))],
                        label="Reset View",
                        method="relayout",
                    )
                ],
                type="buttons",
                direction="right",
                x=0.6,
                xanchor="left",
                y=1.15,
                yanchor="top",
            ),
        ],
    )

    # Show interactive plot
    fig.show()


def visualize_loss_landscape(
    X, Y, loss_values, elev=30, azim=45, colormap="coolwarm", show_contour=True
):
    """
    Visualizes a precomputed loss landscape.

    Args:
        X, Y (numpy.ndarray): Meshgrid coordinates.
        loss_values (numpy.ndarray): Computed loss values.
        elev (int): Elevation angle for 3D plot.
        azim (int): Azimuth angle for 3D plot.
        colormap (str): Colormap to use.
        show_contour (bool): Whether to overlay contour plot.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 3D Surface plot
    ax.plot_surface(X, Y, loss_values, cmap=colormap, edgecolor="k", alpha=0.8)

    # Contour plot at base for better visibility
    if show_contour:
        ax.contourf(X, Y, loss_values, zdir="z", offset=np.min(loss_values), cmap=colormap, alpha=0.7)

    # Set view angles
    ax.view_init(elev=elev, azim=azim)

    # Labels
    ax.set_xlabel("Direction 1 (alpha)")
    ax.set_ylabel("Direction 2 (beta)")
    ax.set_zlabel("Loss")
    ax.set_title("Loss Landscape Visualization")

    # Show plot
    plt.show()



# def compute_loss_landscape(
#     model, 
#     dataloader, 
#     criterion=CrossEntropyLoss(), 
#     grid_size=20, 
#     use_batch=True, 
#     subset_ratio=1.0, 
#     device="cuda"
# ):
#     """
#     Computes and visualizes the loss landscape of a deep learning model.

#     Args:
#         model (torch.nn.Module): The neural network model (e.g., ResNet50).
#         dataloader (DataLoader): The dataset's DataLoader object.
#         criterion (torch.nn.Module): The loss function (default: CrossEntropyLoss).
#         grid_size (int): The resolution of the grid (e.g., 20x20).
#         use_batch (bool): If True, uses only a single batch for loss computation.
#         subset_ratio (float): Fraction of the dataset to use (1.0 = full dataset, 0.1 = 10%).
#         device (str): Compute device ("cuda" or "cpu").

#     Returns:
#         None: Displays a 3D loss landscape plot.
#     """
    
#     # Move model to correct device
#     model = model.to(device)
#     model.eval()

#     # Select data source (full dataset or single batch)
#     if use_batch:
#         inputs, targets = next(iter(dataloader))
#         inputs, targets = inputs.to(device), targets.to(device)
#     else:
#         # Reduce dataset size if subset_ratio < 1.0
#         if subset_ratio < 1.0:
#             dataset_size = int(len(dataloader.dataset) * subset_ratio)
#             subset_indices = torch.randperm(len(dataloader.dataset))[:dataset_size]
#             subset = Subset(dataloader.dataset, subset_indices)
#             dataloader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)
    
#     # Flatten model parameters
#     reference_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    
#     # Generate two random perturbation directions
#     num_params = reference_params.size(0)
#     direction1, direction2 = torch.randn(num_params), torch.randn(num_params)
    
#     # Normalize directions
#     direction1 /= torch.norm(direction1)
#     direction2 /= torch.norm(direction2)

#     # Define grid space
#     alpha_range = np.linspace(-1, 1, grid_size)
#     beta_range = np.linspace(-1, 1, grid_size)
#     loss_values = np.zeros((grid_size, grid_size))

#     # Compute loss for each grid point
#     for i, alpha in enumerate(tqdm(alpha_range, desc="Computing Loss Landscape", leave=True)):
#         for j, beta in enumerate(beta_range):
            
#             perturbed_params = reference_params.cpu() + alpha * direction1 + beta * direction2
#             torch.nn.utils.vector_to_parameters(perturbed_params.to(device), model.parameters())
#             # model = model.to(device)
            
#             if use_batch:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 loss_values[i, j] = loss.item()
#             else:
#                 total_loss, total_samples = 0.0, 0
#                 for batch_inputs, batch_targets in dataloader:
#                     batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
#                     outputs = model(batch_inputs)
#                     loss = criterion(outputs, batch_targets)
#                     total_loss += loss.item() * batch_inputs.size(0)
#                     total_samples += batch_inputs.size(0)
#                 loss_values[i, j] = total_loss / total_samples

#     # Restore original model parameters
#     torch.nn.utils.vector_to_parameters(reference_params, model.parameters())

#     # Plot the 3D loss landscape
#     X, Y = np.meshgrid(alpha_range, beta_range)
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_surface(X, Y, loss_values, cmap="viridis")
#     ax.set_xlabel("Direction 1 (alpha)")
#     ax.set_ylabel("Direction 2 (beta)")
#     ax.set_zlabel("Loss")
#     ax.set_title("Loss Landscape Visualization")
#     plt.show()
