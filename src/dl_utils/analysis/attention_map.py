# import modules
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from PIL import Image
import torch
from torchcam.methods import GradCAM
from dl_utils.utils.utils import NormalizeData
from m3util.viz.layout import layout_fig

class AttentionMapVisualizer:
    def __init__(self, colormap="viridis", alpha=0.5, device=torch.device("cpu")):
        """
        Initializes the AttentionMapVisualizer.

        Args:
            model (torch.nn.Module): The model for which attention maps will be generated.
            colormap (str): Colormap to use for the attention map (default: "viridis").
            alpha (float): Transparency for the overlay (default: 0.5).
        """
        self.colormap = plt.get_cmap(colormap)
        self.alpha = alpha
        self.device = device
            
    def generate_cnn_attention_map(self, model, input_tensor, layer_name="layer4"):
        """
        Generates and visualizes an attention map for a CNN model using Grad-CAM.

        Args:
            model (torch.nn.Module): The model for which attention maps will be generated.
            input_tensor (torch.Tensor): Input tensor with shape (1, C, H, W).
            layer_name (str): The name of the target layer for Grad-CAM.

        Returns:
            None: Displays the attention map.
        """

        # Ensure the model is in evaluation mode
        model.eval()
        model.to(self.device)
        input_tensor = input_tensor.to(self.device)

        # Initialize Grad-CAM extractor for the specified layer
        cam_extractor = GradCAM(model, layer_name)

        # Forward pass with gradients enabled
        input_tensor.requires_grad = True  # Ensure gradients are tracked for the input
        output = model(input_tensor)

        # Extract the class index with the highest prediction score
        pred_class = output.squeeze(0).argmax().item()

        # Compute the attention map for the predicted class
        attention_map = cam_extractor(pred_class, output)

        # Release Grad-CAM hooks
        cam_extractor.remove_hooks()

        # Process the attention map for visualization
        attention_map = attention_map[0].squeeze().cpu().numpy()  # Ensure it's on the CPU

        # Resize the attention map to match the input image resolution
        input_image_np, attention_map_resized = self.process_attention_map(input_tensor, attention_map)
        
        return input_image_np, attention_map_resized
    
        # # Pass the resized attention map to visualization
        # self._visualize_attention_map(input_tensor, attention_map)


    def generate_transformer_attention_map(self, model, input_tensor, attention_layer_idx=-1):
        """
        Generates and visualizes attention maps for a transformer model using token-feature embeddings.

        Args:
            model (torch.nn.Module): The transformer model.
            input_tensor (torch.Tensor): Input tensor with shape (1, C, H, W).
            attention_layer_idx (int): Index of the attention layer to visualize.

        Returns:
            None: Displays the attention map.
        """
        # Ensure the model is in evaluation mode
        model.eval()
        model.to(self.device)
        input_tensor = input_tensor.to(self.device)
        
        # Hook to capture attention weights
        attention_weights = []

        def hook_fn(module, input, output):
            attention_weights.append(output)

        # Register the hook for the specified attention layer
        layer_to_hook = model.blocks[attention_layer_idx].attn
        hook_handle = layer_to_hook.register_forward_hook(hook_fn)

        # Forward pass through the model
        with torch.no_grad():
            model(input_tensor)

        # Remove the hook after the forward pass
        hook_handle.remove()

        # Check the captured attention weights
        if len(attention_weights) == 0:
            raise ValueError("No attention weights were captured. Check the attention layer indexing.")

        attn_weights = attention_weights[0]  # Shape: (1, 1024, 384)

        # Reshape the attention weights into a spatial map
        tokens = attn_weights.shape[1]  # 1024 tokens
        grid_size = int(tokens**0.5)  # Assume square grid (e.g., 32x32 if 1024 tokens)
        if tokens != grid_size**2:
            raise ValueError(
                f"Number of tokens ({tokens}) cannot be reshaped into a square grid. "
                "Check the model's configuration."
            )

        # Aggregate attention across the embedding dimension (e.g., mean or sum)
        attention_map = attn_weights.mean(dim=-1).squeeze(0).cpu().numpy()  # Shape: (1024,)
        attention_map = attention_map.reshape(grid_size, grid_size)  # Reshape to (Grid, Grid)

        # Resize the attention map to match the input image resolution
        input_image_np, attention_map_resized = self.process_attention_map(input_tensor, attention_map)
        
        return input_image_np, attention_map_resized
        # # Pass the resized attention map to visualization
        # self._visualize_attention_map(input_tensor, attention_map)
  
  
    def process_attention_map(self, input_tensor, attention_map):
        """
        Processes the attention map for visualization.
        
        Args:
            input_tensor (torch.Tensor): The input image tensor.
            attention_map (np.ndarray): The attention map to overlay.
        """
        
        # Normalize the attention map
        input_image_np = np.asarray(
            (input_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        )
        
        if np.all(attention_map == 0):
            print("Warning: Attention map is all zeros!")
            # Resize the attention map to match the input image resolution
            attention_map_resized = np.full_like(input_image_np[:,:,0], 0.)
        
        else:
            attention_map = NormalizeData(attention_map)  # Normalize the attention map
            # Resize the attention map to match the input image resolution
            attention_map_resized = np.array(
                Image.fromarray(attention_map).resize(input_image_np.shape[:2], Image.BILINEAR)
            )
            
        return input_image_np, attention_map_resized
    
    
    def visualize_attention_map(self, input_image_np, attention_map_resized, keyword, fig=None, axes=None, title=None):
        """
        Visualizes the attention map overlaid on the input image.

        Args:
            input_image_np (np.ndarray): The input image as a NumPy array.
            attention_map_resized (np.ndarray): The resized attention map.

        Returns:
            None: Displays the attention map.
        """

        # Apply the colormap
        cmap = plt.get_cmap(self.colormap)
        colored_attention_map = cmap(attention_map_resized)[:, :, :3] # Keep RGB only

        # Create overlay manually
        overlay = (self.alpha * input_image_np + (1 - self.alpha) * (colored_attention_map * 255)).astype("uint8")

        if axes is None:
            fig, axes = layout_fig(3, 3, figsize=(8, 2.5), layout='tight')

        imgs = [input_image_np, attention_map_resized, overlay]
        titles = ["Original Image", f"Attention Map ({keyword})", "Overlayed Attention Map"]
        for ax, img, title in zip(axes, imgs, titles):
            ax.axis("off")
            im = ax.imshow(img)
            ax.set_title(title)
            
            if title == f"Attention Map ({keyword})":
                # Add colorbar without resizing the plot
                divider = make_axes_locatable(axes[1])
                cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and pad as needed
                fig.colorbar(im, cax=cax)
        
        if axes is None:
            if title:
                plt.suptitle(title)
            plt.show()