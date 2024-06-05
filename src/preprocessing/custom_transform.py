import torch
import random
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image

class ZoomTransform:
    def __init__(self, zoom_range=(1, 1.5), output_size=(256, 256)):
        self.zoom_range = zoom_range
        self.output_size = output_size
        # self.start_point = start_point

    def __call__(self, img, vectors=None):
        h, w, _ = img.shape        
        zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
                
        # Calculate new size
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        
        # Resize image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = F.resize(img, (new_h, new_w))
        
        left = 0
        upper = 0
        right = left + self.output_size[0]
        lower = upper + self.output_size[1]
        
        # Crop image to the desired output size
        img = img.crop((left, upper, right, lower))
        
        if not isinstance(vectors, type(None)):
            # print(vectors, zoom_factor)
            ts, va, vb = vectors
            ts, va, vb = ts * zoom_factor, va * zoom_factor, vb * zoom_factor
            return img, (ts, va, vb)
        return img


# class ZoomTransform:
#     def __init__(self, zoom_range=(0.5, 1.5), output_size=(256, 256)):
#         self.zoom_range = zoom_range
#         self.output_size = output_size

#     def __call__(self, img, vectors=None):
#         if isinstance(img, np.ndarray):
#             img = Image.fromarray(img)
        
#         # Ensure image is a PIL Image
#         img = F.to_pil_image(img) if isinstance(img, torch.Tensor) else img

#         # Get image dimensions
#         w, h = img.size
#         zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
                
#         # Calculate new size
#         new_w = int(w * zoom_factor)
#         new_h = int(h * zoom_factor)
        
#         # Resize image
#         img = F.resize(img, (new_h, new_w))
        
#         # Determine cropping box
#         left = 0
#         upper = 0
#         right = min(left + self.output_size[0], new_w)
#         lower = min(upper + self.output_size[1], new_h)
        
#         # Crop image to the desired output size
#         img = img.crop((left, upper, right, lower))
        
#         # If the image is smaller than the output size, pad it
#         pad_width = self.output_size[0] - img.size[0]
#         pad_height = self.output_size[1] - img.size[1]
        
#         if pad_width > 0 or pad_height > 0:
#             img = self.pad_image(img, pad_width, pad_height)
        
#         # Convert image back to tensor
#         img = F.to_tensor(img)
        
#         # Apply the same zoom factor to the vectors if provided
#         if vectors is not None:
#             ts, va, vb = vectors
#             ts, va, vb = ts * zoom_factor, va * zoom_factor, vb * zoom_factor
#             return img, (ts, va, vb)
        
#         return img, vectors
    
#     def pad_image(self, img, pad_width, pad_height):
#         # Convert back to PIL image for processing
#         img = F.to_pil_image(img) if isinstance(img, torch.Tensor) else img
        
#         # Get the dimensions of the cropped image
#         img_w, img_h = img.size
        
#         # Create an empty image with the desired output size
#         new_img = Image.new("RGB", self.output_size)
        
#         # Paste the cropped image onto the top-left corner of the new image
#         new_img.paste(img, (0, 0))
        
#         # Tile the cropped region to fill the rest of the new image
#         for i in range(0, self.output_size[0], img_w):
#             for j in range(0, self.output_size[1], img_h):
#                 if i == 0 and j == 0:
#                     continue  # Skip the already pasted region
#                 new_img.paste(img, (i, j))
        
#         return new_img
