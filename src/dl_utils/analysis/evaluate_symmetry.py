import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from m3util.viz.layout import layout_fig
from dl_utils.utils.viz import verify_image_vector


def find_all_regions(img, ts, VA, VB, viz=False):
    """
    Finds all parallelogram-shaped regions in the image defined by the starting point `ts` and vectors `VA` and `VB`.

    Parameters:
        img (ndarray): Input image.
        ts (tuple): Starting point (y, x).
        VA (tuple): First translation vector (dy1, dx1).
        VB (tuple): Second translation vector (dy2, dx2).
        viz (bool): Visualize the cropped regions.

    Returns:
        regions (list): List of cropped image patches.
        masks (list): List of binary masks for each region.
    """
    regions = []
    masks = []

    # Find all valid starting points
    valid_ts_list = find_valid_starting_points(img, ts, VA, VB)

    # Crop the parallelogram-shaped regions
    for valid_ts in valid_ts_list:
        region, mask = crop_image_with_mask(img, valid_ts, VA, VB)
        regions.append(region)
        masks.append(mask)
        
    if viz:
        # fig, axes = layout_fig(len(valid_ts_list), 4, figsize=(8, 2*(len(valid_ts_list)//5+1)))
        # for ax, ts in zip(axes, valid_ts_list):
        #     verify_image_vector(ax=ax, image=img, ts=ts, va=VA, vb=VB, shade_alpha=0.3, shade_color='white')
        # plt.show()
        
        fig, axes = layout_fig(len(valid_ts_list), 4, figsize=(8, 2*(len(valid_ts_list)//5+1)))
        for ax, ts, cropped_img in zip(axes, valid_ts_list, regions):
            ax.imshow(cropped_img)
        plt.suptitle('All regions')
        plt.show()

    return regions, masks, valid_ts_list


def find_valid_starting_points(img, ts, VA, VB):
    """
    Finds all valid starting points `ts` where all four corners 
    (ts, ts+VA, ts+VB, ts+VA+VB) remain inside the image bounds.

    Parameters:
        img_shape (tuple): Shape of the full image (height, width).
        ts (tuple): Initial starting point (y, x).
        VA (tuple): First translation vector (dy1, dx1).
        VB (tuple): Second translation vector (dy2, dx2).

    Returns:
        valid_ts_list (list): List of valid (y, x) starting points.
    """
    img_h, img_w = img.shape[:2]
    valid_ts_list = set()  # Use a set to store unique valid points

    # Directions to iterate (VA, VB, -VA, -VB, VA-VB, VB-VA)
    directions = [VA, VB, (-VA[0], -VA[1]), (-VB[0], -VB[1]), 
                  (VA[0] - VB[0], VA[1] - VB[1]), (VB[0] - VA[0], VB[1] - VA[1])]

    # Try placing patches at different `ts` by iterating over all directions
    for i in range(-img_h // abs(VA[0]) if VA[0] != 0 else 0, img_h // abs(VA[0]) if VA[0] != 0 else 1):
        for j in range(-img_w // abs(VB[1]) if VB[1] != 0 else 0, img_w // abs(VB[1]) if VB[1] != 0 else 1):
            new_ts = (ts[0] + i * VA[0] + j * VB[0], ts[1] + i * VA[1] + j * VB[1])

            # Compute all four corners
            p1 = new_ts
            p2 = (new_ts[0] + VA[0], new_ts[1] + VA[1])
            p3 = (new_ts[0] + VB[0], new_ts[1] + VB[1])
            p4 = (new_ts[0] + VA[0] + VB[0], new_ts[1] + VA[1] + VB[1])

            # Ensure all four corners are inside the image
            if all(0 <= p[0] < img_h and 0 <= p[1] < img_w for p in [p1, p2, p3, p4]):
                valid_ts_list.add(new_ts)
                
    return sorted(valid_ts_list)  # Return sorted list for consistency


def crop_image_with_mask(img, ts, VA, VB):
    """
    Crops the parallelogram-shaped region from the image and generates a corresponding binary mask.
    
    Parameters:
        img (ndarray): Input image.
        ts (tuple): Starting point (y, x).
        VA (tuple): First translation vector (dy1, dx1).
        VB (tuple): Second translation vector (dy2, dx2).

    Returns:
        cropped_img (ndarray): Cropped image with outside pixels set to 0.
        mask (ndarray): Binary mask of the same size as the cropped image.
    """
    # Compute all four corners of the parallelogram (y, x)
    offset = 1
    p1 = np.array(ts, dtype=np.int32) + offset
    p2 = np.array(ts) + np.array(VA, dtype=np.int32) + offset
    p3 = np.array(ts) + np.array(VB, dtype=np.int32) + offset
    p4 = np.array(ts) + np.array(VA, dtype=np.int32) + np.array(VB, dtype=np.int32) + offset

    # Get bounding box coordinates
    ymin, ymax = min(p1[0], p2[0], p3[0], p4[0]), max(p1[0], p2[0], p3[0], p4[0])
    xmin, xmax = min(p1[1], p2[1], p3[1], p4[1]), max(p1[1], p2[1], p3[1], p4[1])

    # Ensure bounds are within image limits
    img_h, img_w = img.shape[:2]
    ymin, ymax = max(0, ymin), min(img_h, ymax)
    xmin, xmax = max(0, xmin), min(img_w, xmax)
    
    # Crop image
    cropped_img = img[ymin:ymax, xmin:xmax].copy()
    
    # Create a blank mask for the cropped region
    mask = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint8)

    # Correct the polygon relative to the cropped region (using x, y format for OpenCV)
    polygon = np.array([
        [p1[1] - xmin, p1[0] - ymin],  # (x, y)
        [p2[1] - xmin, p2[0] - ymin],
        [p4[1] - xmin, p4[0] - ymin],
        [p3[1] - xmin, p3[0] - ymin]
    ], dtype=np.int32)

    # Ensure the shape is in the right format for OpenCV
    polygon = polygon.reshape((-1, 1, 2))

    # Fill the parallelogram with 1s
    cv2.fillPoly(mask, [polygon], 1)

    # Apply mask to the cropped image (set pixels outside the shape to 0)
    if cropped_img.ndim == 3:  # For color images
        cropped_img = cropped_img * mask[:, :, None]
    else:  # For grayscale images
        cropped_img = cropped_img * mask

    return cropped_img, mask



def apply_canny_edge_filter(patch, low_threshold=50, high_threshold=150):
    """Applies Canny edge detection to enhance edges."""
    patch = np.array(patch)

    # Convert to grayscale if RGB
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # Ensure uint8 format (0-255)
    if patch.dtype != np.uint8:
        patch = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Canny edge detection
    edge_patch = cv2.Canny(patch, low_threshold, high_threshold)
    return edge_patch

def apply_canny_edge_filter(patch, low_threshold=50, high_threshold=150):
    """Applies Canny edge detection to enhance edges."""
    patch = np.array(patch)

    # Convert to grayscale if RGB
    if patch.ndim == 3 and patch.shape[-1] == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # Ensure uint8 format (0-255)
    if patch.dtype != np.uint8:
        patch = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Canny edge detection
    edge_patch = cv2.Canny(patch, low_threshold, high_threshold)
    return edge_patch

def apply_highpass_filter(patch):
    """Applies a Laplacian high-pass filter to enhance details, ensuring correct input format."""
    patch = np.array(patch)

    # Convert to grayscale if RGB
    if patch.ndim == 3 and patch.shape[-1] == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    # Ensure uint8 format (0-255)
    if patch.dtype != np.uint8:
        patch = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Laplacian high-pass filter
    filtered_patch = cv2.Laplacian(patch, cv2.CV_64F)
    
    return filtered_patch

def calculate_ncc(patch1, patch2):
    """Computes Normalized Cross-Correlation (NCC) between two patches."""
    patch1 = patch1.astype(np.float32)
    patch2 = patch2.astype(np.float32)

    # Compute cross-correlation
    result = cv2.matchTemplate(patch1, patch2, method=cv2.TM_CCOEFF_NORMED)

    # Find max correlation score
    max_ncc = np.max(result)
    return max_ncc

def calculate_shift_tolerant_ssim(patches, max_shift=2, highpass_filter=False, edge_filter=False, use_ncc=False, weight_ssim=0.6, weight_ncc=0.4):
    """
    Computes SSIM similarity matrix with shift tolerance, applying optional high-pass or edge detection filters.

    Parameters:
        patches (list): List of cropped image patches (grayscale or RGB).
        max_shift (int): Maximum pixel shift to consider in both directions.
        highpass_filter (bool): Apply high-pass filtering before SSIM computation.
        edge_filter (bool): Apply Canny edge detection before SSIM computation.
        use_ncc (bool): Use NCC in addition to SSIM.
        weight_ssim (float): Weight for SSIM.
        weight_ncc (float): Weight for NCC.

    Returns:
        similarity_matrix (ndarray): Combined SSIM + NCC similarity matrix.
        
    Note:
        max_shift=2, highpass_filter=True, works well for translation invariance evaluation of attention map.
    """
    num_patches = len(patches)
    similarity_matrix = np.zeros((num_patches, num_patches))

    # Apply filters if enabled
    if highpass_filter:
        patches = [apply_highpass_filter(patch) for patch in patches]
    if edge_filter:
        patches = [apply_canny_edge_filter(patch) for patch in patches]

    for i in range(num_patches):
        for j in range(i, num_patches):
            best_ssim = -1
            best_ncc = -1 if use_ncc else 0  # NCC is optional

            if max_shift == 0:
                # Compute SSIM without any shifts
                best_ssim = ssim(
                    patches[i], patches[j], 
                    data_range=255, 
                    win_size=min(patches[i].shape[:2]),  # Ensure win_size is valid
                    channel_axis=-1 if patches[i].ndim == 3 else None  # Handle RGB properly
                )
                if use_ncc:
                    best_ncc = calculate_ncc(patches[i], patches[j])
            else:
                # Try small shifts within the given tolerance
                for dy in range(-max_shift, max_shift + 1):
                    for dx in range(-max_shift, max_shift + 1):
                        # Shift patch j by (dx, dy) and compute SSIM
                        shifted_patch = np.roll(patches[j], shift=(dy, dx), axis=(0, 1))
                        
                        # Determine the smallest dimension of the patch
                        min_dim = min(patches[i].shape[:2])

                        # Ensure win_size is valid: it must be odd and <= min_dim
                        win_size = min(7, min_dim) if min_dim < 7 else (min_dim if min_dim % 2 == 1 else min_dim - 1)

                        # Compute SSIM with the corrected window size
                        ssim_score = ssim(
                            patches[i], shifted_patch, 
                            data_range=255, 
                            win_size=win_size, 
                            channel_axis=-1 if patches[i].ndim == 3 else None  # Handle RGB properly
                        )


                        # ssim_score = ssim(
                        #     patches[i], shifted_patch, 
                        #     data_range=255, 
                        #     win_size=min(patches[i].shape[:2]),  # Ensure win_size is valid
                        #     channel_axis=-1 if patches[i].ndim == 3 else None  # Handle RGB properly
                        # )
                        
                        # Keep the maximum SSIM found
                        best_ssim = max(best_ssim, ssim_score)

                        # Compute NCC if enabled
                        if use_ncc:
                            ncc_score = calculate_ncc(patches[i], shifted_patch)
                            best_ncc = max(best_ncc, ncc_score)

            # Compute final similarity score (SSIM + NCC weighted)
            if use_ncc:
                similarity_matrix[i, j] = weight_ssim * best_ssim + weight_ncc * best_ncc
                similarity_matrix[j, i] = similarity_matrix[i, j]
            else:
                similarity_matrix[i, j] = best_ssim
                similarity_matrix[j, i] = best_ssim

    return similarity_matrix


def rot_4f_eval(regions, viz=False):
    similarity_matrix_list = []
    for region in regions:
        patterns = [region, np.rot90(region, 1), np.rot90(region, 2), np.rot90(region, 3)]
        
        similarity_matrix = calculate_shift_tolerant_ssim(patterns, max_shift=2, highpass_filter=True)
        similarity_matrix_list.append(np.mean(similarity_matrix))
        
        if viz:
            fig, axes = layout_fig(4, 4, figsize=(8, 2))
            for i, ax in enumerate(axes):
                ax.imshow(patterns[i])
            plt.suptitle('Rotated 4-fold symmetry, SSIM: {:.2f}'.format(np.mean(similarity_matrix)))
            plt.show()
        
    return similarity_matrix_list

def rot_2f_eval(regions, viz=False):
    similarity_matrix_list = []
    for region in regions:
        patterns = [region, np.rot90(region, 2)]
                
        similarity_matrix = calculate_shift_tolerant_ssim(patterns, max_shift=2, highpass_filter=True)
        similarity_matrix_list.append(np.mean(similarity_matrix))
        
        if viz:
            fig, axes = layout_fig(2, 2, figsize=(8, 2))
            for i, ax in enumerate(axes):
                ax.imshow(patterns[i])
            plt.suptitle('Rotated 2-fold symmetry, SSIM: {:.2f}'.format(np.mean(similarity_matrix)))
            plt.show()

    return similarity_matrix_list


def mirror_eval(regions, viz=False):
    similarity_matrix_list = []
    for region in regions:
        patterns = [region, np.flip(region, axis=0)]
        
        similarity_matrix = calculate_shift_tolerant_ssim(patterns, max_shift=2, highpass_filter=True)
        similarity_matrix_list.append(np.mean(similarity_matrix))
        
        if viz:
            fig, axes = layout_fig(2, 2, figsize=(8, 2))
            for i, ax in enumerate(axes):
                ax.imshow(patterns[i])
            plt.suptitle('Mirror symmetry, SSIM: {:.2f}'.format(np.mean(similarity_matrix)))
            plt.show()
        
    return similarity_matrix_list