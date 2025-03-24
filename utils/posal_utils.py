"""Utility functions for POSAL model weight conversion and preprocessing"""

import numpy as np
import torch
import h5py
import cv2
from skimage import morphology

def load_keras_weights_to_pytorch(pytorch_model, keras_weights_path):
    """
    Utility to load weights from the original Keras implementation
    to our PyTorch model if needed.
    """
    try:
        with h5py.File(keras_weights_path, 'r') as f:
            for name, param in pytorch_model.named_parameters():
                # Map PyTorch parameter names to Keras names
                keras_name = name.replace('.', '/')
                
                if keras_name in f:
                    weight = torch.from_numpy(np.array(f[keras_name]))
                    
                    # Handle shape differences
                    if len(weight.shape) == 4:  # Conv weights
                        weight = weight.permute(3, 2, 0, 1)
                    elif len(weight.shape) == 2:  # Dense weights
                        weight = weight.t()
                    
                    param.data.copy_(weight)
    except Exception as e:
        print(f"Error loading Keras weights: {e}")
    
    return pytorch_model

def get_largest_fillhole(binary):
    """Get largest connected component and fill holes"""
    from skimage import measure
    import scipy.ndimage as ndi
    
    # Label connected components
    label_img = measure.label(binary)
    regions = measure.regionprops(label_img)
    
    if not regions:
        return binary
    
    # Find the largest connected component
    areas = [r.area for r in regions]
    max_idx = np.argmax(areas)
    
    largest_cc = np.zeros_like(binary)
    largest_cc[label_img == (max_idx + 1)] = 1
    
    # Fill holes
    return ndi.binary_fill_holes(largest_cc).astype(np.uint8)

def process_output_mask(prob_map, threshold=0.5):
    """Process probability map to get final segmentation mask with post-processing"""
    # Threshold probabilities to get binary masks
    disc_map = prob_map[0] > threshold
    cup_map = prob_map[1] > threshold
    
    # Apply median filter for smoothing
    disc_mask = cv2.medianBlur(disc_map.astype(np.uint8), 7)
    cup_mask = cv2.medianBlur(cup_map.astype(np.uint8), 7)
    
    # Apply morphological operations
    disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)
    cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)
    
    # Get largest connected component and fill holes
    disc_mask = get_largest_fillhole(disc_mask)
    cup_mask = get_largest_fillhole(cup_mask)
    
    # Cup is contained within disc
    cup_mask = cup_mask * disc_mask
    
    # Create final mask (0=cup, 1=disc, 2=background)
    result_mask = np.zeros_like(disc_mask, dtype=np.uint8)
    result_mask[disc_mask > 0] = 1  # disc
    result_mask[cup_mask > 0] = 2   # cup
    
    return result_mask