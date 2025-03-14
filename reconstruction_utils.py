"""
Utility functions for reconstructing HSI from model outputs.
Handles proper token unmasking and mapping from embedding space back to image space.
"""

import torch
import numpy as np

import torch
import numpy as np


def reconstruct_hsi_from_mae(model, hsi_original, predictions, mask):
    """
    Reconstruct full HSI volume from MAE outputs.

    Args:
        model: The MultiModalSpectralGPT model
        hsi_original: Original HSI data of shape [B, C, T, H, W]
        predictions: Predicted patch embeddings from decoder of shape [B, num_patches, embed_dim]
        mask: Binary mask indicating which tokens were masked (1 = masked, 0 = kept)

    Returns:
        Reconstructed HSI data of the same shape as the input: [B, C, T, H, W]

    Raises:
        ValueError: If there are issues with input shapes or reconstruction
    """
    # Get original dimensions
    B, C, T, H, W = hsi_original.shape

    # Get patch sizes from the model
    patch_h, patch_w = model.patch_size
    t_patch_size = model.t_patch_size

    # Calculate the spatial and temporal grid dimensions
    num_patches_h = H // patch_h
    num_patches_w = W // patch_w
    num_patches_t = T // t_patch_size
    total_patches = num_patches_h * num_patches_w * num_patches_t

    # Validate input shapes
    if predictions.shape[0] != B:
        raise ValueError(f"Batch size mismatch: predictions {predictions.shape[0]} vs original {B}")

    if predictions.shape[1] != total_patches:
        raise ValueError(f"Patch count mismatch: predictions has {predictions.shape[1]} patches, "
                         f"but expected {total_patches} patches (grid: {num_patches_t}x{num_patches_h}x{num_patches_w})")

    # Check if pixel decoder exists and is properly configured
    if not hasattr(model, 'pixel_decoder') or model.pixel_decoder is None:
        raise AttributeError("Model is missing pixel_decoder. Cannot reconstruct HSI.")

    # Reshape predictions to match patch grid
    predictions_reshaped = predictions.reshape(
        B, num_patches_t, num_patches_h, num_patches_w, -1
    )

    # Create reconstruction tensor
    reconstructed = torch.zeros(
        B, C, T, H, W,
        dtype=hsi_original.dtype,
        device=hsi_original.device
    )

    # Reconstruct patch by patch
    for t_idx in range(num_patches_t):
        for h_idx in range(num_patches_h):
            for w_idx in range(num_patches_w):
                # Compute start/end indices for this patch
                t_start = t_idx * t_patch_size
                t_end = t_start + t_patch_size
                h_start = h_idx * patch_h
                h_end = h_start + patch_h
                w_start = w_idx * patch_w
                w_end = w_start + patch_w

                # Extract patch prediction
                patch_pred = predictions_reshaped[0, t_idx, h_idx, w_idx]

                # Use pixel decoder to convert embedding to patch
                patch_pixels = model.pixel_decoder(patch_pred.unsqueeze(0)).squeeze(0)
                patch_pixels = patch_pixels.reshape(C, t_patch_size, patch_h, patch_w)

                # Place in reconstruction tensor
                reconstructed[0, :, t_start:t_end, h_start:h_end, w_start:w_end] = patch_pixels

    # Normalize to match original data range
    orig_min = torch.min(hsi_original)
    orig_max = torch.max(hsi_original)
    reconstructed = (reconstructed - torch.min(reconstructed)) / (
                torch.max(reconstructed) - torch.min(reconstructed) + 1e-8)
    reconstructed = reconstructed * (orig_max - orig_min) + orig_min

    return reconstructed


def get_unmasked_indices(mask):
    """
    Get indices of unmasked tokens.
    """
    return torch.nonzero(mask < 0.5).squeeze()


def get_masked_indices(mask):
    """
    Get indices of masked tokens.
    """
    return torch.nonzero(mask > 0.5).squeeze()


def model_free_unpatchify(tokens, B, C, T, H, W, t_patch_size, patch_h, patch_w):
    """
    Convert tokens back to image space without requiring model-specific unpatchify function.
    
    Args:
        tokens: Tokens of shape [B, T, HW, D]
        B, C, T, H, W: Original dimensions
        t_patch_size, patch_h, patch_w: Patch dimensions
        
    Returns:
        Image-space representation [B, C, T, H, W]
    """
    # Determine grid dimensions
    num_patches_h = H // patch_h
    num_patches_w = W // patch_w
    num_patches_t = T // t_patch_size
    
    # Calculate patch size in embedding dimension
    # We're assuming a simple case where embed_dim is proportional to patch volume
    D = tokens.shape[-1]
    
    # First reshape to separate spatial dimensions [B, T, H, W, D]
    reshaped = tokens.reshape(B, num_patches_t, num_patches_h, num_patches_w, D)
    
    # Implement a basic "unpooling" operation to expand each token back to a patch
    # For simplicity, we'll distribute the embedding dimension values across the patch
    # We'll split D into t_patch_size * patch_h * patch_w * C parts

    # Calculate if we need to adjust D based on the target patch volume
    patch_volume = t_patch_size * patch_h * patch_w * C
    
    # Create a simple unpooling by distributing embedding values and reshaping
    try:
        # First, reshape to prepare for unpooling
        pixel_values = reshaped.permute(0, 4, 1, 2, 3)  # [B, D, T, H, W]
        
        # Use FoldUnfold or similar technique to unpool
        # As a simple approach, we'll use interpolation
        pixel_values = torch.nn.functional.interpolate(
            pixel_values, 
            size=(T, H, W), 
            mode='trilinear',
            align_corners=False
        )
        
        # If embedding dimension doesn't match the expected output channels
        if pixel_values.shape[1] != C:
            # Project to the correct number of channels using 1x1x1 convolution metaphor
            # (implemented as a linear projection across the channel dimension)
            pixel_values = pixel_values.permute(0, 2, 3, 4, 1)  # [B, T, H, W, D]
            # Simple linear projection to target channels
            # Create a weight matrix for projection
            device = tokens.device
            channel_projection = torch.nn.Linear(pixel_values.shape[-1], C, device=device)
            # Apply projection
            pixel_values = channel_projection(pixel_values)  # [B, T, H, W, C]
            # Permute back to expected output format
            pixel_values = pixel_values.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
    except Exception as e:
        # Fallback implementation if the above fails
        print(f"Using fallback unpatchify due to error: {e}")
        
        # Reshape tokens to match the original patches
        tokens_reshaped = reshaped.permute(0, 1, 2, 3, 4).reshape(
            B, num_patches_t, num_patches_h, num_patches_w, -1
        )
        
        # Create output tensor
        pixel_values = torch.zeros((B, C, T, H, W), device=tokens.device, dtype=tokens.dtype)
        
        # Unfold tokens back to patches using simple reshaping and placement
        for t in range(num_patches_t):
            for h in range(num_patches_h):
                for w in range(num_patches_w):
                    # Extract the token for this position
                    token = tokens_reshaped[:, t, h, w, :]
                    
                    # Convert token to patch (simplified approach)
                    # Distribute token values across a patch of zeros
                    patch = token.reshape(B, C, t_patch_size, patch_h, patch_w)
                    
                    # Place patch in the output tensor
                    t_start = t * t_patch_size
                    h_start = h * patch_h
                    w_start = w * patch_w
                    pixel_values[:, :, t_start:t_start+t_patch_size, 
                                h_start:h_start+patch_h, 
                                w_start:w_start+patch_w] = patch
    
    return pixel_values


def get_unmasked_indices(mask):
    """
    Get indices of unmasked tokens.
    
    Args:
        mask: Binary mask where 1 indicates masked tokens, 0 indicates kept tokens
        
    Returns:
        Indices of unmasked (kept) tokens
    """
    return torch.nonzero(mask < 0.5).squeeze()


def get_masked_indices(mask):
    """
    Get indices of masked tokens.
    
    Args:
        mask: Binary mask where 1 indicates masked tokens, 0 indicates kept tokens
        
    Returns:
        Indices of masked tokens
    """
    return torch.nonzero(mask > 0.5).squeeze()
