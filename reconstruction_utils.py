"""
Utility functions for reconstructing HSI from model outputs.
Handles proper token unmasking and mapping from embedding space back to image space.
"""

import torch
import numpy as np
import torch.nn.functional as F





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


def convert_hsi_to_rgb(hsi_data, wavelength_range=(450, 905)):
    """
    More robust HSI to RGB conversion with no modifications.
    This is just a placeholder duplicate of the function in reconstruction_utils.
    """
    # Ensure numpy array with float32
    if torch.is_tensor(hsi_data):
        hsi_data = hsi_data.detach().cpu().numpy()

    hsi_data = hsi_data.astype(np.float32)

    # Robust normalization
    def robust_normalize(img):
        p2, p98 = np.percentile(img, (2, 98))
        img_norm = np.interp(img, (p2, p98), (0, 1))
        return img_norm

    # Rest of existing conversion logic remains the same
    min_wl, max_wl = wavelength_range
    wavelengths = np.linspace(min_wl, max_wl, hsi_data.shape[0])

    r_idx = np.argmin(np.abs(wavelengths - 650))
    g_idx = np.argmin(np.abs(wavelengths - 550))
    b_idx = np.argmin(np.abs(wavelengths - 450))

    rgb_img = np.zeros((hsi_data.shape[1], hsi_data.shape[2], 3))

    # Apply per-channel robust normalization
    rgb_img[:, :, 0] = robust_normalize(hsi_data[r_idx])
    rgb_img[:, :, 1] = robust_normalize(hsi_data[g_idx])
    rgb_img[:, :, 2] = robust_normalize(hsi_data[b_idx])

    return rgb_img


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