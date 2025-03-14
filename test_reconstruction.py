#!/usr/bin/env python
"""
Complete test script for HSI reconstruction that shows:
1. Original image
2. Masked image (black for masked regions)
3. Full reconstruction (all patches)
4. Hybrid image (only masked patches reconstructed)

This clearly shows the difference between masked and unmasked patches
in an untrained model.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from MultiModalSpectralGPT import MultiModalSpectralGPT
from dataset import create_patient_dataloader


def convert_hsi_to_rgb(hsi_data, wavelength_range=(450, 905)):
    """Convert HSI to RGB for visualization."""
    # Handle batch dimension
    if len(hsi_data.shape) == 5:
        hsi_data = hsi_data[0]

    # Convert to numpy
    hsi_data = hsi_data.squeeze(0).detach().cpu().numpy()

    # Calculate wavelength indices
    num_bands = hsi_data.shape[0]
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_bands)
    r_idx = np.argmin(np.abs(wavelengths - 650))
    g_idx = np.argmin(np.abs(wavelengths - 550))
    b_idx = np.argmin(np.abs(wavelengths - 450))

    # Create RGB image
    rgb = np.zeros((hsi_data.shape[1], hsi_data.shape[2], 3))
    rgb[:, :, 0] = hsi_data[r_idx]
    rgb[:, :, 1] = hsi_data[g_idx]
    rgb[:, :, 2] = hsi_data[b_idx]

    # Normalize
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    return rgb


def test_reconstruction_hybrid():
    """
    Test reconstruction with a hybrid visualization that shows the difference
    between masked and unmasked patches in an untrained model.
    """
    print("Starting hybrid reconstruction test...")

    # Setup
    data_dir = "dummydata"
    output_dir = "test_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloader
    print(f"Loading data from {data_dir}")
    try:
        dataloader = create_patient_dataloader(
            parent_dir=data_dir,
            analysis_dim=500,
            target_bands=30,
            batch_size=1,
            num_workers=0
        )

        # Get a batch
        batch = next(iter(dataloader))
        print("Successfully loaded data from dataloader")
    except Exception as e:
        print(f"Error loading data: {e}, creating synthetic data")

        # Create synthetic data if real data fails
        hsi_data = torch.randn(1, 1, 30, 500, 500)
        aux_data = {
            'ir': torch.randn(1, 1, 500, 500),
            'af': torch.randn(1, 1, 500, 500),
            'thickness': torch.randn(1, 1, 500, 500)
        }
        batch = {
            'hsi': hsi_data,
            'aux_data': aux_data,
            'batch_idx': torch.tensor([0]),
            'patient_id': ['synthetic']
        }

    # Create model
    print("Creating model...")
    model = MultiModalSpectralGPT(
        analysis_dim=500,
        patch_size=(25, 25),
        embed_dim=768,
        depth=4,  # Smaller for testing
        num_heads=8,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        num_frames=30,
        t_patch_size=5,
        in_chans=1,
        aux_chans=1,
        mask_ratio=0.75,
        contrastive_mode='global'
    )

    # Move to device
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Move data to device
    hsi = batch['hsi'].to(device)
    aux = {k: v.to(device) if v is not None else None for k, v in batch['aux_data'].items()}
    batch_idx = batch['batch_idx'].to(device)

    # Forward pass
    print("Running model forward pass...")
    with torch.no_grad():
        output = model(hsi, aux, batch_idx)

    # Get outputs
    pred = output['pred']
    mask = output['mask']
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")

    # Reconstruction
    print("Reconstructing HSI from predictions...")
    B, C, T, H, W = hsi.shape
    try:
        reconstructed = model.reconstruct_pixels(pred, B, C, T, H, W)
        reconstruction_success = True
        print(f"Reconstruction shape: {reconstructed.shape}")
    except Exception as e:
        print(f"Error in reconstruct_pixels: {e}")
        import traceback
        traceback.print_exc()
        reconstructed = torch.randn_like(hsi)
        reconstruction_success = False

    # Get patch dimensions
    patch_h, patch_w = model.patch_size
    t_patch_size = model.t_patch_size

    # Calculate grid dimensions
    num_patches_h = H // patch_h
    num_patches_w = W // patch_w
    num_patches_t = T // t_patch_size
    total_patches = num_patches_h * num_patches_w * num_patches_t

    print(f"Patch grid: {num_patches_t}×{num_patches_h}×{num_patches_w} = {total_patches} patches")
    print(f"Mask has {mask.sum().item()} masked patches out of {mask.shape[1]} total")

    # Reshape mask for easier indexing if possible
    try:
        if mask.shape[1] == total_patches:
            # Perfect match, reshape directly
            mask_3d = mask.reshape(B, num_patches_t, num_patches_h, num_patches_w)
            print("Successfully reshaped mask to 3D grid")
        else:
            # Mismatch, need a different approach
            print(f"WARNING: Mask shape mismatch: {mask.shape[1]} vs {total_patches}")
            # Create a sequential mask (some approximation)
            mask_3d = torch.zeros(B, num_patches_t, num_patches_h, num_patches_w, device=device)
            # Fill as many indices as we have
            seq_indices = torch.argsort(torch.rand_like(mask))
            mask_flat = mask.reshape(-1)
            for i in range(min(len(seq_indices), total_patches)):
                idx = seq_indices[i]
                if idx < total_patches:
                    t_idx = (idx // (num_patches_h * num_patches_w)) % num_patches_t
                    h_idx = (idx // num_patches_w) % num_patches_h
                    w_idx = idx % num_patches_w

                    if i < len(mask_flat):
                        mask_3d[0, t_idx, h_idx, w_idx] = mask_flat[i]
    except Exception as e:
        print(f"Error reshaping mask: {e}, using fallback approach")
        # Fallback: create a random mask pattern for visualization
        # This won't match the actual masking but will show the concept
        mask_3d = torch.bernoulli(torch.ones(B, num_patches_t, num_patches_h, num_patches_w, device=device) * 0.75)

    # Create a hybrid image - original for unmasked, reconstruction for masked
    hybrid = hsi.clone()

    # Also create a masked version showing only unmasked patches
    masked_vis = hsi.clone()

    # Loop through patches
    print("Creating hybrid and masked visualizations...")
    for t in range(0, T, t_patch_size):
        t_idx = t // t_patch_size
        if t_idx >= mask_3d.shape[1]:
            continue

        for h in range(0, H, patch_h):
            h_idx = h // patch_h
            if h_idx >= mask_3d.shape[2]:
                continue

            for w in range(0, W, patch_w):
                w_idx = w // patch_w
                if w_idx >= mask_3d.shape[3]:
                    continue

                # Check if this patch was masked
                if mask_3d[0, t_idx, h_idx, w_idx] > 0.5:
                    # This was a masked patch
                    t_end = min(t + t_patch_size, T)
                    h_end = min(h + patch_h, H)
                    w_end = min(w + patch_w, W)

                    # For hybrid: use reconstruction for masked patches
                    hybrid[0, :, t:t_end, h:h_end, w:w_end] = reconstructed[0, :, t:t_end, h:h_end, w:w_end]

                    # For masked visualization: set masked patches to zero (black)
                    masked_vis[0, :, t:t_end, h:h_end, w:w_end] = 0.0

    # Visualize
    print("Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original
    axes[0, 0].imshow(convert_hsi_to_rgb(hsi))
    axes[0, 0].set_title("Original HSI")
    axes[0, 0].axis('off')

    # Masked (with black for masked regions)
    axes[0, 1].imshow(convert_hsi_to_rgb(masked_vis))
    axes[0, 1].set_title("Masked HSI (black = masked)")
    axes[0, 1].axis('off')

    # Full reconstruction
    axes[1, 0].imshow(convert_hsi_to_rgb(reconstructed))
    axes[1, 0].set_title("Full Reconstruction (untrained)")
    axes[1, 0].axis('off')

    # Hybrid (only masked patches reconstructed)
    axes[1, 1].imshow(convert_hsi_to_rgb(hybrid))
    axes[1, 1].set_title("Hybrid (only masked patches reconstructed)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    hybrid_path = os.path.join(output_dir, "hybrid_visualization.png")
    plt.savefig(hybrid_path, dpi=300)
    print(f"Hybrid visualization saved to {hybrid_path}")

    # Also create a visualization of the mask pattern
    plt.figure(figsize=(8, 8))
    # Average across spectral dimension for clarity
    mask_spatial = mask_3d.mean(dim=1).squeeze().cpu().numpy()
    plt.imshow(mask_spatial, cmap='gray')
    plt.title("Mask Pattern (white = masked)")
    plt.colorbar(label="Mask Value")
    mask_path = os.path.join(output_dir, "mask_pattern.png")
    plt.savefig(mask_path, dpi=300)
    print(f"Mask visualization saved to {mask_path}")

    # Create a simple comparison showing just original vs reconstruction
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(convert_hsi_to_rgb(hsi))
    axes[0].set_title("Original HSI")
    axes[0].axis('off')

    axes[1].imshow(convert_hsi_to_rgb(reconstructed))
    axes[1].set_title("Reconstructed HSI")
    axes[1].axis('off')

    plt.tight_layout()
    simple_path = os.path.join(output_dir, "simple_comparison.png")
    plt.savefig(simple_path, dpi=300)
    print(f"Simple comparison saved to {simple_path}")

    return hybrid_path


if __name__ == "__main__":
    test_reconstruction_hybrid()