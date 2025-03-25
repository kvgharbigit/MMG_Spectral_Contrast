import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from hsi_to_rgb import simple_hsi_to_rgb


def verify_reconstruction_pipeline(model, hsi_image, device, save_path=None):
    """
    Verify the HSI reconstruction pipeline by:
    1. Passing an HSI image through the patch embedding process
    2. Reconstructing the HSI from the embeddings using the pseudo-inverse
    3. Comparing the original and reconstructed HSI

    Args:
        model: The MultiModalSpectralGPT model
        hsi_image: The original HSI image tensor [B, C, T, H, W]
        device: The computation device
        save_path: Path to save the visualization

    Returns:
        tuple: (figure, reconstruction_error)
    """
    # Ensure inputs are on the correct device
    hsi_image = hsi_image.to(device)
    model = model.to(device)

    # Step 1: Get the patch embedding weights
    proj_weight = model.patch_embed.proj.weight  # [D, C, Tp, Ph, Pw]
    D, C, Tp, Ph, Pw = proj_weight.shape

    # Step 2: Create a pseudo-inverse
    flattened_weight = proj_weight.reshape(D, -1)  # [D, C*Tp*Ph*Pw]
    pseudo_inv = torch.pinverse(flattened_weight)  # [C*Tp*Ph*Pw, D]

    # Step 3: Pass the HSI image through the patch embedding process
    with torch.no_grad():
        # Use the model's patch embedding directly
        embeddings = model.patch_embed(hsi_image)  # [B, t_patches, h_patches*w_patches, D]

        # Get shapes
        B, t_patches, hw_patches, D = embeddings.shape
        h_patches = w_patches = int(np.sqrt(hw_patches))

        # Reshape embeddings for processing
        embeddings = embeddings.reshape(B, t_patches * hw_patches, D)  # [B, L, D]

    # Step 4: Reconstruct the HSI from embeddings
    # Get dimensions from model
    T = model.patch_embed.frames
    H = W = model.analysis_dim
    patch_h, patch_w = model.patch_size
    t_patch_size = model.t_patch_size

    # Create tensor for reconstructed HSI
    reconstructed_hsi = torch.zeros((B, C, T, H, W), device=device)

    # Reshape embeddings to match 3D patch organization
    embeddings_3d = embeddings.reshape(B, t_patches, h_patches, w_patches, D)

    # Loop through each patch
    for b in range(B):
        for t_idx in range(t_patches):
            for h_idx in range(h_patches):
                for w_idx in range(w_patches):
                    # Get token embedding for this patch
                    token = embeddings_3d[b, t_idx, h_idx, w_idx]

                    # Apply pseudo-inverse to get patch pixels
                    patch_pixels = torch.matmul(pseudo_inv, token)

                    # Reshape to patch dimensions [C, Tp, Ph, Pw]
                    patch_pixels = patch_pixels.reshape(C, t_patch_size, patch_h, patch_w)

                    # Calculate location in the HSI volume
                    t_start = t_idx * t_patch_size
                    h_start = h_idx * patch_h
                    w_start = w_idx * patch_w

                    # Place the patch in the reconstructed HSI
                    reconstructed_hsi[b, :,
                    t_start:t_start + t_patch_size,
                    h_start:h_start + patch_h,
                    w_start:w_start + patch_w] = patch_pixels

    # Step 5: Calculate reconstruction error
    # Mean squared error across the entire HSI volume
    mse = ((reconstructed_hsi - hsi_image) ** 2).mean().item()
    # Peak signal-to-noise ratio (PSNR)
    max_val = max(hsi_image.max().item(), reconstructed_hsi.max().item())
    psnr = 10 * np.log10((max_val ** 2) / mse) if mse > 0 else float('inf')

    # Normalize the reconstructed HSI for better visualization
    for b in range(B):
        # Get min/max values
        recon_min = reconstructed_hsi[b].min()
        recon_max = reconstructed_hsi[b].max()
        # Normalize to [0, 1]
        reconstructed_hsi[b] = (reconstructed_hsi[b] - recon_min) / (recon_max - recon_min + 1e-8)

    # Step 6: Visualize the results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Convert to RGB for visualization
    orig_rgb = simple_hsi_to_rgb(hsi_image)
    recon_rgb = simple_hsi_to_rgb(reconstructed_hsi)

    # Prepare for plotting
    orig_rgb_np = orig_rgb[0].permute(1, 2, 0).cpu().numpy()
    recon_rgb_np = recon_rgb[0].permute(1, 2, 0).cpu().numpy()

    # Clip to valid range
    orig_rgb_np = np.clip(orig_rgb_np, 0, 1)
    recon_rgb_np = np.clip(recon_rgb_np, 0, 1)

    # Plot RGB representations
    axes[0, 0].imshow(orig_rgb_np)
    axes[0, 0].set_title("Original HSI (RGB)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(recon_rgb_np)
    axes[0, 1].set_title("Reconstructed HSI (RGB)")
    axes[0, 1].axis('off')

    # Calculate and plot RGB error
    rgb_error = np.abs(orig_rgb_np - recon_rgb_np)
    error_img = axes[0, 2].imshow(rgb_error)
    axes[0, 2].set_title(f"RGB Error (MSE: {mse:.6f}, PSNR: {psnr:.2f}dB)")
    axes[0, 2].axis('off')
    plt.colorbar(error_img, ax=axes[0, 2])

    # Select a spectral slice for comparison (middle slice)
    slice_idx = T // 2

    # Get the slices
    orig_slice = hsi_image[0, 0, slice_idx].cpu().numpy()
    recon_slice = reconstructed_hsi[0, 0, slice_idx].cpu().numpy()

    # Plot spectral slices
    axes[1, 0].imshow(orig_slice, cmap='viridis')
    axes[1, 0].set_title(f"Original (Spectral Band {slice_idx})")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(recon_slice, cmap='viridis')
    axes[1, 1].set_title(f"Reconstructed (Spectral Band {slice_idx})")
    axes[1, 1].axis('off')

    # Calculate and plot error for this slice
    slice_error = np.abs(orig_slice - recon_slice)
    slice_mse = np.mean(slice_error ** 2)
    slice_psnr = 10 * np.log10((max(orig_slice.max(), recon_slice.max()) ** 2) / slice_mse) if slice_mse > 0 else float(
        'inf')

    slice_error_img = axes[1, 2].imshow(slice_error, cmap='hot')
    axes[1, 2].set_title(f"Spectral Slice Error (MSE: {slice_mse:.6f}, PSNR: {slice_psnr:.2f}dB)")
    axes[1, 2].axis('off')
    plt.colorbar(slice_error_img, ax=axes[1, 2])

    plt.tight_layout()
    fig.suptitle("Verification of HSI Reconstruction Pipeline", fontsize=16)
    plt.subplots_adjust(top=0.94)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Return the figure and error metrics
    return fig, {
        'mse': mse,
        'psnr': psnr,
        'slice_mse': slice_mse,
        'slice_psnr': slice_psnr
    }


def evaluate_reconstruction_pipeline(model, dataloader, device, output_dir=None):
    """
    Run a comprehensive evaluation of the HSI reconstruction pipeline on a dataset.

    Args:
        model: The MultiModalSpectralGPT model
        dataloader: DataLoader containing HSI data
        device: Computation device
        output_dir: Directory to save visualizations

    Returns:
        dict: Evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Collect metrics
    all_metrics = {
        'mse': [],
        'psnr': [],
        'slice_mse': [],
        'slice_psnr': []
    }

    # Get a few samples for evaluation
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Only process a few batches for evaluation
            if batch_idx >= 5:
                break

            # Get HSI data
            hsi = batch['hsi']

            # Run verification
            save_path = os.path.join(output_dir, f'reconstruction_verification_{batch_idx}.png') if output_dir else None
            _, metrics = verify_reconstruction_pipeline(model, hsi, device, save_path)

            # Collect metrics
            for key, value in metrics.items():
                all_metrics[key].append(value)

            print(f"Batch {batch_idx}: MSE = {metrics['mse']:.6f}, PSNR = {metrics['psnr']:.2f} dB")

    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    print("\nReconstruction Pipeline Evaluation Results:")
    print(f"Average MSE: {avg_metrics['mse']:.6f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average Spectral Slice MSE: {avg_metrics['slice_mse']:.6f}")
    print(f"Average Spectral Slice PSNR: {avg_metrics['slice_psnr']:.2f} dB")

    return avg_metrics