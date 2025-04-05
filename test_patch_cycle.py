#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Import your custom modules
from MultiModalSpectralGPT import MultiModalSpectralGPT
from dataset import create_patient_dataloader
from hsi_to_rgb import simple_hsi_to_rgb


def detailed_hsi_analysis(hsi_img):
    """
    Perform a detailed analysis of the hyperspectral image
    """
    print("\n=== Detailed HSI Image Analysis ===")

    # Ensure we're working with a numpy array or detached tensor
    if isinstance(hsi_img, torch.Tensor):
        hsi_data = hsi_img.detach().cpu()
    else:
        hsi_data = torch.from_numpy(hsi_img).detach().cpu()

    # Squeeze to remove single-dimensional entries
    hsi_data = hsi_data.squeeze()

    print(f"HSI Image Shape: {hsi_data.shape}")
    print(f"Data Type: {hsi_data.dtype}")

    # Compute statistics across different dimensions
    print("\nStatistics:")
    print(f"Overall Min: {hsi_data.min().item():.4f}")
    print(f"Overall Max: {hsi_data.max().item():.4f}")
    print(f"Overall Mean: {hsi_data.mean().item():.4f}")
    print(f"Overall Std Dev: {hsi_data.std().item():.4f}")

    # Determine the correct dimension for bands
    if len(hsi_data.shape) == 3:
        num_bands = hsi_data.shape[0]
        print("\nBand-wise Statistics:")
        for i in range(min(num_bands, 10)):  # Show first 10 bands
            band = hsi_data[i]
            print(f"Band {i}:")
            print(f"  Min: {band.min().item():.4f}")
            print(f"  Max: {band.max().item():.4f}")
            print(f"  Mean: {band.mean().item():.4f}")
            print(f"  Std Dev: {band.std().item():.4f}")
    elif len(hsi_data.shape) == 4:
        num_bands = hsi_data.shape[1]
        print("\nBand-wise Statistics:")
        for i in range(min(num_bands, 10)):  # Show first 10 bands
            band = hsi_data[0, i]
            print(f"Band {i}:")
            print(f"  Min: {band.min().item():.4f}")
            print(f"  Max: {band.max().item():.4f}")
            print(f"  Mean: {band.mean().item():.4f}")
            print(f"  Std Dev: {band.std().item():.4f}")


def visualize_hsi_patches(model, hsi_img):
    """
    Visualize patches at different stages of processing
    """
    print("\n=== Patch Visualization ===")

    # Ensure correct device and tensor type
    if not isinstance(hsi_img, torch.Tensor):
        hsi_img = torch.from_numpy(hsi_img)

    device = hsi_img.device
    model = model.to(device)
    hsi_img = hsi_img.to(device)

    # Ensure correct shape [B, C, T, H, W]
    if len(hsi_img.shape) == 3:
        hsi_img = hsi_img.unsqueeze(0).unsqueeze(0)
    elif len(hsi_img.shape) == 4:
        hsi_img = hsi_img.unsqueeze(0)

    # Perform patch embedding with no gradient computation
    with torch.no_grad():
        patched = model.patch_embed(hsi_img)

    B, T, HW, D = patched.shape

    print(f"Patch Embedding Shape: {patched.shape}")

    # Select a subset of patches for visualization
    num_patches_to_viz = min(16, T * HW)

    plt.figure(figsize=(15, 15))

    # Reshape patches for visualization
    patches_reshaped = patched[0].reshape(T * HW, D)

    # Randomly select patches
    indices = torch.randperm(T * HW)[:num_patches_to_viz]

    for i, idx in enumerate(indices):
        # Ensure no gradient computation
        with torch.no_grad():
            # Project token to pixel space
            patch_pixels = model.pixel_projection(patches_reshaped[idx].detach().unsqueeze(0))

        # Reshape and prepare for visualization
        patch_h, patch_w = model.patch_size
        t_patch = model.t_patch_size

        # Select middle temporal slice for 2D visualization
        mid_t = t_patch // 2
        patch_img = (
            patch_pixels
            .detach()  # Remove gradient information
            .cpu()  # Move to CPU
            .reshape(t_patch, patch_h, patch_w, 1)[mid_t, :, :, 0]
            .numpy()  # Convert to numpy
        )

        plt.subplot(4, 4, i + 1)
        plt.imshow(patch_img, cmap='viridis')
        plt.title(f"Patch {idx}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("debug_output/patch_visualization.png")
    plt.close()


def visualize_raw_patches(hsi_img, model):
    """
    Visualize raw patches before any neural network processing
    """
    print("\n=== Raw Patch Visualization ===")

    # Ensure correct tensor type and shape
    if not isinstance(hsi_img, torch.Tensor):
        hsi_img = torch.from_numpy(hsi_img)

    # Squeeze to remove single-dimensional entries
    hsi_img = hsi_img.squeeze()

    # Get patch dimensions
    patch_h, patch_w = model.patch_size
    t_patch = model.t_patch_size

    # Determine correct spatial dimensions
    if len(hsi_img.shape) == 3:
        H, W = hsi_img.shape[1], hsi_img.shape[2]
        mid_t = hsi_img.shape[0] // 2
    elif len(hsi_img.shape) == 4:
        H, W = hsi_img.shape[2], hsi_img.shape[3]
        mid_t = hsi_img.shape[1] // 2
    else:
        raise ValueError(f"Unexpected HSI image shape: {hsi_img.shape}")

    plt.figure(figsize=(20, 20))

    # Randomly select patches
    num_patches_to_viz = 25

    for i in range(num_patches_to_viz):
        # Randomly select patch location
        h_start = np.random.randint(0, H - patch_h)
        w_start = np.random.randint(0, W - patch_w)

        # Select patch for visualization
        if len(hsi_img.shape) == 3:
            patch = hsi_img[mid_t, h_start:h_start + patch_h, w_start:w_start + patch_w]
        else:
            patch = hsi_img[0, mid_t, h_start:h_start + patch_h, w_start:w_start + patch_w]

        patch_np = patch.detach().cpu().numpy()

        plt.subplot(5, 5, i + 1)
        plt.imshow(patch_np, cmap='viridis')
        plt.title(f"Raw Patch {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("debug_output/raw_patch_visualization.png")
    plt.close()


def create_patch_statistics_report(hsi_img, model):
    """
    Create a detailed statistical report of patches
    """
    print("\n=== Patch Statistics Report ===")

    # Ensure correct tensor type
    if not isinstance(hsi_img, torch.Tensor):
        hsi_img = torch.from_numpy(hsi_img)

    # Squeeze to remove single-dimensional entries
    hsi_img = hsi_img.squeeze()

    # Get patch dimensions
    patch_h, patch_w = model.patch_size
    t_patch = model.t_patch_size

    # Determine correct dimensions
    if len(hsi_img.shape) == 3:
        H, W = hsi_img.shape[1], hsi_img.shape[2]
        mid_t = hsi_img.shape[0] // 2
    elif len(hsi_img.shape) == 4:
        H, W = hsi_img.shape[2], hsi_img.shape[3]
        mid_t = hsi_img.shape[1] // 2
    else:
        raise ValueError(f"Unexpected HSI image shape: {hsi_img.shape}")

    # Collect patch statistics
    patch_means = []
    patch_stds = []
    patch_min_vals = []
    patch_max_vals = []

    # Sample patches
    num_patches_to_sample = 100

    for _ in range(num_patches_to_sample):
        # Randomly select patch location
        h_start = np.random.randint(0, H - patch_h)
        w_start = np.random.randint(0, W - patch_w)

        # Select patch for statistics
        if len(hsi_img.shape) == 3:
            patch = hsi_img[mid_t, h_start:h_start + patch_h, w_start:w_start + patch_w]
        else:
            patch = hsi_img[0, mid_t, h_start:h_start + patch_h, w_start:w_start + patch_w]

        patch_np = patch.detach().cpu().numpy()

        patch_means.append(patch_np.mean())
        patch_stds.append(patch_np.std())
        patch_min_vals.append(patch_np.min())
        patch_max_vals.append(patch_np.max())

    # Create report
    report_path = "debug_output/patch_statistics_report.txt"
    os.makedirs("debug_output", exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("Patch Statistics Report\n")
        f.write("=======================\n\n")
        f.write(f"Patch Size: {patch_h}x{patch_w}, Temporal Slice: {t_patch}\n")
        f.write(f"Number of Patches Sampled: {num_patches_to_sample}\n\n")
        f.write("Patch Mean Statistics:\n")
        f.write(f"  Min Mean: {min(patch_means):.4f}\n")
        f.write(f"  Max Mean: {max(patch_means):.4f}\n")
        f.write(f"  Overall Mean of Means: {np.mean(patch_means):.4f}\n\n")
        f.write("Patch Standard Deviation Statistics:\n")
        f.write(f"  Min Std Dev: {min(patch_stds):.4f}\n")
        f.write(f"  Max Std Dev: {max(patch_stds):.4f}\n")
        f.write(f"  Overall Mean of Std Devs: {np.mean(patch_stds):.4f}\n\n")
        f.write("Patch Extreme Value Statistics:\n")
        f.write(f"  Min Value: {min(patch_min_vals):.4f}\n")
        f.write(f"  Max Value: {max(patch_max_vals):.4f}\n")

    print(f"Patch statistics report saved to {report_path}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Detailed Patch Cycle Test")
    parser.add_argument("--data_dir", type=str, default="dummydata", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="debug_output", help="Output directory for visualizations")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create dataloader
    dataloader = create_patient_dataloader(
        parent_dir=args.data_dir,
        analysis_dim=500,
        target_bands=30,
        batch_size=1,
        shuffle=False
    )

    # Get a sample batch
    try:
        sample_batch = next(iter(dataloader))
        hsi_img = sample_batch['hsi']

        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hsi_img = hsi_img.to(device)

        # Create model on the same device
        model = MultiModalSpectralGPT(
            analysis_dim=500,
            patch_size=(25, 25),
            embed_dim=768,
            t_patch_size=5,
            num_frames=30,
            in_chans=1,
            aux_chans=1
        ).to(device)

        # Perform comprehensive analysis
        detailed_hsi_analysis(hsi_img)

        # Add additional error handling for each visualization step
        try:
            visualize_hsi_patches(model, hsi_img)
        except Exception as patch_viz_error:
            print(f"Error in HSI patch visualization: {patch_viz_error}")
            import traceback
            traceback.print_exc()

        try:
            visualize_raw_patches(hsi_img, model)
        except Exception as raw_patch_error:
            print(f"Error in raw patch visualization: {raw_patch_error}")
            import traceback
            traceback.print_exc()

        try:
            create_patch_statistics_report(hsi_img, model)
        except Exception as stats_error:
            print(f"Error in patch statistics report: {stats_error}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()