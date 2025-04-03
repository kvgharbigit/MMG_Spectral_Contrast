#!/usr/bin/env python
"""
Test script to verify the patch-unpatch cycle in MultiModalSpectralGPT.
This script tests whether the patch embedding and unpatchify functions
are correctly inverse operations of each other.
"""

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


def test_patch_cycle(model, input_image, output_dir="test_outputs"):
    """
    Test if patch embedding and unpatchify work together correctly.

    Args:
        model: Your MultiModalSpectralGPT model
        input_image: Input HSI data tensor [B, C, T, H, W]
        output_dir: Directory to save visualizations

    Returns:
        tuple: (error, reconstructed image)
    """
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Step 1: Apply patch embedding
        print(f"Input image shape: {input_image.shape}")
        patched = model.patch_embed(input_image)

        # Print shape information
        print(f"After patch embedding: {patched.shape}")

        # Step 2: Flatten tokens as done in the model
        B, T, HW, D = patched.shape
        flattened = patched.reshape(B, T * HW, D)
        print(f"After flattening: {flattened.shape}")

        # Step 3: Try to unpatchify directly (without any model processing)
        reconstructed = model.unpatchify(flattened, input_image.shape)
        print(f"After unpatchify: {reconstructed.shape}")

        # Calculate error metrics
        mse = ((reconstructed - input_image) ** 2).mean().item()
        print(f"Reconstruction MSE: {mse:.6f}")

        # More detailed metrics per dimension
        channel_mse = ((reconstructed - input_image) ** 2).mean(dim=(0, 2, 3, 4)).tolist()
        spectral_mse = ((reconstructed - input_image) ** 2).mean(dim=(0, 1, 3, 4)).tolist()

        print(f"Channel-wise MSE: {channel_mse}")
        print(f"Spectral-wise MSE (first 5 bands): {spectral_mse[:5]}")

        # Visualize using RGB conversion
        try:
            # Try to convert both to RGB
            orig_rgb = simple_hsi_to_rgb(input_image[0])
            recon_rgb = simple_hsi_to_rgb(reconstructed[0])

            # Ensure they're in the right format for matplotlib
            if orig_rgb.shape[0] == 3:  # [3, H, W]
                orig_rgb = orig_rgb.permute(1, 2, 0).cpu().numpy()
                recon_rgb = recon_rgb.permute(1, 2, 0).cpu().numpy()
            else:
                orig_rgb = orig_rgb.cpu().numpy()
                recon_rgb = recon_rgb.cpu().numpy()

            # Create visualization
            plt.figure(figsize=(12, 8))

            # Original
            plt.subplot(2, 2, 1)
            plt.imshow(orig_rgb)
            plt.title("Original (RGB)")
            plt.axis('off')

            # Reconstructed
            plt.subplot(2, 2, 2)
            plt.imshow(recon_rgb)
            plt.title("After Patch-Unpatch (RGB)")
            plt.axis('off')

            # Difference
            plt.subplot(2, 2, 3)
            diff = np.abs(orig_rgb - recon_rgb).mean(axis=2)
            plt.imshow(diff, cmap='hot')
            plt.colorbar()
            plt.title("Absolute Difference")
            plt.axis('off')

            # Sample spectral band
            mid_band = input_image.shape[2] // 2
            plt.subplot(2, 2, 4)
            orig_band = input_image[0, 0, mid_band].cpu().numpy()
            recon_band = reconstructed[0, 0, mid_band].cpu().numpy()

            # Plot a single spectral band comparison
            plt.imshow(np.hstack([orig_band, recon_band]), cmap='viridis')
            plt.title(f"Spectral Band {mid_band} (Original | Reconstructed)")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "patch_cycle_test.png"), dpi=200)
            plt.close()

            # Also visualize a few spectral bands separately
            num_bands = min(5, input_image.shape[2])
            plt.figure(figsize=(15, 3 * num_bands))

            for i in range(num_bands):
                band_idx = i * (input_image.shape[2] // num_bands)
                orig_band = input_image[0, 0, band_idx].cpu().numpy()
                recon_band = reconstructed[0, 0, band_idx].cpu().numpy()
                band_diff = np.abs(orig_band - recon_band)

                plt.subplot(num_bands, 3, i * 3 + 1)
                plt.imshow(orig_band, cmap='viridis')
                plt.title(f"Band {band_idx} Original")
                plt.axis('off')

                plt.subplot(num_bands, 3, i * 3 + 2)
                plt.imshow(recon_band, cmap='viridis')
                plt.title(f"Band {band_idx} Reconstructed")
                plt.axis('off')

                plt.subplot(num_bands, 3, i * 3 + 3)
                plt.imshow(band_diff, cmap='hot')
                plt.title(f"Difference (MSE: {((orig_band - recon_band) ** 2).mean():.6f})")
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "spectral_bands_test.png"), dpi=200)
            plt.close()

        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()

        return mse, reconstructed


def test_patches_visualization(model, input_image, output_dir="test_outputs"):
    """
    Visualize individual patches before and after the patching process.

    Args:
        model: Your MultiModalSpectralGPT model
        input_image: Input HSI data tensor [B, C, T, H, W]
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Apply patch embedding
        patched = model.patch_embed(input_image)
        B, T, HW, D = patched.shape
        flattened = patched.reshape(B, T * HW, D)

        # Project tokens to pixel space using model's pixel projection
        if hasattr(model, 'pixel_projection'):
            # Get number of patches to visualize
            num_patches = min(16, T * HW)
            patch_h, patch_w = model.patch_size
            t_patch = model.t_patch_size
            C = input_image.shape[1]

            # Select random patches to visualize
            indices = torch.randperm(T * HW)[:num_patches]

            plt.figure(figsize=(12, 12))
            for i, idx in enumerate(indices):
                # Project token to pixel space
                token = flattened[0, idx]
                if isinstance(model.pixel_projection, torch.nn.Sequential):
                    patch_pixels = model.pixel_projection(token.unsqueeze(0)).reshape(t_patch, patch_h, patch_w, C)
                else:
                    patch_pixels = model.pixel_projection(token.unsqueeze(0)).reshape(t_patch, patch_h, patch_w, C)

                # Get mid-point of temporal dimension for visualization
                mid_t = t_patch // 2
                patch_img = patch_pixels[mid_t, :, :, 0].cpu().numpy()

                # Plot patch
                plt.subplot(4, 4, i + 1)
                plt.imshow(patch_img, cmap='viridis')
                plt.title(f"Patch {idx}")
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "sample_patches.png"), dpi=200)
            plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Test patch-unpatch cycle in MultiModalSpectralGPT")
    parser.add_argument("--data_dir", type=str, default="dummydata", help="Path to data directory (default: dummydata)")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="test_outputs", help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--analysis_dim", type=int, default=500, help="Analysis dimension")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of spectral frames")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--patch_size", type=int, default=25, help="Patch size")
    parser.add_argument("--t_patch_size", type=int, default=5, help="Temporal patch size")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Warning: Data directory {args.data_dir} does not exist, creating it.")
        os.makedirs(args.data_dir, exist_ok=True)

        # Check if it's empty (no H5 files)
        h5_files = list(Path(args.data_dir).glob("**/*.h5"))
        if len(h5_files) == 0:
            print("No data files found. Creating synthetic test data...")

            # Create synthetic dummy data
            try:
                import numpy as np
                from PIL import Image

                # Create subdirectory for synthetic data
                dummy_dir = os.path.join(args.data_dir, "synthetic_patient")
                os.makedirs(dummy_dir, exist_ok=True)

                # Create a synthetic H5 file
                try:
                    import h5py
                    h5_path = os.path.join(dummy_dir, "synthetic_hsi.h5")
                    with h5py.File(h5_path, 'w') as f:
                        # Create a 30-band HSI cube
                        hsi_data = np.random.rand(30, 500, 500).astype(np.float32)
                        f.create_dataset('Cube/Images', data=hsi_data)
                    print(f"Created synthetic HSI data: {h5_path}")
                except ImportError:
                    print("H5py not available. Creating dummy TIFF files only.")

                # Create synthetic auxillary TIFF files
                for aux_type in ["FAF", "IR", "thickness"]:
                    tiff_path = os.path.join(dummy_dir, f"{aux_type}.tiff")
                    aux_data = np.random.rand(500, 500).astype(np.float32)
                    if aux_type == "thickness":
                        # Create a circular mask for thickness
                        y, x = np.ogrid[-250:250, -250:250]
                        mask = (x * x + y * y <= 200 * 200).astype(np.float32)
                        aux_data = aux_data * mask

                    Image.fromarray((aux_data * 255).astype(np.uint8)).save(tiff_path)
                    print(f"Created synthetic {aux_type} data: {tiff_path}")

            except Exception as e:
                print(f"Error creating synthetic data: {e}")
                print("Please provide a valid data directory with HSI data.")
                sys.exit(1)

    # Create dataloader
    dataloader = create_patient_dataloader(
        parent_dir=args.data_dir,
        analysis_dim=args.analysis_dim,
        target_bands=args.num_frames,
        batch_size=1,
        shuffle=False
    )

    try:
        sample_batch = next(iter(dataloader))
        print(f"Loaded sample batch with shapes:")
        print(f"  HSI: {sample_batch['hsi'].shape}")
        for key, value in sample_batch['aux_data'].items():
            if value is not None:
                print(f"  {key}: {value.shape}")
    except StopIteration:
        print(f"Error: Could not load any data from {args.data_dir}")
        print("Please ensure the directory contains proper HSI data files.")
        sys.exit(1)

    # Create model
    model = MultiModalSpectralGPT(
        analysis_dim=args.analysis_dim,
        patch_size=(args.patch_size, args.patch_size),
        embed_dim=args.embed_dim,
        t_patch_size=args.t_patch_size,
        num_frames=args.num_frames,
        in_chans=1,
        aux_chans=1
    ).to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from checkpoint: {args.checkpoint}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model weights from: {args.checkpoint}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()

    # Run tests
    print("\n=== Testing Patch-Unpatch Cycle ===")
    error, reconstructed = test_patch_cycle(
        model,
        sample_batch['hsi'].to(device),
        output_dir=args.output_dir
    )

    print("\n=== Testing Individual Patches ===")
    test_patches_visualization(
        model,
        sample_batch['hsi'].to(device),
        output_dir=args.output_dir
    )

    print(f"\nAll tests complete. Visualizations saved to {args.output_dir}")
    print(f"Patch-unpatch cycle MSE: {error:.6f}")

    # Analyze results
    if error < 0.01:
        print("✅ GOOD: Very low reconstruction error. Patch-unpatch cycle works well.")
    elif error < 0.1:
        print("⚠️ WARNING: Moderate reconstruction error. Check visualization for details.")
    else:
        print("❌ ERROR: High reconstruction error. Patch-unpatch cycle has issues.")
        print("Possible problems:")
        print("1. Mismatch in token ordering between patch embedding and unpatchify")
        print("2. Incorrect reshaping or permutation operations in unpatchify")
        print("3. Issues with the pixel projection layer initialization")
        print("4. Dimension mismatch in spectral or spatial coordinates")


if __name__ == "__main__":
    main()