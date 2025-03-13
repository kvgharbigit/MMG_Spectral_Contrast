import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from timm.layers import to_2tuple
from MultiModalSpectralGPT import MultiModalSpectralGPT
import glob
from PIL import Image
import tifffile


def create_data_with_real_thickness(batch_size=2):
    """Create test data using a real thickness image from file."""
    # Create HSI data: [B, C, T, H, W]
    hsi_data = torch.randn(batch_size, 1, 30, 500, 500)

    # Create auxiliary data with the same spatial size
    aux_data = {
        'ir': torch.randn(batch_size, 1, 500, 500),
        'af': torch.randn(batch_size, 1, 500, 500),
        'thickness': torch.zeros(batch_size, 1, 500, 500)  # Will be filled with real data
    }

    # Create batch indices for contrastive learning
    batch_indices = torch.arange(batch_size)

    # Search for thickness tiff files in the dummydata directory and all subdirectories
    thickness_files = []

    # Use os.walk to search recursively through all subdirectories
    for root, dirs, files in os.walk('dummydata'):
        for file in files:
            # Check if the file is a TIFF and has 'thickness' in the name
            if 'thickness' in file.lower() and (file.lower().endswith('.tif') or file.lower().endswith('.tiff')):
                thickness_files.append(os.path.join(root, file))

    # Print the found files for debugging
    if thickness_files:
        print(f"Found {len(thickness_files)} thickness files:")
        for i, file in enumerate(thickness_files[:5]):  # Show at most 5 files
            print(f"  {i + 1}. {file}")
        if len(thickness_files) > 5:
            print(f"  ...and {len(thickness_files) - 5} more files")

    if not thickness_files:
        print("No thickness TIFF files found in dummydata directory!")
        print("Creating dummy circular masks instead.")
        # Fall back to creating circular masks
        for b in range(batch_size):
            # Create a grid of coordinates
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, 500),
                torch.linspace(-1, 1, 500),
                indexing='ij'
            )
            # Vary the circle radius for each batch
            radius = 0.7 if b == 0 else 0.5
            mask = ((x ** 2 + y ** 2) < radius ** 2).float()

            # Apply the mask to random thickness data
            random_data = torch.randn(500, 500)
            aux_data['thickness'][b, 0] = random_data * mask
    else:
        # Use the first thickness file found
        thickness_file = thickness_files[0]
        print(f"Using thickness file: {thickness_file}")

        try:
            # Try to open with tifffile first (handles more TIFF variants)
            thickness_img = tifffile.imread(thickness_file)

            # Normalize to [0, 1] float range
            if thickness_img.dtype == np.uint8:
                thickness_img = thickness_img.astype(np.float32) / 255.0
            elif thickness_img.dtype == np.uint16:
                thickness_img = thickness_img.astype(np.float32) / 65535.0
            else:
                # Already floating point, just ensure it's in [0, 1]
                thickness_img = thickness_img.astype(np.float32)
                if thickness_img.max() > 1.0:
                    thickness_img = thickness_img / thickness_img.max()

        except Exception as e:
            print(f"Error reading with tifffile: {e}")
            # Fall back to PIL if tifffile fails
            try:
                thickness_img = np.array(Image.open(thickness_file))
                # Convert to float and normalize
                thickness_img = thickness_img.astype(np.float32) / 255.0
            except Exception as e:
                print(f"Error reading with PIL: {e}")
                print("Falling back to dummy data.")
                # Create a dummy circular mask
                y, x = torch.meshgrid(
                    torch.linspace(-1, 1, 500),
                    torch.linspace(-1, 1, 500),
                    indexing='ij'
                )
                thickness_img = ((x.numpy() ** 2 + y.numpy() ** 2) < 0.6 ** 2).astype(np.float32)

        # Resize to 500x500 if needed
        if thickness_img.shape[0] != 500 or thickness_img.shape[1] != 500:
            print(f"Resizing thickness image from {thickness_img.shape} to (500, 500)")
            try:
                from skimage.transform import resize
                thickness_img = resize(thickness_img, (500, 500), preserve_range=True)
            except ImportError:
                print("scikit-image not available for resizing, using basic interpolation")
                # Basic resize - not ideal but a fallback
                h, w = thickness_img.shape[:2]
                h_indices = np.linspace(0, h - 1, 500).astype(int)
                w_indices = np.linspace(0, w - 1, 500).astype(int)
                thickness_img = thickness_img[h_indices[:, np.newaxis], w_indices]

        # Handle multi-channel images
        if len(thickness_img.shape) > 2 and thickness_img.shape[2] > 1:
            print(f"Taking first channel of multi-channel thickness image (shape: {thickness_img.shape})")
            thickness_img = thickness_img[:, :, 0]

        # Ensure the array is 2D
        thickness_img = thickness_img.squeeze()

        # Convert to torch tensor and add to data
        thickness_tensor = torch.tensor(thickness_img, dtype=torch.float32)

        # Use the same image for all batch items
        for b in range(batch_size):
            aux_data['thickness'][b, 0] = thickness_tensor

    return hsi_data, aux_data, batch_indices


def test_loss_masking_and_contrastive_modes():
    """Test both contrastive modes and verify loss masking is working using a real thickness image."""
    print("\n===== TESTING LOSS MASKING WITH REAL THICKNESS IMAGE =====")

    # Create output directory for visualizations
    os.makedirs("test_visualizations", exist_ok=True)

    # Check if dummydata directory exists, create it if it doesn't
    if not os.path.exists("dummydata"):
        os.makedirs("dummydata", exist_ok=True)
        print("Created dummydata directory. Please place thickness TIFF files there.")

    # Basic model parameters
    analysis_dim = 500
    patch_size = (25, 25)
    t_patch_size = 5
    embed_dim = 768
    batch_size = 2

    # Generate data with real thickness image if available
    hsi_data, aux_data, batch_indices = create_data_with_real_thickness(batch_size=batch_size)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Move data to device
    hsi_data = hsi_data.to(device)
    aux_data = {k: v.to(device) for k, v in aux_data.items()}
    batch_indices = batch_indices.to(device)

    # Visualize the thickness images
    try:
        plt.figure(figsize=(10, 5))
        for b in range(min(batch_size, 2)):
            plt.subplot(1, 2, b + 1)
            plt.imshow(aux_data['thickness'][b, 0].cpu().numpy(), cmap='gray')
            plt.title(f"Thickness Image (Batch {b})")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig("test_visualizations/real_thickness_images.png")
        plt.close()
    except Exception as e:
        print(f"Error visualizing thickness images: {e}")

    # Test global contrastive mode
    global_model = None
    spatial_model = None
    global_output = None
    spatial_output = None

    try:
        print("\n==== Testing Global Contrastive Mode ====")
        global_model = MultiModalSpectralGPT(
            analysis_dim=analysis_dim,
            patch_size=patch_size,
            embed_dim=embed_dim,
            t_patch_size=t_patch_size,
            in_chans=1,
            aux_chans=1,
            contrastive_mode='global'
        ).to(device)

        # First, check the spatial registration function
        hsi_registered, aux_registered, thickness_mask = global_model.spatial_registration(hsi_data, aux_data)

        # Verify thickness mask is created
        if thickness_mask is not None:
            print(f"Thickness mask shape: {thickness_mask.shape}")
            print(f"Thickness mask min/max: {thickness_mask.min().item():.4f}/{thickness_mask.max().item():.4f}")

            # Visualize the detected mask
            plt.figure(figsize=(10, 5))
            for b in range(min(batch_size, 2)):
                plt.subplot(1, 2, b + 1)
                plt.imshow(thickness_mask[b, 0].cpu().numpy(), cmap='gray')
                plt.title(f"Detected Thickness Mask (Batch {b})")
                plt.axis('off')
            plt.tight_layout()
            plt.savefig("test_visualizations/detected_thickness_masks.png")
            plt.close()
        else:
            print("No thickness mask was created!")

        # Now run forward pass
        global_output = global_model(hsi_data, aux_data, batch_indices)

        # Check if thickness mask is in the output
        if 'thickness_mask' in global_output:
            print("✓ Thickness mask passed through global model")
        else:
            print("✗ Thickness mask not in global model output!")

        # Visualize patch-level mask if we have direct access
        if hasattr(global_model, 'create_patch_mask_from_pixel_mask') and thickness_mask is not None:
            # Generate patch mask from pixel mask
            patch_mask = global_model.create_patch_mask_from_pixel_mask(thickness_mask)

            # Print shape info for debugging
            print(f"Patch mask shape: {patch_mask.shape}")

            # Get model structure details
            spatial_patches_h = analysis_dim // patch_size[0]
            spatial_patches_w = analysis_dim // patch_size[1]
            spectral_patches = global_model.spectral_patches
            print(
                f"Expected patch structure: spatial={spatial_patches_h}×{spatial_patches_w}, spectral={spectral_patches}")

            # Visualize the patch-level mask
            plt.figure(figsize=(15, 5))

            # Original pixel-level mask
            plt.subplot(1, 3, 1)
            plt.imshow(thickness_mask[0, 0].cpu().numpy(), cmap='gray')
            plt.title("Original Thickness Mask")
            plt.axis('off')

            # Patch-level mask - try to reshape safely
            plt.subplot(1, 3, 2)
            try:
                # Check if the shapes align with expectations
                expected_patches = spatial_patches_h * spatial_patches_w * spectral_patches
                if patch_mask[0].numel() == expected_patches:
                    # Safely reshape
                    spatial_patch_mask = patch_mask[0].reshape(spatial_patches_h * spatial_patches_w, spectral_patches)
                    spatial_patch_mask = spatial_patch_mask.mean(dim=1)
                    spatial_patch_mask = spatial_patch_mask.reshape(spatial_patches_h, spatial_patches_w)
                else:
                    # Fallback: reshape to square for visualization
                    side_len = int(np.sqrt(patch_mask[0].numel() / spectral_patches))
                    spatial_patch_mask = patch_mask[0].reshape(-1, spectral_patches)
                    spatial_patch_mask = spatial_patch_mask.mean(dim=1)
                    spatial_patch_mask = spatial_patch_mask.reshape(side_len, side_len)
                    print(
                        f"Warning: Patch mask size doesn't match expectation. Using {side_len}×{side_len} for visualization.")

                plt.imshow(spatial_patch_mask.cpu().numpy(), cmap='gray')
                plt.title("Patch-Level Mask for Loss")
            except Exception as e:
                print(f"Error reshaping patch mask: {e}")
                # Simpler fallback
                plt.imshow(patch_mask[0][:400].reshape(20, 20).cpu().numpy(), cmap='gray')
                plt.title("Patch-Level Mask (Partial)")
            plt.axis('off')

            # MAE mask
            plt.subplot(1, 3, 3)
            try:
                # Get the actual size of the mask
                mask_size = global_output['mask'][0].numel()
                print(f"MAE mask shape: {global_output['mask'].shape}, total elements: {mask_size}")

                # Try to reshape to a sensible visualization
                side_len = int(np.sqrt(mask_size))
                plt.imshow(global_output['mask'][0].reshape(side_len, side_len).cpu().numpy(), cmap='gray')
                plt.title(f"MAE Mask ({side_len}×{side_len})")
            except:
                # Fallback without reshaping
                plt.imshow(global_output['mask'][0].unsqueeze(0).cpu().numpy(), cmap='gray')
                plt.title("MAE Mask (1D)")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig("test_visualizations/mask_comparison_global.png")
            plt.close()

        print(f"Global model forward pass successful!")
        print(f"Reconstruction loss: {global_output['loss_recon'].item():.4f}")
        print(f"Contrastive loss: {global_output['loss_contrast'].item():.4f}")
        print(f"Total loss: {global_output['loss'].item():.4f}")

    except Exception as e:
        print(f"Error in global contrastive mode test: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n==== Testing Spatial Contrastive Mode ====")
        spatial_model = MultiModalSpectralGPT(
            analysis_dim=analysis_dim,
            patch_size=patch_size,
            embed_dim=embed_dim,
            t_patch_size=t_patch_size,
            in_chans=1,
            aux_chans=1,
            contrastive_mode='spatial'
        ).to(device)

        # Test forward pass with spatial mode
        spatial_output = spatial_model(hsi_data, aux_data, batch_indices)

        # Check if thickness mask is in the output
        if 'thickness_mask' in spatial_output:
            print("✓ Thickness mask passed through spatial model")
        else:
            print("✗ Thickness mask not in spatial model output!")

        print(f"Spatial model forward pass successful!")
        print(f"Reconstruction loss: {spatial_output['loss_recon'].item():.4f}")
        print(f"Contrastive loss: {spatial_output['loss_contrast'].item():.4f}")
        print(f"Total loss: {spatial_output['loss'].item():.4f}")

    except Exception as e:
        print(f"Error in spatial contrastive mode test: {e}")
        import traceback
        traceback.print_exc()

    # Compare losses if both models ran successfully
    if global_output is not None and spatial_output is not None:
        try:
            print("\n==== Comparing Losses Between Modes ====")
            print(f"Global reconstruction loss: {global_output['loss_recon'].item():.4f}")
            print(f"Spatial reconstruction loss: {spatial_output['loss_recon'].item():.4f}")
            print(f"Global contrastive loss: {global_output['loss_contrast'].item():.4f}")
            print(f"Spatial contrastive loss: {spatial_output['loss_contrast'].item():.4f}")
            print(f"Global total loss: {global_output['loss'].item():.4f}")
            print(f"Spatial total loss: {spatial_output['loss'].item():.4f}")
        except Exception as e:
            print(f"Error comparing losses: {e}")

    # Add this to your test function
    if global_model is not None:
        print("\n==== Testing Global Mode With and Without Masking ====")

        # Run with original thickness data (with mask areas)
        global_output_with_mask = global_model(hsi_data, aux_data, batch_indices)

        # Run with the same data but nullify the thickness mask effect
        # (temporarily modify the contrastive_loss_global method or make a copy)
        # This is a bit hacky but avoids modifying the model architecture
        original_method = global_model.contrastive_loss_global

        # Create a wrapper that ignores the thickness mask
        def no_mask_global(model, features, embeddings, batch_idx, thickness_mask=None):
            return original_method(features, embeddings, batch_idx, None)

        # Temporarily replace method
        global_model.contrastive_loss_global = lambda features, embeddings, batch_idx, thickness_mask: no_mask_global(
            global_model, features, embeddings, batch_idx)

        # Run without using the mask
        global_output_no_mask = global_model(hsi_data, aux_data, batch_indices)

        # Restore original method
        global_model.contrastive_loss_global = original_method

        # Compare results
        print(f"Global contrastive loss with rim masking: {global_output_with_mask['loss_contrast'].item():.4f}")
        print(f"Global contrastive loss without rim masking: {global_output_no_mask['loss_contrast'].item():.4f}")
        print(
            f"Difference: {(global_output_with_mask['loss_contrast'] - global_output_no_mask['loss_contrast']).item():.4f}")

    # Test without mask
    if global_model is not None and spatial_model is not None:
        try:
            print("\n==== Testing Without Thickness Mask ====")

            # Create data without masks
            hsi_data_no_mask, aux_data_no_mask, batch_indices = create_data_with_real_thickness(batch_size=batch_size)

            # Override the thickness maps with all ones (no mask)
            for b in range(batch_size):
                aux_data_no_mask['thickness'][b, 0] = torch.ones_like(aux_data_no_mask['thickness'][b, 0])

            # Move to device
            hsi_data_no_mask = hsi_data_no_mask.to(device)
            aux_data_no_mask = {k: v.to(device) for k, v in aux_data_no_mask.items()}

            # Run models
            with torch.no_grad():
                global_output_no_mask = global_model(hsi_data_no_mask, aux_data_no_mask, batch_indices)
                spatial_output_no_mask = spatial_model(hsi_data_no_mask, aux_data_no_mask, batch_indices)

            # Compare losses
            print("\n==== Loss Comparison With vs Without Masking ====")
            print(f"Global reconstruction loss with mask:    {global_output['loss_recon'].item():.4f}")
            print(f"Global reconstruction loss without mask: {global_output_no_mask['loss_recon'].item():.4f}")
            print(f"Spatial reconstruction loss with mask:    {spatial_output['loss_recon'].item():.4f}")
            print(f"Spatial reconstruction loss without mask: {spatial_output_no_mask['loss_recon'].item():.4f}")

            # Calculate ratios
            global_ratio = global_output['loss_recon'].item() / global_output_no_mask['loss_recon'].item()
            spatial_ratio = spatial_output['loss_recon'].item() / spatial_output_no_mask['loss_recon'].item()

            print(f"\nGlobal model loss ratio (masked/unmasked): {global_ratio:.4f}")
            print(f"Spatial model loss ratio (masked/unmasked): {spatial_ratio:.4f}")

            if global_ratio < 1.0 and spatial_ratio < 1.0:
                print("✓ Masking appears to be working correctly (masked loss is lower)")
            else:
                print("✗ Masking may not be working as expected (masked loss is not lower)")

        except Exception as e:
            print(f"Error in unmasked test: {e}")
            import traceback
            traceback.print_exc()

    print("\n===== TEST COMPLETE =====")
    print("Visualizations saved to 'test_visualizations/' directory")


if __name__ == '__main__':
    # Run the test
    test_loss_masking_and_contrastive_modes()