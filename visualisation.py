import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from hsi_to_rgb import hsi_to_rgb, simple_hsi_to_rgb
import torch
import numpy as np
import matplotlib.pyplot as plt
from hsi_to_rgb import simple_hsi_to_rgb

def visualize_patient_data(patient_data, save_dir='visualizations', show=True):
    """
    Visualize the data for a single patient and save the visualization to a file.
    Includes auxiliary modalities and thickness mask if available.
    Uses HSI to RGB conversion for hyperspectral images.
    """
    hsi = patient_data['hsi']
    aux_data = patient_data['aux_data']
    patient_id = patient_data['patient_id']
    thickness_mask = patient_data.get('thickness_mask', None)

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Determine how many subplots we need based on available modalities
    # HSI + potential auxiliary modalities (AF, IR, thickness) + thickness mask
    num_aux = sum(1 for modality, data in aux_data.items() if data is not None)
    num_plots = 1 + num_aux  # HSI + aux modalities
    if thickness_mask is not None:
        num_plots += 1  # Add one for thickness mask

    # Create figure with appropriate number of subplots
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    fig.suptitle(f"Patient: {patient_id}", fontsize=16)

    # If there's only one subplot, make axes into a list for consistent indexing
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot HSI data converted to RGB
    try:
        # Convert HSI to RGB
        if len(hsi.shape) == 5:  # [B, C, T, H, W]
            rgb_img = simple_hsi_to_rgb(hsi)
            rgb_img = rgb_img[0, 0].cpu().numpy()  # Get first batch, first channel
            rgb_img = np.transpose(rgb_img, (1, 2, 0))  # Change from [3, H, W] to [H, W, 3]
        elif len(hsi.shape) == 4:  # [B, T, H, W] or [C, T, H, W]
            rgb_img = simple_hsi_to_rgb(hsi)
            rgb_img = rgb_img[0].cpu().numpy()  # Get first batch/channel
            rgb_img = np.transpose(rgb_img, (1, 2, 0))  # Change from [3, H, W] to [H, W, 3]
        elif len(hsi.shape) == 3:  # [T, H, W]
            rgb_img = simple_hsi_to_rgb(hsi)
            rgb_img = rgb_img.cpu().numpy()
            rgb_img = np.transpose(rgb_img, (1, 2, 0))  # Change from [3, H, W] to [H, W, 3]
        else:
            raise ValueError(f"Unexpected HSI shape: {hsi.shape}")

        axes[plot_idx].imshow(rgb_img)
        axes[plot_idx].set_title("HSI (RGB Visualization)")
    except Exception as e:
        print(f"Error creating RGB visualization: {e}. Falling back to middle band.")
        try:
            # Determine the shape and select a middle band as fallback
            if len(hsi.shape) == 5:  # [B, C, T, H, W]
                mid_wavelength = hsi.shape[2] // 2
                hsi_slice = hsi[0, 0, mid_wavelength].cpu().numpy()
            elif len(hsi.shape) == 4:  # [B, T, H, W] or [C, T, H, W]
                mid_wavelength = hsi.shape[1] // 2
                hsi_slice = hsi[0, mid_wavelength].cpu().numpy()
            elif len(hsi.shape) == 3:  # [T, H, W]
                mid_wavelength = hsi.shape[0] // 2
                hsi_slice = hsi[mid_wavelength].cpu().numpy()
            else:
                hsi_slice = np.zeros((100, 100))

            axes[plot_idx].imshow(hsi_slice, cmap='viridis')
            axes[plot_idx].set_title(f"HSI (Wavelength {mid_wavelength}) - RGB conversion failed")
        except Exception as e2:
            print(f"Error displaying HSI data: {e2}")
            axes[plot_idx].text(0.5, 0.5, "Error displaying HSI data",
                                ha='center', va='center', transform=axes[plot_idx].transAxes)
            axes[plot_idx].set_title("HSI Data Error")

    axes[plot_idx].axis('off')
    plot_idx += 1

    # Plot all available auxiliary modalities
    for modality, data in aux_data.items():
        if data is not None:
            try:
                # Get the auxiliary data
                aux_data_tensor = data[0] if data.shape[0] == 1 else data

                # Handle different possible shapes
                if len(aux_data_tensor.shape) == 3 and aux_data_tensor.shape[0] == 1:  # [1, H, W]
                    aux_img = aux_data_tensor[0].cpu().numpy()
                elif len(aux_data_tensor.shape) == 3:  # [C, H, W]
                    aux_img = aux_data_tensor[0].cpu().numpy()
                elif len(aux_data_tensor.shape) == 2:  # [H, W]
                    aux_img = aux_data_tensor.cpu().numpy()
                else:
                    aux_img = aux_data_tensor.squeeze().cpu().numpy()

                axes[plot_idx].imshow(aux_img, cmap='gray')
                axes[plot_idx].set_title(f"{modality.upper()}")
            except Exception as e:
                print(f"Error displaying {modality} data: {e}")
                axes[plot_idx].text(0.5, 0.5, f"Error displaying {modality}",
                                    ha='center', va='center', transform=axes[plot_idx].transAxes)
                axes[plot_idx].set_title(f"{modality.upper()} Error")
            axes[plot_idx].axis('off')
            plot_idx += 1

    # Plot thickness mask if available
    if thickness_mask is not None:
        try:
            # Get the thickness mask
            if thickness_mask.shape[0] == 1:  # Single batch
                mask_img = thickness_mask[0, 0].cpu().numpy()
            else:
                mask_img = thickness_mask[0].cpu().numpy()

            # Display the mask
            axes[plot_idx].imshow(mask_img, cmap='gray')
            axes[plot_idx].set_title("Thickness Mask")
        except Exception as e:
            print(f"Error displaying thickness mask: {e}")
            axes[plot_idx].text(0.5, 0.5, "Error displaying thickness mask",
                                ha='center', va='center', transform=axes[plot_idx].transAxes)
            axes[plot_idx].set_title("Thickness Mask Error")
        axes[plot_idx].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save the figure to a file
    save_path = os.path.join(save_dir, f"{patient_id}_visualization.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show the figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_batch(batch, save_dir='visualizations/batch', show=False):
    """
    Visualize a batch of patient data and save the visualizations.
    Includes all auxiliary modalities and thickness mask if available.
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get batch size and patient IDs
    batch_size = batch['hsi'].shape[0]
    patient_ids = batch['patient_id']

    # Process each patient in the batch
    for i in range(batch_size):
        # Create a patient data dict for this sample
        patient_data = {
            'hsi': batch['hsi'][i:i + 1],  # Keep batch dimension
            'aux_data': {key: tensor[i:i + 1] if tensor is not None else None
                         for key, tensor in batch['aux_data'].items()},
            'patient_id': patient_ids[i],
            'batch_idx': batch['batch_idx'][i] if 'batch_idx' in batch else i
        }

        # Add thickness mask if available
        if 'thickness_mask' in batch and batch['thickness_mask'] is not None:
            patient_data['thickness_mask'] = batch['thickness_mask'][i:i + 1]

        # Visualize this patient's data
        patient_save_dir = os.path.join(save_dir, f"patient_{i}")
        visualize_patient_data(patient_data, save_dir=patient_save_dir, show=show)


def visualize_augmentation(original_batch, augmented_batch, save_dir='visualizations/augmentation'):
    """
    Visualize original and augmented data side by side to verify augmentation.

    Args:
        original_batch: Original data batch before augmentation
        augmented_batch: Batch after applying augmentation
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get batch size
    batch_size = min(3, original_batch['hsi'].shape[0])  # Visualize at most 3 samples

    for i in range(batch_size):
        # Extract original and augmented data for this sample
        orig_data = {
            'hsi': original_batch['hsi'][i:i + 1],
            'aux_data': {k: v[i:i + 1] if v is not None else None for k, v in original_batch['aux_data'].items()},
            'patient_id': f"Sample_{i}_Original",
            'batch_idx': i
        }

        if 'thickness_mask' in original_batch and original_batch['thickness_mask'] is not None:
            orig_data['thickness_mask'] = original_batch['thickness_mask'][i:i + 1]

        aug_data = {
            'hsi': augmented_batch['hsi'][i:i + 1],
            'aux_data': {k: v[i:i + 1] if v is not None else None for k, v in augmented_batch['aux_data'].items()},
            'patient_id': f"Sample_{i}_Augmented",
            'batch_idx': i
        }

        if 'thickness_mask' in augmented_batch and augmented_batch['thickness_mask'] is not None:
            aug_data['thickness_mask'] = augmented_batch['thickness_mask'][i:i + 1]

        # Save visualizations
        orig_dir = os.path.join(save_dir, f"sample_{i}_original")
        aug_dir = os.path.join(save_dir, f"sample_{i}_augmented")

        visualize_patient_data(orig_data, save_dir=orig_dir, show=False)
        visualize_patient_data(aug_data, save_dir=aug_dir, show=False)

        print(f"Saved visualization for sample {i}")

    print(f"All visualizations saved to {save_dir}")


def visualize_reconstruction_quality(original, reconstruction, mask, thickness_mask=None, save_path=None):
    """
    Visualize the reconstruction quality with special focus on areas that contribute to the loss.

    Now handles direct pixel reconstruction rather than embedding reconstruction.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Select first batch item for visualization
    orig = original[0].detach().cpu()
    recon = reconstruction[0].detach().cpu()  # Now directly in pixel space

    # Convert HSI to RGB for visualization
    orig_rgb = simple_hsi_to_rgb(orig)
    recon_rgb = simple_hsi_to_rgb(recon)

    # Calculate error map
    error = ((recon - orig) ** 2).mean(dim=(0, 1))  # Mean across channels and spectral dimension
    error = error.numpy()

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Handle different possible shapes of RGB tensor
    # Print shape for debugging
    print(f"Original RGB shape: {orig_rgb.shape}")

    # Convert to numpy and properly reshape for matplotlib
    if len(orig_rgb.shape) == 5:  # [B, C, 3, H, W]
        orig_rgb_np = orig_rgb[0, 0].permute(1, 2, 0).cpu().numpy()
        recon_rgb_np = recon_rgb[0, 0].permute(1, 2, 0).cpu().numpy()
    elif len(orig_rgb.shape) == 4:  # [B, 3, H, W]
        orig_rgb_np = orig_rgb[0].permute(1, 2, 0).cpu().numpy()
        recon_rgb_np = recon_rgb[0].permute(1, 2, 0).cpu().numpy()
    elif len(orig_rgb.shape) == 3 and orig_rgb.shape[0] == 3:  # [3, H, W]
        orig_rgb_np = orig_rgb.permute(1, 2, 0).cpu().numpy()
        recon_rgb_np = recon_rgb.permute(1, 2, 0).cpu().numpy()
    else:  # Already in [H, W, 3] format or other
        orig_rgb_np = orig_rgb.cpu().numpy()
        recon_rgb_np = recon_rgb.cpu().numpy()

    # Ensure we have a proper RGB image with values in [0, 1]
    orig_rgb_np = np.clip(orig_rgb_np, 0, 1)
    recon_rgb_np = np.clip(recon_rgb_np, 0, 1)

    # Plot original RGB
    axes[0, 0].imshow(orig_rgb_np)
    axes[0, 0].set_title("Original HSI (RGB)")
    axes[0, 0].axis('off')

    # Plot reconstructed RGB
    axes[0, 1].imshow(recon_rgb_np)
    axes[0, 1].set_title("Reconstructed HSI (RGB)")
    axes[0, 1].axis('off')

    # Plot error map
    error_img = axes[0, 2].imshow(error, cmap='hot')
    axes[0, 2].set_title("Reconstruction Error")
    axes[0, 2].axis('off')
    fig.colorbar(error_img, ax=axes[0, 2])

    # Convert patch mask to pixel mask for visualization
    if mask is not None:
        B, L = mask.shape
        H, W = original.shape[-2], original.shape[-1]
        patch_h, patch_w = model.patch_size

        # Create a pixel-level mask visualization
        pixel_mask = torch.zeros((H, W), device='cpu')

        # Reshape mask to match spectral and spatial patches
        mask_reshaped = mask[0].reshape(-1, H // patch_h, W // patch_w)

        for t_idx in range(mask_reshaped.shape[0]):
            for h_idx in range(mask_reshaped.shape[1]):
                for w_idx in range(mask_reshaped.shape[2]):
                    h_start = h_idx * patch_h
                    w_start = w_idx * patch_w
                    if mask_reshaped[t_idx, h_idx, w_idx] > 0.5:
                        pixel_mask[h_start:h_start + patch_h, w_start:w_start + patch_w] = 1.0

        # Plot mask
        axes[1, 0].imshow(pixel_mask.numpy(), cmap='gray')
        axes[1, 0].set_title("MAE Mask (White = Masked)")
        axes[1, 0].axis('off')

        # Plot valid regions (thickness mask)
        if thickness_mask is not None:
            thick_mask = thickness_mask[0, 0].detach().cpu().numpy()
            axes[1, 1].imshow(thick_mask, cmap='gray')
            axes[1, 1].set_title("Valid Regions (White = Valid)")
            axes[1, 1].axis('off')

            # Plot combined mask (areas that contribute to loss)
            combined = pixel_mask.numpy() * thick_mask
            axes[1, 2].imshow(combined, cmap='gray')
            axes[1, 2].set_title("Areas Contributing to Loss")
            axes[1, 2].axis('off')
        else:
            axes[1, 1].axis('off')
            axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig





def visualize_pixel_reconstruction(model, original_input, reconstructed_pixels, mask,
                                   thickness_mask=None, save_path=None, num_wavelengths=5):
    """
    Visualize the reconstruction quality in pixel space comparing RGB visualization
    and selected wavelength bands.

    Args:
        model: The model
        original_input: Original HSI data tensor [B, C, T, H, W]
        reconstructed_pixels: Reconstructed pixel values [B, C, T, H, W]
        mask: Binary mask indicating masked tokens [B, L]
        thickness_mask: Optional thickness mask [B, 1, H, W]
        save_path: Path to save the visualization
        num_wavelengths: Number of wavelength bands to visualize (default=5)

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from hsi_to_rgb import simple_hsi_to_rgb
    import gc

    # Use only the first batch item for visualization
    orig_input = original_input[0].detach().cpu()
    reconstructed = reconstructed_pixels[0].detach().cpu()

    # Print statistics for debugging
    print(
        f"Original stats: min={orig_input.min().item():.4f}, max={orig_input.max().item():.4f}, mean={orig_input.mean().item():.4f}, std={orig_input.std().item():.4f}")
    print(
        f"Reconstructed stats: min={reconstructed.min().item():.4f}, max={reconstructed.max().item():.4f}, mean={reconstructed.mean().item():.4f}, std={reconstructed.std().item():.4f}")

    # Convert to RGB for visualization
    if orig_input.dim() >= 3:
        # Extract HSI data, handling different possible shapes
        if orig_input.dim() == 4 and orig_input.shape[0] == 1:  # [1, T, H, W]
            orig_input_hsi = orig_input[0]  # Now [T, H, W]
            recon_input_hsi = reconstructed[0]
        elif orig_input.dim() == 5 and orig_input.shape[0] == 1 and orig_input.shape[1] == 1:  # [1, 1, T, H, W]
            orig_input_hsi = orig_input[0, 0]  # Now [T, H, W]
            recon_input_hsi = reconstructed[0, 0]
        else:
            orig_input_hsi = orig_input
            recon_input_hsi = reconstructed

        # Convert to RGB
        orig_rgb = simple_hsi_to_rgb(orig_input_hsi.unsqueeze(0) if orig_input_hsi.dim() == 3 else orig_input_hsi)
        recon_rgb = simple_hsi_to_rgb(recon_input_hsi.unsqueeze(0) if recon_input_hsi.dim() == 3 else recon_input_hsi)

        # Remove batch dimension if present
        if orig_rgb.dim() == 4 and orig_rgb.shape[0] == 1:
            orig_rgb = orig_rgb[0]
        if recon_rgb.dim() == 4 and recon_rgb.shape[0] == 1:
            recon_rgb = recon_rgb[0]

        # Convert to numpy with proper format for matplotlib (H, W, 3)
        if orig_rgb.shape[0] == 3:  # If [3, H, W]
            orig_rgb_np = orig_rgb.permute(1, 2, 0).numpy()
            recon_rgb_np = recon_rgb.permute(1, 2, 0).numpy()
        else:
            raise ValueError(
                f"Unexpected RGB shape: {orig_rgb.shape}. Expected first dimension to be 3 (RGB channels).")

        # Print RGB stats for debugging
        print(
            f"RGB Original stats: min={orig_rgb_np.min():.4f}, max={orig_rgb_np.max():.4f}, mean={orig_rgb_np.mean():.4f}")
        print(
            f"RGB Reconstructed stats: min={recon_rgb_np.min():.4f}, max={recon_rgb_np.max():.4f}, mean={recon_rgb_np.mean():.4f}")
    else:
        raise ValueError(f"Input dimension too low: {orig_input.dim()}. Expected at least 3 dimensions.")

    # Create a simplified error map
    error_map = ((reconstructed - orig_input) ** 2).mean(dim=(0, 1) if reconstructed.dim() > 3 else 0).numpy()

    # Rest of the function remains the same...
    # (Continue with existing code for wavelength visualization)

    # Clean up memory
    del orig_input, reconstructed
    gc.collect()

    return fig


if __name__ == "__main__":
    import os
    import sys
    import torch
    import numpy as np
    from dataset import PatientDataset, create_patient_dataloader

    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "dummydata"
    print(f"Loading data from: {data_dir}")

    # Create dataloader to get a batch of real data
    dataloader = create_patient_dataloader(
        data_dir,
        batch_size=1,  # Just load one sample for testing
        analysis_dim=500,
        target_bands=30
    )

    # Try to get a batch
    try:
        batch = next(iter(dataloader))
        print(f"Successfully loaded a batch with HSI shape: {batch['hsi'].shape}")

        # Create a simple "reconstructed" version for testing by adding noise
        original = batch['hsi']
        reconstructed = original + 0.05 * torch.randn_like(original)

        # Create a mask similar to what the model would produce (1=masked, 0=visible)
        # Using dimensions from your model: 20x20 spatial patches, 6 spectral chunks = 2400 patches
        total_patches = 2400
        mask = torch.zeros(1, total_patches)
        mask_indices = torch.randperm(total_patches)[:int(0.75 * total_patches)]
        mask[0, mask_indices] = 1.0

        # Get or create thickness mask
        if 'thickness_mask' in batch and batch['thickness_mask'] is not None:
            thickness_mask = batch['thickness_mask']
            print(f"Using real thickness mask with shape: {thickness_mask.shape}")
        else:
            # Create a circular mask
            B, C, T, H, W = original.shape
            thickness_mask = torch.zeros(B, 1, H, W)
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, H),
                torch.linspace(-1, 1, W),
                indexing='ij'
            )
            thickness_mask[0, 0] = ((x ** 2 + y ** 2) < 0.7 ** 2).float()
            print(f"Created synthetic thickness mask with shape: {thickness_mask.shape}")

        # Make sure output directory exists
        os.makedirs("debug_output", exist_ok=True)

        # Test visualization function
        print("Testing visualization function...")
        print(f"Original HSI shape: {original.shape}")

        fig = visualize_reconstruction_quality(
            original=original,
            reconstruction=reconstructed,
            mask=mask,
            thickness_mask=thickness_mask,
            save_path="debug_output/reconstruction_quality_test.png"
        )

        print(f"Visualization saved to debug_output/reconstruction_quality_test.png")

    except Exception as e:
        print(f"Error during visualization test: {e}")
        import traceback

        traceback.print_exc()

