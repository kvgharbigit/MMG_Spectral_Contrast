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


def visualize_reconstruction_quality(original, reconstruction, mask, thickness_mask=None,
                                     save_path=None, model=None, num_wavelengths=5):
    """
    Comprehensive visualization of reconstruction quality with advanced masking analysis.

    Args:
        original (torch.Tensor): Original input tensor of shape [B, C, T, H, W]
        reconstruction (torch.Tensor): Reconstructed input tensor of shape [B, C, T, H, W]
        mask (torch.Tensor): Token-level mask of shape [B, L]
        thickness_mask (torch.Tensor, optional): Mask indicating valid image regions
        save_path (str, optional): Path to save the visualization
        model (MultiModalSpectralGPT, optional): Model used for mask conversion
        num_wavelengths (int, optional): Number of wavelength bands to visualize

    Returns:
        matplotlib.figure.Figure: Visualization figure
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from hsi_to_rgb import simple_hsi_to_rgb

    # Input validation
    if model is None:
        raise ValueError("Model must be provided to convert token mask to pixel mask")

    # Use only the first batch item for visualization
    orig_input = original[0].detach().cpu()
    reconstructed = reconstruction[0].detach().cpu()

    # Convert HSI to RGB for visualization
    def convert_to_rgb(hsi_tensor):
        # Ensure tensor is 3D [T, H, W] for visualization
        if hsi_tensor.dim() == 4 and hsi_tensor.shape[0] == 1:
            hsi_data = hsi_tensor[0]
        elif hsi_tensor.dim() == 5 and hsi_tensor.shape[0] == 1 and hsi_tensor.shape[1] == 1:
            hsi_data = hsi_tensor[0, 0]
        else:
            hsi_data = hsi_tensor

        # Ensure tensor is 3D [T, H, W]
        if hsi_data.dim() == 2:
            hsi_data = hsi_data.unsqueeze(0)

        return simple_hsi_to_rgb(hsi_data.unsqueeze(0))

    # Convert to RGB
    orig_rgb = convert_to_rgb(orig_input)
    recon_rgb = convert_to_rgb(reconstructed)

    # Normalize RGB images
    def normalize_rgb(rgb_tensor):
        if rgb_tensor.dim() == 4 and rgb_tensor.shape[0] == 1:
            rgb_img = rgb_tensor[0]
        elif rgb_tensor.dim() == 3:
            rgb_img = rgb_tensor

        # Permute to [H, W, 3] if needed
        if rgb_img.shape[0] == 3:
            rgb_img = rgb_img.permute(1, 2, 0)

        # Clip and normalize
        return np.clip(rgb_img.numpy(), 0, 1)

    orig_rgb_np = normalize_rgb(orig_rgb)
    recon_rgb_np = normalize_rgb(recon_rgb)

    # Create error map
    error_map = ((reconstructed - orig_input) ** 2).mean(dim=(0, 1) if reconstructed.dim() > 3 else 0).numpy()

    # Convert token mask to 3D pixel mask
    pixel_mask_3d = model.token_mask_to_pixel_mask(mask, original.shape)

    # Create masking heatmap
    mask_heatmap = pixel_mask_3d.sum(dim=1)[0].cpu().numpy()

    # Create figure with more detailed layout
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 4)

    # Top row: RGB and error comparisons
    ax_orig_rgb = fig.add_subplot(gs[0, 0])
    ax_recon_rgb = fig.add_subplot(gs[0, 1])
    ax_error = fig.add_subplot(gs[0, 2:])

    # Display RGB images
    ax_orig_rgb.imshow(orig_rgb_np)
    ax_orig_rgb.set_title("Original (RGB)")
    ax_orig_rgb.axis('off')

    ax_recon_rgb.imshow(recon_rgb_np)
    ax_recon_rgb.set_title("Reconstructed (RGB)")
    ax_recon_rgb.axis('off')

    # Display error map
    error_img = ax_error.imshow(error_map, cmap='hot')
    ax_error.set_title("Reconstruction Error")
    ax_error.axis('off')
    fig.colorbar(error_img, ax=ax_error, shrink=0.8)

    # Select wavelength bands to display
    num_bands = orig_input.shape[0] if orig_input.dim() == 3 else orig_input.shape[1]
    band_indices = np.linspace(0, num_bands - 1, min(num_wavelengths, num_bands), dtype=int)

    # Wavelength bands visualization - Original
    ax_orig_bands = [fig.add_subplot(gs[1, i]) for i in range(min(num_wavelengths, 4))]
    for i, band_idx in enumerate(band_indices[:4]):
        # Handle different input dimensions
        if orig_input.dim() == 4 and orig_input.shape[0] == 1:
            band_data = orig_input[0, band_idx].numpy()
        elif orig_input.dim() == 5 and orig_input.shape[0] == 1 and orig_input.shape[1] == 1:
            band_data = orig_input[0, 0, band_idx].numpy()
        elif orig_input.dim() == 3:
            band_data = orig_input[band_idx].numpy()
        else:
            band_data = orig_input[band_idx].numpy()

        ax_orig_bands[i].imshow(band_data, cmap='viridis')
        ax_orig_bands[i].set_title(f"Original λ{band_idx}")
        ax_orig_bands[i].axis('off')

    # Reconstructed wavelength bands
    ax_recon_bands = [fig.add_subplot(gs[2, i]) for i in range(min(num_wavelengths, 4))]
    for i, band_idx in enumerate(band_indices[:4]):
        # Handle different input dimensions
        if reconstructed.dim() == 4 and reconstructed.shape[0] == 1:
            band_data = reconstructed[0, band_idx].numpy()
        elif reconstructed.dim() == 5 and reconstructed.shape[0] == 1 and reconstructed.shape[1] == 1:
            band_data = reconstructed[0, 0, band_idx].numpy()
        elif reconstructed.dim() == 3:
            band_data = reconstructed[band_idx].numpy()
        else:
            band_data = reconstructed[band_idx].numpy()

        ax_recon_bands[i].imshow(band_data, cmap='viridis')
        ax_recon_bands[i].set_title(f"Reconstructed λ{band_idx}")
        ax_recon_bands[i].axis('off')

    # Masking and analysis visualizations
    # Masking Intensity Heatmap
    ax_mask_heatmap = fig.add_subplot(gs[3, 0:2])
    mask_im = ax_mask_heatmap.imshow(mask_heatmap, cmap='hot', interpolation='nearest')
    ax_mask_heatmap.set_title("Masking Intensity Heatmap")
    ax_mask_heatmap.axis('off')
    fig.colorbar(mask_im, ax=ax_mask_heatmap, shrink=0.8, label='Masking Intensity')

    # Thickness Mask (if available)
    ax_thickness_mask = fig.add_subplot(gs[3, 2])
    if thickness_mask is not None:
        # Ensure 2D numpy array
        if thickness_mask.dim() == 4 and thickness_mask.shape[0] == 1 and thickness_mask.shape[1] == 1:
            thickness_mask_np = thickness_mask[0, 0].detach().cpu().numpy()
        elif thickness_mask.dim() == 3 and thickness_mask.shape[0] == 1:
            thickness_mask_np = thickness_mask[0].detach().cpu().numpy()
        elif thickness_mask.dim() == 2:
            thickness_mask_np = thickness_mask.detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected thickness mask shape: {thickness_mask.shape}")

        ax_thickness_mask.imshow(thickness_mask_np, cmap='gray')
        ax_thickness_mask.set_title("Thickness Mask")
    else:
        ax_thickness_mask.text(0.5, 0.5, "No Thickness Mask",
                               horizontalalignment='center',
                               verticalalignment='center')
        ax_thickness_mask.axis('off')

    # Additional Statistics Subplot
    ax_stats = fig.add_subplot(gs[3, 3])
    ax_stats.axis('off')

    # Compute and display statistics
    stats_text = [
        f"Reconstruction Error:",
        f"  Mean: {error_map.mean():.4f}",
        f"  Max: {error_map.max():.4f}",
        f"  Std: {error_map.std():.4f}",
        f"\nMasking Statistics:",
        f"  Masked Pixels: {(mask_heatmap > 0).mean() * 100:.2f}%",
        f"  Max Mask Intensity: {mask_heatmap.max():.4f}",
        f"  Mean Mask Intensity: {mask_heatmap.mean():.4f}"
    ]

    ax_stats.text(0, 1, '\n'.join(stats_text),
                  verticalalignment='top',
                  fontsize=10,
                  fontfamily='monospace')

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    # Close the figure to free memory
    plt.close(fig)

    return fig

def visualize_mask_heatmap(pixel_mask):
    """
    Create a heatmap showing the masking intensity across spatial regions.

    Args:
        pixel_mask (torch.Tensor): 3D pixel mask of shape [B, T, H, W]

    Returns:
        torch.Tensor: 2D heatmap of masking intensity with shape [H, W]
    """
    # Ensure we're working with the first batch item
    if pixel_mask.dim() == 4 and pixel_mask.shape[0] == 1:
        pixel_mask = pixel_mask[0]

    # Sum across spectral dimension to get spatial masking intensity
    # This creates a 2D heatmap where higher values indicate more masking
    spatial_mask_intensity = pixel_mask.sum(dim=0)

    return spatial_mask_intensity

def visualize_pixel_reconstruction(model, original_input, reconstructed_pixels, mask, output=None,
                                   thickness_mask=None, save_path=None, num_wavelengths=5,
                                   add_numerical_viz=True, sample_size=8):
    """
    Visualize the reconstruction quality with diversity analysis using the original patches from the model output.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from hsi_to_rgb import simple_hsi_to_rgb
    import gc
    import pandas as pd
    import os
    import seaborn as sns

    # Check inputs
    if original_input is None or reconstructed_pixels is None:
        print("Error: Input tensors cannot be None")
        return None

    # Use only the first batch item for visualization
    orig_input = original_input[0].detach().cpu()
    reconstructed = reconstructed_pixels[0].detach().cpu()

    # Validate input shapes
    if orig_input.dim() < 3:
        raise ValueError(f"Input dimension too low: {orig_input.dim()}. Expected at least 3 dimensions.")

    # Normalize input and reconstructed tensors
    input_min, input_max = orig_input.min(), orig_input.max()
    recon_min, recon_max = reconstructed.min(), reconstructed.max()

    print(f"Original input range: [{input_min.item():.4f}, {input_max.item():.4f}]")
    print(f"Reconstructed input range: [{recon_min.item():.4f}, {recon_max.item():.4f}]")

    # Convert HSI to RGB
    def convert_to_rgb(hsi_tensor):
        if hsi_tensor.dim() == 4 and hsi_tensor.shape[0] == 1:  # [1, T, H, W]
            hsi_data = hsi_tensor[0]
        elif hsi_tensor.dim() == 5 and hsi_tensor.shape[0] == 1 and hsi_tensor.shape[1] == 1:  # [1, 1, T, H, W]
            hsi_data = hsi_tensor[0, 0]
        else:
            hsi_data = hsi_tensor

        # Ensure tensor is 3D [T, H, W]
        if hsi_data.dim() == 2:
            hsi_data = hsi_data.unsqueeze(0)

        return simple_hsi_to_rgb(hsi_data.unsqueeze(0))

    # Convert to RGB
    orig_rgb = convert_to_rgb(orig_input)
    recon_rgb = convert_to_rgb(reconstructed)

    # Normalize RGB images
    def normalize_rgb(rgb_tensor):
        # Handle different possible shapes
        if rgb_tensor.dim() == 4 and rgb_tensor.shape[0] == 1:
            rgb_img = rgb_tensor[0]
        elif rgb_tensor.dim() == 3:
            rgb_img = rgb_tensor

        # Permute to [H, W, 3] if needed
        if rgb_img.shape[0] == 3:
            rgb_img = rgb_img.permute(1, 2, 0)

        # Clip and normalize
        return np.clip(rgb_img.numpy(), 0, 1)

    orig_rgb_np = normalize_rgb(orig_rgb)
    recon_rgb_np = normalize_rgb(recon_rgb)

    # Create error map
    error_map = ((reconstructed - orig_input) ** 2).mean(dim=(0, 1) if reconstructed.dim() > 3 else 0).numpy()

    # Determine figure layout
    num_rows = 4  # Base rows
    if add_numerical_viz:
        num_rows += 1
    num_rows += 1  # For masks

    # Create figure
    fig = plt.figure(figsize=(18, 4 * num_rows))
    gs = plt.GridSpec(num_rows, 4, figure=fig)

    # Top row - RGB and error comparisons
    ax_orig_rgb = fig.add_subplot(gs[0, 0])
    ax_recon_rgb = fig.add_subplot(gs[0, 1])
    ax_error = fig.add_subplot(gs[0, 2:])

    # Display RGB images
    ax_orig_rgb.imshow(orig_rgb_np)
    ax_orig_rgb.set_title("Original (RGB)")
    ax_orig_rgb.axis('off')

    ax_recon_rgb.imshow(recon_rgb_np)
    ax_recon_rgb.set_title("Reconstructed (RGB)")
    ax_recon_rgb.axis('off')

    # Display error map
    error_img = ax_error.imshow(error_map, cmap='hot')
    ax_error.set_title("Reconstruction Error")
    ax_error.axis('off')
    fig.colorbar(error_img, ax=ax_error)

    # Select wavelength bands to display
    num_bands = orig_input.shape[0] if orig_input.dim() == 3 else orig_input.shape[1]
    band_indices = np.linspace(0, num_bands - 1, min(num_wavelengths, num_bands), dtype=int)

    # Wavelength bands visualization
    wavelength_rows = [1, 2]  # Rows for original and reconstructed wavelengths
    for row_idx, input_type in zip(wavelength_rows, [orig_input, reconstructed]):
        for i, band_idx in enumerate(band_indices[:4]):  # Show up to 4 bands
            ax = fig.add_subplot(gs[row_idx, i])
            # Handle different input dimensions
            if input_type.dim() == 4 and input_type.shape[0] == 1:
                band_data = input_type[0, band_idx].numpy()
            elif input_type.dim() == 5 and input_type.shape[0] == 1 and input_type.shape[1] == 1:
                band_data = input_type[0, 0, band_idx].numpy()
            elif input_type.dim() == 3:
                band_data = input_type[band_idx].numpy()
            else:
                band_data = input_type[band_idx].numpy()

            ax.imshow(band_data, cmap='viridis')
            ax.set_title(f"{'Original' if row_idx == 1 else 'Recon'} λ{band_idx}")
            ax.axis('off')

    # Mask row
    mask_row = num_rows - 3  # Row before numerical/diversity analysis

    # If using the 3D pixel mask
    if mask is not None:
        try:
            # Convert to 3D pixel mask
            pixel_mask_3d = model.token_mask_to_pixel_mask(mask, original_input.shape)

            # Create masking heatmap
            mask_heatmap = visualize_mask_heatmap(pixel_mask_3d)

            # Display masking heatmap
            ax_mask_heatmap = fig.add_subplot(gs[mask_row, 2])
            im = ax_mask_heatmap.imshow(mask_heatmap, cmap='hot', interpolation='nearest')
            ax_mask_heatmap.set_title("Masking Intensity Heatmap")
            ax_mask_heatmap.axis('off')

            # Add colorbar
            fig.colorbar(im, ax=ax_mask_heatmap,
                         fraction=0.046, pad=0.04,
                         label='Masking Intensity')

            # Print some statistics about the masking
            print(f"Masking Heatmap Statistics:")
            print(f"  Mean Masking Intensity: {mask_heatmap.mean().item():.4f}")
            print(f"  Max Masking Intensity: {mask_heatmap.max().item():.4f}")
            print(f"  Percentage of Pixels Masked: {(mask_heatmap > 0).float().mean().item() * 100:.2f}%")

        except Exception as e:
            print(f"Error creating masking heatmap: {e}")
            import traceback
            traceback.print_exc()

    # Visualize thickness mask
    if thickness_mask is not None:
        try:
            # Ensure 2D numpy array
            if thickness_mask.dim() == 4 and thickness_mask.shape[0] == 1 and thickness_mask.shape[1] == 1:
                thickness_mask_np = thickness_mask[0, 0].detach().cpu().numpy()
            elif thickness_mask.dim() == 3 and thickness_mask.shape[0] == 1:
                thickness_mask_np = thickness_mask[0].detach().cpu().numpy()
            elif thickness_mask.dim() == 2:
                thickness_mask_np = thickness_mask.detach().cpu().numpy()
            else:
                raise ValueError(f"Unexpected thickness mask shape: {thickness_mask.shape}")

            # Display thickness mask
            ax_thickness_mask = fig.add_subplot(gs[mask_row, 1])
            ax_thickness_mask.imshow(thickness_mask_np, cmap='gray')
            ax_thickness_mask.set_title("Thickness Mask (White = Valid)")
            ax_thickness_mask.axis('off')

            # Combined mask if both masks are available
            if mask is not None:
                combined_mask = pixel_mask.numpy() * thickness_mask_np
                ax_combined_mask = fig.add_subplot(gs[mask_row, 2])
                ax_combined_mask.imshow(combined_mask, cmap='gray')
                ax_combined_mask.set_title("Combined Mask (Contributing Areas)")
                ax_combined_mask.axis('off')
        except Exception as e:
            print(f"Error processing thickness mask: {e}")

    # Save the figure
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    # Close the figure to free memory
    plt.close(fig)

    return fig


if __name__ == "__main__":
    import os
    import sys
    import torch
    import numpy as np
    from dataset import PatientDataset, create_patient_dataloader
    # Import the model
    from MultiModalSpectralGPT import MultiModalSpectralGPT

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
        # Assuming your model uses 500x500 image with 25x25 patches
        spatial_patches_h = 500 // 25  # 20 patches
        spatial_patches_w = 500 // 25  # 20 patches
        spectral_patches = 30 // 5  # 6 spectral patches
        total_patches = spatial_patches_h * spatial_patches_w * spectral_patches

        mask = torch.zeros(1, total_patches)
        mask_indices = torch.randperm(total_patches)[:int(0.75 * total_patches)]
        mask[0, mask_indices] = 1.0  # Mask 75% of patches

        print(f"Total patches: {total_patches}")
        print(f"Mask shape: {mask.shape}")
        print(f"Percentage masked: {(mask[0] == 1).float().mean().item() * 100:.2f}%")

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

        # Initialize the model with parameters that match your dataset and masking approach
        model = MultiModalSpectralGPT(
            analysis_dim=500,
            patch_size=25,
            embed_dim=768,
            depth=12,
            num_heads=12,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            num_frames=30,
            t_patch_size=5,
            in_chans=1,
            aux_chans=1,
            aux_embed_dim=64,
            temperature=0.07,
            mask_ratio=0.75,
            contrastive_mode='combined',
            use_thickness_mask=True
        )

        # Test visualization function
        print("Testing visualization function...")
        print(f"Original HSI shape: {original.shape}")

        fig = visualize_reconstruction_quality(
            original=original,
            reconstruction=reconstructed,
            mask=mask,
            thickness_mask=thickness_mask,
            save_path="debug_output/reconstruction_quality_test.png",
            model=model  # Use the real model
        )

        print(f"Visualization saved to debug_output/reconstruction_quality_test.png")

    except Exception as e:
        print(f"Error during visualization test: {e}")
        import traceback
        traceback.print_exc()