import os
import numpy as np
import matplotlib.pyplot as plt
from hsi_to_rgb import hsi_to_rgb, simple_hsi_to_rgb


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


if __name__ == "__main__":
    import sys
    import torch
    from dataset import create_patient_dataloader, PatientDataset, MultiModalTransforms

    # Test code for augmentation visualization
    try:
        print("Testing visualization with augmentation...")

        # Get data directory from command line or use default
        data_dir = sys.argv[1] if len(sys.argv) > 1 else "dummydata"
        print(f"Loading data from: {data_dir}")

        # Create dataset and dataloader
        batch_size = 4
        dataloader = create_patient_dataloader(
            data_dir,
            batch_size=batch_size,
            augment=False  # No augmentation in the dataloader
        )

        # Get a batch
        batch = next(iter(dataloader))

        # Create a copy for augmentation
        augmented_batch = {
            'hsi': batch['hsi'].clone(),
            'aux_data': {k: v.clone() if v is not None else None for k, v in batch['aux_data'].items()},
            'batch_idx': batch['batch_idx'].clone(),
            'patient_id': batch['patient_id']
        }
        if 'thickness_mask' in batch and batch['thickness_mask'] is not None:
            augmented_batch['thickness_mask'] = batch['thickness_mask'].clone()

        # In visualisation.py around line 235-236, just before the transform call:
        print("\n----- DEBUG: BEFORE AUGMENTATION -----")
        print(f"HSI data shape: {augmented_batch['hsi'].shape}")
        for k, v in augmented_batch['aux_data'].items():
            if v is not None:
                print(f"{k} data shape: {v.shape}")
            else:
                print(f"{k} data is None")
        print(f"Thickness mask: {type(augmented_batch.get('thickness_mask'))}")
        if augmented_batch.get('thickness_mask') is not None:
            print(f"Thickness mask shape: {augmented_batch['thickness_mask'].shape}")
        print("---------------------------------------\n")


        # Apply augmentation
        transform = MultiModalTransforms(prob=1.0)  # Force augmentation to occur
        augmented_batch['hsi'], augmented_batch['aux_data'], augmented_batch['thickness_mask'] = transform(
            augmented_batch['hsi'], augmented_batch['aux_data'],
            augmented_batch.get('thickness_mask', None)
        )

        # Visualize
        visualize_augmentation(batch, augmented_batch)

        print("Augmentation visualization complete!")

    except Exception as e:
        import traceback

        print(f"Error testing augmentation visualization: {e}")
        traceback.print_exc()