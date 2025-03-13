import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_patient_data(patient_data, save_dir='visualizations', show=True):
    """
    Visualize the data for a single patient and save the visualization to a file.
    Includes thickness mask if available.
    """
    hsi = patient_data['hsi']
    aux_data = patient_data['aux_data']
    patient_id = patient_data['patient_id']
    thickness_mask = patient_data.get('thickness_mask', None)

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Patient: {patient_id}", fontsize=16)

    # Plot HSI data (middle band)
    try:
        # Determine the shape and select a middle band
        if len(hsi.shape) == 5:  # [B, C, T, H, W]
            mid_wavelength = hsi.shape[2] // 2
            hsi_slice = hsi[0, 0, mid_wavelength].numpy()
        elif len(hsi.shape) == 4:  # [B, T, H, W] or [C, T, H, W]
            mid_wavelength = hsi.shape[1] // 2
            hsi_slice = hsi[0, mid_wavelength].numpy()
        elif len(hsi.shape) == 3:  # [T, H, W]
            mid_wavelength = hsi.shape[0] // 2
            hsi_slice = hsi[mid_wavelength].numpy()
        else:
            hsi_slice = np.zeros((100, 100))

        axes[0].imshow(hsi_slice, cmap='viridis')
        axes[0].set_title(f"HSI (Wavelength {mid_wavelength})")
    except Exception as e:
        print(f"Error displaying HSI data: {e}")
        axes[0].text(0.5, 0.5, "Error displaying HSI data",
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("HSI Data Error")
    axes[0].axis('off')

    # Plot auxiliary images if available
    aux_titles = {
        'af': 'Auto Fluorescence (AF)',
        'ir': 'Infrared (IR)',
        'thickness': 'Thickness Map'
    }

    i = 1
    for key, title in aux_titles.items():
        if aux_data[key] is not None:
            try:
                # Get the auxiliary data
                aux_data_tensor = aux_data[key][0] if aux_data[key].shape[0] == 1 else aux_data[key]

                # Handle different possible shapes
                if len(aux_data_tensor.shape) == 3 and aux_data_tensor.shape[0] == 1:  # [1, H, W]
                    aux_img = aux_data_tensor[0].numpy()
                elif len(aux_data_tensor.shape) == 3:  # [C, H, W]
                    aux_img = aux_data_tensor[0].numpy()
                elif len(aux_data_tensor.shape) == 2:  # [H, W]
                    aux_img = aux_data_tensor.numpy()
                else:
                    aux_img = aux_data_tensor.squeeze().numpy()

                axes[i].imshow(aux_img, cmap='gray')
                axes[i].set_title(title)
            except Exception as e:
                print(f"Error displaying {key} data: {e}")
                axes[i].text(0.5, 0.5, f"Error displaying {title}",
                             ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{title} Error")
        else:
            axes[i].text(0.5, 0.5, f"No {title} available",
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"Missing: {title}")
        axes[i].axis('off')
        i += 1

    # Plot thickness mask if available
    if thickness_mask is not None:
        try:
            # Get the thickness mask
            if thickness_mask.shape[0] == 1:  # Single batch
                mask_img = thickness_mask[0, 0].numpy()
            else:
                mask_img = thickness_mask[0].numpy()

            # Display the mask
            axes[3].imshow(mask_img, cmap='gray')
            axes[3].set_title("Thickness Mask")
        except Exception as e:
            print(f"Error displaying thickness mask: {e}")
            axes[3].text(0.5, 0.5, "Error displaying thickness mask",
                         ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title("Thickness Mask Error")
        axes[3].axis('off')

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
    Includes thickness mask if available.
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