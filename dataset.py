import os
import glob
import h5py
import torch
import numpy as np
from PIL import Image
import tifffile
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class PatientDataset(Dataset):
    """
    Dataset for loading patient data from a parent directory.
    A patient file is identified by the presence of a .h5 file directly inside it.

    For each patient, it loads:
    - HSI data (.h5 file)
    - FAF image (tiff/tif file with 'FAF' in the filename)
    - IR image (tiff/tif file with 'IR' in the filename)
    - Thickness map (tiff/tif file with 'thickness' in the filename)

    Note: Some patients may be missing auxiliary files.
    """

    def __init__(self, parent_dir, transform=None):
        """
        Initialize the dataset.

        Args:
            parent_dir (str): Path to the parent directory containing patient files
            transform (callable, optional): Optional transform to be applied to the data
        """
        self.parent_dir = parent_dir
        self.transform = transform
        self.patient_dirs = []

        # Find all patient directories (containing .h5 files)
        self._find_patient_dirs()

        print(f"Found {len(self.patient_dirs)} patient directories")

    def _find_patient_dirs(self):
        """Find all directories containing .h5 files at any level of nesting."""
        for root, dirs, files in os.walk(self.parent_dir):
            if any(file.endswith('.h5') for file in files):
                self.patient_dirs.append(root)

    def _load_hsi(self, patient_dir):
        """Load HSI data from .h5 file."""
        h5_files = glob.glob(os.path.join(patient_dir, '*.h5'))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 file found in {patient_dir}")

        h5_path = h5_files[0]
        try:
            with h5py.File(h5_path, 'r') as f:
                # Print file structure for debugging
                print(f"H5 file structure in {h5_path}:")
                print(f"Keys at root level: {list(f.keys())}")

                # Try to find the HSI data - specifically looking for the 'Cube' structure
                # that appears in your files
                if 'Cube' in f:
                    if 'Images' in f['Cube']:
                        # This matches your specific file structure
                        hsi_data = f['Cube']['Images'][:]
                        print(f"Found dataset in Cube/Images with shape {hsi_data.shape}")
                    elif isinstance(f['Cube'], h5py.Dataset):
                        hsi_data = f['Cube'][:]
                        print(f"Found dataset Cube with shape {hsi_data.shape}")
                    else:
                        # If 'Cube' is a group but doesn't have 'Images', check its contents
                        cube_keys = list(f['Cube'].keys())
                        print(f"Cube contains keys: {cube_keys}")
                        for key in cube_keys:
                            if isinstance(f['Cube'][key], h5py.Dataset):
                                hsi_data = f['Cube'][key][:]
                                print(f"Found dataset in Cube/{key} with shape {hsi_data.shape}")
                                break
                else:
                    # If no 'Cube' key, look for other standard keys
                    for key in ['hsi', 'data', 'image', 'hyperspectral']:
                        if key in f and isinstance(f[key], h5py.Dataset):
                            hsi_data = f[key][:]
                            print(f"Found dataset in {key} with shape {hsi_data.shape}")
                            break
                    else:
                        # If still not found, use the first dataset we can find
                        for key in f.keys():
                            if isinstance(f[key], h5py.Dataset):
                                hsi_data = f[key][:]
                                print(f"Found dataset in {key} with shape {hsi_data.shape}")
                                break

                if 'hsi_data' not in locals():
                    print(f"Could not find HSI data in {h5_path}")
                    return torch.zeros((1, 30, 500, 500), dtype=torch.float32).unsqueeze(
                        1)  # Return [1, 1, 30, 500, 500]

                print(f"Raw data shape: {hsi_data.shape}, dtype: {hsi_data.dtype}")

                # Based on your file structure, the data appears to be in (T, H, W) format
                # with shape (91, 1536, 1536)
                if len(hsi_data.shape) == 3 and hsi_data.shape[0] < 100 and hsi_data.shape[1] > 500:
                    # This matches your specific format (91, 1536, 1536)
                    # Reshape to [1, C, T, H, W] format for model (add batch and channel dims)
                    print(f"Reshaping from [T, H, W] to [1, C, T, H, W]")
                    hsi_tensor = torch.from_numpy(hsi_data).float()
                    hsi_tensor = hsi_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                elif len(hsi_data.shape) == 3:
                    # For other 3D arrays, try to determine if spectral dimension is first or last
                    if hsi_data.shape[0] < hsi_data.shape[1] and hsi_data.shape[0] < hsi_data.shape[2]:
                        # Likely (T, H, W)
                        print(f"Interpreted as [T, H, W] -> [1, 1, T, H, W]")
                        hsi_tensor = torch.from_numpy(hsi_data).float()
                        hsi_tensor = hsi_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                    else:
                        # Likely (H, W, T)
                        print(f"Interpreted as [H, W, T] -> [1, 1, T, H, W]")
                        hsi_tensor = torch.from_numpy(hsi_data.transpose(2, 0, 1)).float()
                        hsi_tensor = hsi_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                elif len(hsi_data.shape) == 2:
                    # It's a 2D image, not HSI data
                    # Create a fake spectral dimension
                    print(f"Found 2D data with shape {hsi_data.shape}, creating fake spectral dimension")
                    hsi_tensor = torch.from_numpy(hsi_data).float().unsqueeze(0).unsqueeze(0).unsqueeze(
                        0)  # [H,W] -> [1,1,1,H,W]
                    # Repeat along spectral dimension to create [1,1,30,H,W]
                    hsi_tensor = hsi_tensor.repeat(1, 1, 30, 1, 1)
                else:
                    print(f"Unexpected HSI data shape: {hsi_data.shape}, creating dummy data")
                    return torch.zeros((1, 1, 30, 500, 500), dtype=torch.float32)  # Return with batch and channel dims

                # Print debug info about the tensor
                print(f"HSI data shape: {hsi_tensor.shape}")
                print(f"HSI data type: {hsi_tensor.dtype}")
                print(f"HSI value range: {hsi_tensor.min().item():.6f} to {hsi_tensor.max().item():.6f}")

                return hsi_tensor
        except Exception as e:
            print(f"Error loading HSI data from {h5_path}: {e}")
            import traceback
            traceback.print_exc()
            # Return dummy data with a standard shape as fallback
            return torch.zeros((1, 1, 30, 500, 500), dtype=torch.float32)  # Return with batch and channel dims

    def _load_tiff_file(self, patient_dir, identifier):
        """
        Load a TIFF file containing the specified identifier in the filename.

        Args:
            patient_dir (str): Path to the patient directory
            identifier (str): String identifier to search for in filenames

        Returns:
            torch.Tensor or None: Loaded image as tensor, or None if not found
        """
        # Search for files with the identifier
        pattern = os.path.join(patient_dir, f"*{identifier}*.tif*")
        matching_files = glob.glob(pattern, recursive=True)

        if not matching_files:
            # Try searching in subdirectories
            for subdir, _, _ in os.walk(patient_dir):
                if subdir == patient_dir:
                    continue
                pattern = os.path.join(subdir, f"*{identifier}*.tif*")
                subdir_matches = glob.glob(pattern)
                matching_files.extend(subdir_matches)

        if not matching_files:
            return None

        # Use the first matching file
        tiff_path = matching_files[0]
        try:
            # Try using tifffile first (better for scientific TIFF files)
            img = tifffile.imread(tiff_path)

            # Normalize to [0, 1] float range
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            else:
                # Already floating point, just ensure it's normalized
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / img.max()

            # Handle grayscale vs. color images
            if len(img.shape) == 3 and img.shape[2] > 1:
                # Take first channel of multi-channel image
                img = img[:, :, 0]

            # Ensure the array is 2D
            img = img.squeeze()

            # Convert to torch tensor with shape [1, 1, H, W]
            return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

        except Exception as e:
            print(f"Error loading {identifier} image from {tiff_path}: {e}")

            # Try with PIL as fallback
            try:
                img = np.array(Image.open(tiff_path))
                # Convert to float and normalize
                img = img.astype(np.float32) / 255.0

                # Handle grayscale vs. color images
                if len(img.shape) == 3 and img.shape[2] > 1:
                    # Take first channel of multi-channel image
                    img = img[:, :, 0]

                # Ensure the array is 2D
                img = img.squeeze()

                # Convert to torch tensor with shape [1, 1, H, W]
                return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
            except Exception as e:
                print(f"Error loading {identifier} image with PIL from {tiff_path}: {e}")
                return None

    def __len__(self):
        """Return the number of patients in the dataset."""
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        """Get data for a single patient."""
        patient_dir = self.patient_dirs[idx]

        # Load HSI data - now returns [1, C, T, H, W] format
        hsi_data = self._load_hsi(patient_dir)

        # Load auxiliary images
        faf_img = self._load_tiff_file(patient_dir, 'FAF')
        ir_img = self._load_tiff_file(patient_dir, 'IR')
        thickness_img = self._load_tiff_file(patient_dir, 'thickness')

        # Create auxiliary data dictionary
        aux_data = {
            'af': faf_img,  # Using 'af' key to match the model's expected keys
            'ir': ir_img,
            'thickness': thickness_img
        }

        # Apply transforms if specified
        if self.transform:
            hsi_data, aux_data = self.transform(hsi_data, aux_data)

        # Create batch index for contrastive learning
        batch_idx = torch.tensor(idx, dtype=torch.long)

        # Extract patient ID from directory path
        patient_id = os.path.basename(patient_dir)

        return {
            'hsi': hsi_data,  # Already has [1, C, T, H, W] shape
            'aux_data': aux_data,
            'batch_idx': batch_idx,
            'patient_id': patient_id,
            'patient_dir': patient_dir
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle missing modalities in the batch.
    This ensures that None values are properly handled when creating a batch.

    Args:
        batch (list): List of samples from the dataset

    Returns:
        dict: Batch of data with properly collated values
    """
    # Process each key separately
    result = {}

    # First, get all keys from the first sample
    keys = batch[0].keys()

    # Process each key
    for key in keys:
        if key == 'aux_data':
            # Handle auxiliary data dictionary separately
            aux_result = {}

            # Get all modality keys from the first sample
            modality_keys = batch[0]['aux_data'].keys()

            for mod_key in modality_keys:
                # Collect all values for this modality across the batch
                mod_values = [sample['aux_data'][mod_key] for sample in batch]

                # Check if any values are None
                if any(v is None for v in mod_values):
                    # If any are None, replace with zero tensors of the right shape
                    has_values = [i for i, v in enumerate(mod_values) if v is not None]
                    if not has_values:
                        # If all are None, just store None
                        aux_result[mod_key] = None
                    else:
                        # Get shape from the first non-None tensor
                        template = mod_values[has_values[0]]
                        shape = template.shape
                        device = template.device
                        dtype = template.dtype

                        # Replace None with zero tensors
                        for i in range(len(mod_values)):
                            if mod_values[i] is None:
                                mod_values[i] = torch.zeros(shape[1:], dtype=dtype, device=device)
                                # Add batch dimension of 1
                                mod_values[i] = mod_values[i].unsqueeze(0)

                        # Stack along batch dimension
                        aux_result[mod_key] = torch.cat(mod_values, dim=0)
                else:
                    # All values are tensors, use standard stacking
                    aux_result[mod_key] = torch.cat(mod_values, dim=0)

            result['aux_data'] = aux_result
        elif key == 'patient_id' or key == 'patient_dir':
            # Handle string values by keeping them as a list
            result[key] = [sample[key] for sample in batch]
        elif key == 'batch_idx':
            # For batch indices, stack them properly
            values = [sample[key] for sample in batch]
            if all(isinstance(v, torch.Tensor) for v in values):
                # Check for zero-dimensional tensors
                if any(v.dim() == 0 for v in values):
                    # Convert zero-dimensional tensors to 1D
                    values = [v.unsqueeze(0) if v.dim() == 0 else v for v in values]
                result[key] = torch.cat(values, dim=0)
            else:
                # Handle non-tensor batch indices
                result[key] = torch.tensor(values)
        elif key == 'hsi':
            # Handle HSI data specifically (5D tensors)
            values = [sample[key] for sample in batch]
            # Remove the single-item batch dimension from each sample
            # Each sample has shape [1, C, T, H, W], we need to remove the first dimension
            values = [v.squeeze(0) for v in values]
            # Stack along a new batch dimension
            result[key] = torch.stack(values, dim=0)
        else:
            # For other keys, use standard stacking
            values = [sample[key] for sample in batch]
            if all(isinstance(v, torch.Tensor) for v in values):
                # Check for zero-dimensional tensors
                if any(v.dim() == 0 for v in values):
                    # Convert zero-dimensional tensors to 1D
                    values = [v.unsqueeze(0) if v.dim() == 0 else v for v in values]
                result[key] = torch.cat(values, dim=0)
            else:
                # For non-tensor values, keep as a list
                result[key] = values

    return result


def create_patient_dataloader(parent_dir, batch_size=4, num_workers=4, shuffle=True):
    """
    Create a DataLoader for the PatientDataset.

    Args:
        parent_dir (str): Path to the parent directory containing patient files
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        shuffle (bool): Whether to shuffle the data

    Returns:
        DataLoader: DataLoader for the PatientDataset
    """
    dataset = PatientDataset(parent_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn  # Use the custom collate function to handle None values
    )


def visualize_patient_data(patient_data, save_dir='visualizations', show=True):
    """
    Visualize the data for a single patient and save the visualization to a file.

    Args:
        patient_data (dict): Patient data from the dataset
        save_dir (str): Directory to save visualizations in
        show (bool): Whether to display the visualization in addition to saving it
    """
    hsi = patient_data['hsi']
    aux_data = patient_data['aux_data']
    patient_id = patient_data['patient_id']

    print(f"Visualizing data for patient: {patient_id}")
    print(f"HSI data shape: {hsi.shape}")

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Patient: {patient_id}", fontsize=16)

    # Plot HSI data (just one wavelength in the middle)
    # Safely display HSI data
    try:
        # Determine the shape and correctly select a middle band
        if len(hsi.shape) == 5:  # [B, C, T, H, W]
            num_wavelengths = hsi.shape[2]
            mid_wavelength = num_wavelengths // 2
            hsi_slice = hsi[0, 0, mid_wavelength].numpy()  # Shape: [H, W]
        elif len(hsi.shape) == 4:  # [B, T, H, W] or [C, T, H, W]
            num_wavelengths = hsi.shape[1]
            mid_wavelength = num_wavelengths // 2
            hsi_slice = hsi[0, mid_wavelength].numpy()  # Shape: [H, W]
        elif len(hsi.shape) == 3:  # [T, H, W]
            num_wavelengths = hsi.shape[0]
            mid_wavelength = num_wavelengths // 2
            hsi_slice = hsi[mid_wavelength].numpy()  # Shape: [H, W]
        else:
            # Fallback for unexpected shapes
            print(f"Unexpected HSI shape: {hsi.shape}, creating dummy image")
            hsi_slice = np.zeros((100, 100))

        print(f"  Using wavelength {mid_wavelength} of {num_wavelengths}")

        # Display the image without cropping
        axes[0].imshow(hsi_slice, cmap='viridis')
        axes[0].set_title(f"HSI (Wavelength {mid_wavelength})")
    except Exception as e:
        print(f"Error displaying HSI data: {e}")
        import traceback
        traceback.print_exc()
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
                # Get the auxiliary data and determine its shape
                aux_data_tensor = aux_data[key][0] if aux_data[key].shape[0] == 1 else aux_data[key]

                # Handle different possible shapes
                if len(aux_data_tensor.shape) == 3 and aux_data_tensor.shape[0] == 1:  # [1, H, W]
                    aux_img = aux_data_tensor[0].numpy()
                elif len(aux_data_tensor.shape) == 3:  # [C, H, W]
                    aux_img = aux_data_tensor[0].numpy()
                elif len(aux_data_tensor.shape) == 2:  # [H, W]
                    aux_img = aux_data_tensor.numpy()
                else:
                    # Try to squeeze to 2D
                    aux_img = aux_data_tensor.squeeze().numpy()

                # Display the image without cropping
                axes[i].imshow(aux_img, cmap='gray')
                axes[i].set_title(title)
            except Exception as e:
                print(f"Error displaying {key} data: {e}")
                import traceback
                traceback.print_exc()
                axes[i].text(0.5, 0.5, f"Error displaying {title}",
                             ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{title} Error")
        else:
            axes[i].text(0.5, 0.5, f"No {title} available",
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"Missing: {title}")
        axes[i].axis('off')
        i += 1

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save the figure to a file
    save_path = os.path.join(save_dir, f"{patient_id}_visualization.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")

    # Show the figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_batch(batch, save_dir='visualizations/batch', show=False):
    """
    Visualize a batch of patient data and save the visualizations.

    Args:
        batch (dict): Batch of data from the DataLoader
        save_dir (str): Directory to save visualizations in
        show (bool): Whether to display the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get batch size and patient IDs
    batch_size = batch['hsi'].shape[0]
    patient_ids = batch['patient_id']

    print(f"Visualizing batch of {batch_size} patients")

    # Process each patient in the batch
    for i in range(batch_size):
        # Create a patient data dict for this sample
        patient_data = {
            'hsi': batch['hsi'][i:i + 1],  # Keep batch dimension to match expected format
            'aux_data': {key: tensor[i:i + 1] if tensor is not None else None
                         for key, tensor in batch['aux_data'].items()},
            'patient_id': patient_ids[i],
            'batch_idx': batch['batch_idx'][i] if 'batch_idx' in batch else i
        }

        # Visualize this patient's data
        patient_save_dir = os.path.join(save_dir, f"patient_{i}")
        visualize_patient_data(patient_data, save_dir=patient_save_dir, show=show)


if __name__ == "__main__":
    # Test the dataset and visualization
    import sys

    if len(sys.argv) > 1:
        parent_dir = sys.argv[1]
    else:
        parent_dir = "dummydata"  # Replace with default path

    print(f"Loading data from: {parent_dir}")

    # Create dataset
    dataset = PatientDataset(parent_dir)

    if len(dataset) == 0:
        print("No patient data found!")
        sys.exit(1)

    # Set up visualization directory
    visualization_dir = "visualizations/dataset_test"
    os.makedirs(visualization_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {visualization_dir}")

    # Visualize data for a few patients
    num_to_visualize = min(3, len(dataset))
    for i in range(num_to_visualize):
        print(f"\nProcessing patient {i + 1}/{num_to_visualize}")
        patient_data = dataset[i]
        patient_save_dir = os.path.join(visualization_dir, f"patient_{i}")
        visualize_patient_data(patient_data, save_dir=patient_save_dir, show=False)

    print("\nTesting DataLoader and batch visualization...")
    dataloader = create_patient_dataloader(parent_dir, batch_size=min(2, len(dataset)))

    # Get one batch and visualize it
    batch = next(iter(dataloader))
    print(f"Batch size: {batch['hsi'].shape[0]}")
    print(f"HSI shape: {batch['hsi'].shape}")

    # Visualize the batch
    visualize_batch(batch, save_dir=os.path.join(visualization_dir, "batch_test"), show=False)

    # Print available auxiliary modalities for each patient in batch
    for i in range(batch['hsi'].shape[0]):
        print(f"\nPatient {i} ({batch['patient_id'][i]}):")
        print("  Available modalities:")
        for modality, tensor in batch['aux_data'].items():
            if tensor is not None:
                print(f"  - {modality}: {tensor[i].shape}")
            else:
                print(f"  - {modality}: Not available")

    print("\nAll visualizations saved to directory:", visualization_dir)