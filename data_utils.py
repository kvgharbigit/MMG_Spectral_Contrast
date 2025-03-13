import os
import glob
import h5py
import torch
import numpy as np
from PIL import Image
import tifffile
import torch.nn.functional as F


def find_patient_dirs(parent_dir):
    """Find all directories containing .h5 files at any level of nesting."""
    patient_dirs = []
    for root, _, files in os.walk(parent_dir):
        if any(file.endswith('.h5') for file in files):
            patient_dirs.append(root)
    return patient_dirs


def load_hsi_data(patient_dir):
    """Load HSI data from .h5 file with detailed error handling and shape logging."""
    h5_files = glob.glob(os.path.join(patient_dir, '*.h5'))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 file found in {patient_dir}")

    h5_path = h5_files[0]
    print(f"Loading HSI data from {h5_path}")

    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"H5 file structure: Keys at root level: {list(f.keys())}")

            # Try to find the HSI data - look for common patterns
            if 'Cube' in f:
                if 'Images' in f['Cube']:
                    hsi_data = f['Cube']['Images'][:]
                    print(f"Found HSI data in Cube/Images with shape {hsi_data.shape}")
                elif isinstance(f['Cube'], h5py.Dataset):
                    hsi_data = f['Cube'][:]
                    print(f"Found HSI data in Cube with shape {hsi_data.shape}")
                else:
                    # Check Cube's contents
                    print(f"Cube contains keys: {list(f['Cube'].keys())}")
                    for key in f['Cube'].keys():
                        if isinstance(f['Cube'][key], h5py.Dataset):
                            hsi_data = f['Cube'][key][:]
                            print(f"Found HSI data in Cube/{key} with shape {hsi_data.shape}")
                            break
            else:
                # Look for other standard keys
                for key in ['hsi', 'data', 'image', 'hyperspectral']:
                    if key in f and isinstance(f[key], h5py.Dataset):
                        hsi_data = f[key][:]
                        print(f"Found HSI data in {key} with shape {hsi_data.shape}")
                        break
                else:
                    # Use the first dataset we can find
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            hsi_data = f[key][:]
                            print(f"Found HSI data in {key} with shape {hsi_data.shape}")
                            break

            if 'hsi_data' not in locals():
                print(f"Could not find HSI data in {h5_path}")
                return torch.zeros((1, 1, 30, 500, 500), dtype=torch.float32)

            # Format the data properly
            print(f"Raw HSI data shape: {hsi_data.shape}, dtype: {hsi_data.dtype}")

            if len(hsi_data.shape) == 3 and hsi_data.shape[0] < 100 and hsi_data.shape[1] > 500:
                # Likely (T, H, W) format
                print(f"Interpreting as [T, H, W] format with T={hsi_data.shape[0]}")
                hsi_tensor = torch.from_numpy(hsi_data).float()
                hsi_tensor = hsi_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                print(f"Converted to tensor with shape {hsi_tensor.shape}")
            elif len(hsi_data.shape) == 3:
                # Determine if spectral dimension is first or last
                if hsi_data.shape[0] < hsi_data.shape[1] and hsi_data.shape[0] < hsi_data.shape[2]:
                    # Likely (T, H, W)
                    print(f"Interpreting as [T, H, W] format with T={hsi_data.shape[0]}")
                    hsi_tensor = torch.from_numpy(hsi_data).float()
                    hsi_tensor = hsi_tensor.unsqueeze(0).unsqueeze(0)
                    print(f"Converted to tensor with shape {hsi_tensor.shape}")
                else:
                    # Likely (H, W, T)
                    print(f"Interpreting as [H, W, T] format with T={hsi_data.shape[2]}")
                    hsi_tensor = torch.from_numpy(hsi_data.transpose(2, 0, 1)).float()
                    hsi_tensor = hsi_tensor.unsqueeze(0).unsqueeze(0)
                    print(f"Converted to tensor with shape {hsi_tensor.shape}")
            elif len(hsi_data.shape) == 2:
                # Create a fake spectral dimension for 2D images
                print(f"Found 2D data with shape {hsi_data.shape}, creating fake spectral dimension")
                hsi_tensor = torch.from_numpy(hsi_data).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
                hsi_tensor = hsi_tensor.repeat(1, 1, 30, 1, 1)
                print(f"Converted to tensor with shape {hsi_tensor.shape}")
            else:
                print(f"Unexpected HSI data shape: {hsi_data.shape}, creating dummy data")
                return torch.zeros((1, 1, 30, 500, 500), dtype=torch.float32)

            # Print value range
            print(f"HSI value range: {hsi_tensor.min().item():.6f} to {hsi_tensor.max().item():.6f}")
            return hsi_tensor
    except Exception as e:
        print(f"Error loading HSI data from {h5_path}: {e}")
        return torch.zeros((1, 1, 30, 500, 500), dtype=torch.float32)


def load_tiff_file(patient_dir, identifier):
    """Load a TIFF file with the specified identifier in the filename."""
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
        # Try using tifffile first
        img = tifffile.imread(tiff_path)

        # Normalize to [0, 1] float range
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / img.max()

        # Handle multi-channel images
        if len(img.shape) == 3 and img.shape[2] > 1:
            img = img[:, :, 0]

        # Ensure the array is 2D
        img = img.squeeze()

        # Convert to torch tensor with shape [1, 1, H, W]
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    except Exception as e:
        # Try with PIL as fallback
        try:
            img = np.array(Image.open(tiff_path))
            img = img.astype(np.float32) / 255.0

            if len(img.shape) == 3 and img.shape[2] > 1:
                img = img[:, :, 0]

            img = img.squeeze()
            return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        except Exception:
            return None


def select_spectral_bands(hsi_img, target_bands):
    """Select specific spectral bands from the HSI image."""
    B, C, T, H, W = hsi_img.shape

    # Check if already has the target number of bands
    if T == target_bands:
        print(f"No band selection needed - already have {T} bands")
        return hsi_img

    # Determine band selection strategy
    if T <= target_bands:
        print(f"Input has fewer bands ({T}) than target ({target_bands}), using all available bands")
        selected_indices = list(range(T))
    else:
        step = max(1, (T - 1) // (target_bands - 1))
        selected_indices = list(range(0, T, step))[:target_bands]
        print(f"Selecting {len(selected_indices)} bands from {T} total with step size {step}")

    # Ensure we have exactly the target number of bands
    if len(selected_indices) > target_bands:
        print(f"Trimming selected bands from {len(selected_indices)} to {target_bands}")
        selected_indices = selected_indices[:target_bands]
    elif len(selected_indices) < target_bands:
        padding_count = target_bands - len(selected_indices)
        print(
            f"Padding selected bands from {len(selected_indices)} to {target_bands} (adding {padding_count} repeat bands)")
        selected_indices.extend([selected_indices[-1]] * padding_count)

    # Print final selection
    print(f"Selected band indices: {selected_indices}")

    # Select the specified bands
    index_tensor = torch.tensor(selected_indices, device=hsi_img.device, dtype=torch.long)
    result = torch.index_select(hsi_img, 2, index_tensor)
    print(f"Band selection complete: {T} bands → {result.shape[2]} bands")
    return result


def detect_mask(image):
    """Detect black mask in an image tensor (1 for valid regions, 0 for masked)."""
    threshold = 0.005

    if image.shape[1] > 1:
        return (image.mean(dim=1, keepdim=True) > threshold).float()
    else:
        return (image > threshold).float()


def apply_spatial_registration(hsi_img, aux_data, analysis_dim, target_bands):
    """Apply spatial registration to HSI and auxiliary data."""
    # Log original shape before any processing
    original_shape = hsi_img.shape
    print(f"Original HSI shape before registration: {original_shape}")

    # Process HSI data
    B, C, T, H, W = hsi_img.shape

    # Select spectral bands if needed
    if T != target_bands:
        print(f"Selecting {target_bands} bands from original {T} bands")
        hsi_img = select_spectral_bands(hsi_img, target_bands)
        B, C, T, H, W = hsi_img.shape
        print(f"Shape after band selection: {hsi_img.shape}")

    # Resize HSI if needed
    if H != analysis_dim or W != analysis_dim:
        print(f"Resizing spatial dimensions from {H}x{W} to {analysis_dim}x{analysis_dim}")
        hsi_reshaped = hsi_img.view(B * T, C, H, W)
        hsi_resized = F.interpolate(
            hsi_reshaped,
            size=(analysis_dim, analysis_dim),
            mode='bilinear',
            align_corners=False
        )
        hsi_registered = hsi_resized.view(B, C, T, analysis_dim, analysis_dim)
        print(f"Shape after resizing: {hsi_registered.shape}")
    else:
        print(f"No spatial resizing needed - dimensions already {H}x{W}")
        hsi_registered = hsi_img

    # Process auxiliary data
    aux_registered = {}
    thickness_mask = None

    for modality, data in aux_data.items():
        if data is not None:
            if data.shape[2] != analysis_dim or data.shape[3] != analysis_dim:
                aux_registered[modality] = F.interpolate(
                    data,
                    size=(analysis_dim, analysis_dim),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                aux_registered[modality] = data
        else:
            aux_registered[modality] = None

    # Create thickness mask if available
    if 'thickness' in aux_registered and aux_registered['thickness'] is not None:
        thickness_mask = detect_mask(aux_registered['thickness'])

    return hsi_registered, aux_registered, thickness_mask