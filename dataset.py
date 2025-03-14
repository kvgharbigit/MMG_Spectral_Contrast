import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
from scipy.ndimage import rotate as scipy_rotate
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Import helpers from utility modules
from data_utils import (
    find_patient_dirs, load_hsi_data, load_tiff_file,
    apply_spatial_registration
)
from visualisation import visualize_patient_data, visualize_batch


class PatientDataset(Dataset):
    """
    Dataset for loading patient data with integrated spatial registration.
    Performs preprocessing during data loading to ensure consistent dimensions.
    """

    def __init__(self, parent_dir, analysis_dim=500, target_bands=30, transform=None, augment=False):
        """
        Initialize the dataset with spatial registration parameters.

        Args:
            parent_dir (str): Path to the parent directory containing patient files
            analysis_dim (int): Target spatial dimension for all modalities
            target_bands (int): Target number of spectral bands for HSI
            transform (callable, optional): Optional external transform to be applied
            augment (bool): Whether to apply data augmentation during training
        """
        self.parent_dir = parent_dir
        self.external_transform = transform
        self.analysis_dim = analysis_dim
        self.target_bands = target_bands
        self.augment = augment

        # Create augmentation transform if enabled
        if self.augment:
            self.transform = MultiModalTransforms(prob=0.5)
        else:
            self.transform = None

        # Find all patient directories
        self.patient_dirs = find_patient_dirs(parent_dir)
        print(f"Found {len(self.patient_dirs)} patient directories")
        print(f"Data will be registered to {analysis_dim}x{analysis_dim} spatial dimensions")
        print(f"Target number of spectral bands: {target_bands}")
        print(f"Data augmentation: {'Enabled' if self.augment else 'Disabled'}")

    def __len__(self):
        """Return the number of patients in the dataset."""
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        """Get data for a single patient with spatial registration and optional augmentations applied."""
        patient_dir = self.patient_dirs[idx]

        # Load HSI data
        hsi_data = load_hsi_data(patient_dir)

        # Load auxiliary images
        aux_data = {
            'af': load_tiff_file(patient_dir, 'FAF'),
            'ir': load_tiff_file(patient_dir, 'IR'),
            'thickness': load_tiff_file(patient_dir, 'thickness')
        }

        # Apply spatial registration
        hsi_registered, aux_registered, thickness_mask = apply_spatial_registration(
            hsi_data, aux_data, self.analysis_dim, self.target_bands
        )

        # Apply data augmentation if enabled
        if self.transform:
            hsi_registered, aux_registered, thickness_mask = self.transform(
                hsi_registered, aux_registered, thickness_mask
            )

        # Apply external transforms if specified
        if self.external_transform:
            hsi_registered, aux_registered = self.external_transform(hsi_registered, aux_registered)

        # Create batch index for contrastive learning
        batch_idx = torch.tensor(idx, dtype=torch.long)

        # Extract patient ID from directory path
        patient_id = os.path.basename(patient_dir)

        return {
            'hsi': hsi_registered,
            'aux_data': aux_registered,
            'thickness_mask': thickness_mask,
            'batch_idx': batch_idx,
            'patient_id': patient_id,
            'patient_dir': patient_dir
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle missing modalities and thickness masks.
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
        elif key == 'thickness_mask':
            # Handle thickness mask
            values = [sample[key] for sample in batch]
            if all(v is not None for v in values):
                result[key] = torch.cat(values, dim=0)
            else:
                # Handle missing masks
                has_values = [i for i, v in enumerate(values) if v is not None]
                if not has_values:
                    sample_hsi = batch[0]['hsi']
                    B = len(batch)
                    H, W = sample_hsi.shape[-2], sample_hsi.shape[-1]
                    result[key] = torch.ones(B, 1, H, W, device=sample_hsi.device)
                else:
                    # Use the first non-None mask as a template
                    template = values[has_values[0]]
                    for i in range(len(values)):
                        if values[i] is None:
                            values[i] = torch.ones_like(template)
                    result[key] = torch.cat(values, dim=0)
        elif key == 'patient_id' or key == 'patient_dir':
            # Handle string values by keeping them as a list
            result[key] = [sample[key] for sample in batch]
        elif key == 'batch_idx':
            # For batch indices, stack them properly
            values = [sample[key] for sample in batch]
            if all(isinstance(v, torch.Tensor) for v in values):
                # Convert zero-dimensional tensors to 1D if needed
                values = [v.unsqueeze(0) if v.dim() == 0 else v for v in values]
                result[key] = torch.cat(values, dim=0)
            else:
                result[key] = torch.tensor(values)
        elif key == 'hsi':
            # Handle HSI data (5D tensors)
            values = [sample[key].squeeze(0) for sample in batch]  # Remove single-item batch dimension
            result[key] = torch.stack(values, dim=0)
        else:
            # For other keys, use standard stacking
            values = [sample[key] for sample in batch]
            if all(isinstance(v, torch.Tensor) for v in values):
                values = [v.unsqueeze(0) if v.dim() == 0 else v for v in values]
                result[key] = torch.cat(values, dim=0)
            else:
                result[key] = values

    return result


def create_patient_dataloader(parent_dir, analysis_dim=500, target_bands=30,
                              batch_size=4, num_workers=4, shuffle=True, augment=False):
    """
    Create a DataLoader for the PatientDataset with integrated spatial registration.
    """
    dataset = PatientDataset(
        parent_dir,
        analysis_dim,
        target_bands,
        augment=augment
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )


class MultiModalTransforms:
    """
    Custom transformations for multi-modal hyperspectral and auxiliary data.
    Ensures the same spatial transformations are applied to all modalities.
    """

    def __init__(self, prob=0.5, rotation_degrees=10, scale_range=(0.9, 1.1)):
        """
        Initialize transformation parameters.

        Args:
            prob (float): Probability of applying each transformation
            rotation_degrees (int): Maximum rotation in degrees
            scale_range (tuple): Range for random scaling (min, max)
        """
        self.prob = prob
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range

    def __call__(self, hsi_data, aux_data, thickness_mask=None):
        """
        Apply the same random transformations to HSI and all auxiliary modalities.
        """
        # Get spatial dimensions
        B, C, T, H, W = hsi_data.shape

        # Apply the same transformation to all modalities for each batch item
        for b in range(B):
            # Decide which transformations to apply
            do_flip = random.random() < self.prob
            do_rotate = random.random() < self.prob
            do_scale = random.random() < self.prob

            # Random flip (horizontal)
            if do_flip:
                # Flip HSI - need to handle the spectral dimension
                for t in range(T):
                    # Use torch.flip instead of TF.hflip for tensor
                    hsi_data[b, 0, t] = torch.flip(hsi_data[b, 0, t], dims=[1])

                # Flip auxiliary modalities
                for modality in aux_data:
                    if aux_data[modality] is not None:
                        aux_data[modality][b, 0] = torch.flip(aux_data[modality][b, 0], dims=[1])

                # Flip thickness mask if provided
                if thickness_mask is not None:
                    thickness_mask[b, 0] = torch.flip(thickness_mask[b, 0], dims=[1])

            # Random rotation - using NumPy-based rotation instead of TF.rotate
            if do_rotate:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)

                # Rotate HSI using custom rotation function
                for t in range(T):
                    img_tensor = hsi_data[b, 0, t]  # Shape: [H, W]
                    rotated = self.custom_rotate(img_tensor, angle)
                    hsi_data[b, 0, t] = rotated

                # Rotate auxiliary modalities
                for modality in aux_data:
                    if aux_data[modality] is not None:
                        img_tensor = aux_data[modality][b, 0]  # Shape: [H, W]
                        rotated = self.custom_rotate(img_tensor, angle)
                        aux_data[modality][b, 0] = rotated

                # Rotate thickness mask if provided
                if thickness_mask is not None:
                    img_tensor = thickness_mask[b, 0]  # Shape: [H, W]
                    rotated = self.custom_rotate(img_tensor, angle, is_mask=True)
                    thickness_mask[b, 0] = rotated

            # Similarly for scaling - implement with proper reshaping
            if do_scale:
                scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])

                # Calculate new dimensions
                new_h = int(H * scale_factor)
                new_w = int(W * scale_factor)

                # Scale HSI using custom scaling function
                for t in range(T):
                    img_tensor = hsi_data[b, 0, t]  # Shape: [H, W]
                    scaled = self.custom_scale(img_tensor, new_h, new_w, (H, W))
                    hsi_data[b, 0, t] = scaled

                # Scale auxiliary modalities
                for modality in aux_data:
                    if aux_data[modality] is not None:
                        img_tensor = aux_data[modality][b, 0]  # Shape: [H, W]
                        scaled = self.custom_scale(img_tensor, new_h, new_w, (H, W))
                        aux_data[modality][b, 0] = scaled

                # Scale thickness mask if provided
                if thickness_mask is not None:
                    img_tensor = thickness_mask[b, 0]  # Shape: [H, W]
                    scaled = self.custom_scale(img_tensor, new_h, new_w, (H, W), is_mask=True)
                    thickness_mask[b, 0] = scaled

        return hsi_data, aux_data, thickness_mask

    def custom_rotate(self, img_tensor, angle, is_mask=False):
        """Custom rotation function that works with 2D tensors"""
        # Convert tensor to numpy
        img_np = img_tensor.detach().cpu().numpy()

        # Use scipy's rotate which works well with 2D arrays
        # For masks, use nearest neighbor interpolation to preserve binary values
        interpolation_order = 0 if is_mask else 1
        rotated_np = scipy_rotate(
            img_np,
            angle,
            reshape=False,
            order=interpolation_order,
            mode='constant',
            cval=0.0
        )

        # Convert back to tensor
        rotated_tensor = torch.from_numpy(rotated_np).to(img_tensor.device).to(img_tensor.dtype)

        return rotated_tensor

    def custom_scale(self, img_tensor, new_h, new_w, target_size, is_mask=False):
        """Custom scaling function that works with 2D tensors"""
        # Convert to numpy for easier handling
        img_np = img_tensor.detach().cpu().numpy()
        H, W = target_size

        # For non-square images, maintain aspect ratio
        if new_h / new_w != H / W:
            aspect_ratio = H / W
            if new_h / new_w > aspect_ratio:
                # Too tall, adjust height
                new_h = int(new_w * aspect_ratio)
            else:
                # Too wide, adjust width
                new_w = int(new_h / aspect_ratio)

        # Resize using PyTorch's F.interpolate
        # First add batch and channel dimensions
        img_tensor_reshaped = img_tensor.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]

        # Different interpolation mode for masks
        mode = 'nearest' if is_mask else 'bilinear'
        scaled_tensor = F.interpolate(
            img_tensor_reshaped,
            size=(new_h, new_w),
            mode=mode,
            align_corners=False if mode == 'bilinear' else None
        )

        # Remove batch and channel dimensions
        scaled_tensor = scaled_tensor.squeeze(0).squeeze(0)  # [1, 1, new_h, new_w] -> [new_h, new_w]

        # Center crop or pad to get back to original size
        final_tensor = self._center_crop_or_pad_2d(scaled_tensor, (H, W))

        return final_tensor

    def _center_crop_or_pad_2d(self, img, target_size):
        """Center crop or pad a 2D tensor to the target size."""
        h, w = img.shape
        th, tw = target_size

        # If image is larger than target in both dimensions, center crop
        if h >= th and w >= tw:
            # Calculate starting positions for cropping
            h_start = (h - th) // 2
            w_start = (w - tw) // 2
            return img[h_start:h_start + th, w_start:w_start + tw]

        # Otherwise, create new tensor and pad
        result = torch.zeros(th, tw, dtype=img.dtype, device=img.device)

        # Calculate where to place the image
        h_start = max(0, (th - h) // 2)
        w_start = max(0, (tw - w) // 2)

        # Calculate which part of the image to use
        h_img_start = max(0, (h - th) // 2)
        w_img_start = max(0, (w - tw) // 2)

        # Calculate how much of the image to use
        h_to_use = min(h - h_img_start, th - h_start)
        w_to_use = min(w - w_img_start, tw - w_start)

        # Place the image within the result tensor
        result[h_start:h_start + h_to_use, w_start:w_start + w_to_use] = img[h_img_start:h_img_start + h_to_use,
                                                                         w_img_start:w_img_start + w_to_use]

        return result

if __name__ == "__main__":
    import sys

    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "dummydata"
    print(f"Loading data from: {data_dir}")

    # Create dataset with spatial registration
    dataset = PatientDataset(data_dir, analysis_dim=500, target_bands=30)

    if len(dataset) == 0:
        print("No patient data found!")
        sys.exit(1)

    # Set up visualization directory
    visualization_dir = "visualizations/dataset_test"
    os.makedirs(visualization_dir, exist_ok=True)

    # Visualize a few patients
    num_to_visualize = min(4, len(dataset))
    for i in range(num_to_visualize):
        print(f"\nProcessing patient {i + 1}/{num_to_visualize}")
        patient_data = dataset[i]
        visualize_patient_data(
            patient_data,
            save_dir=os.path.join(visualization_dir, f"patient_{i}"),
            show=False
        )

    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = create_patient_dataloader(
        data_dir,
        batch_size=min(2, len(dataset))
    )

    # Get and visualize a batch
    batch = next(iter(dataloader))
    print(f"Batch loaded with {batch['hsi'].shape[0]} patients")
    print(f"HSI shape: {batch['hsi'].shape}")

    if 'thickness_mask' in batch and batch['thickness_mask'] is not None:
        print(f"Thickness mask shape: {batch['thickness_mask'].shape}")

    visualize_batch(
        batch,
        save_dir=os.path.join(visualization_dir, "batch_test"),
        show=False
    )

    print("All visualizations saved to:", visualization_dir)