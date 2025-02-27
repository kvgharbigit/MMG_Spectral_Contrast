import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from timm.models.layers import to_2tuple
from base import MultiModalSpectralGPT


def calculate_num_patches(hsi_img, patch_size, t_patch_size):
    """
    Dynamically calculate the number of patches based on preprocessed HSI image.

    Args:
        hsi_img (torch.Tensor): Preprocessed HSI image tensor of shape [B, C, T, H, W]
        patch_size (tuple): Spatial patch size
        t_patch_size (int): Temporal/spectral patch size

    Returns:
        int: Number of patches
    """
    # Get dimensions after preprocessing
    _, _, T, H, W = hsi_img.shape

    # Calculate number of patches
    spatial_patches_h = H // patch_size[0]
    spatial_patches_w = W // patch_size[1]
    temporal_patches = T // t_patch_size

    return spatial_patches_h * spatial_patches_w * temporal_patches


def create_dummy_data(batch_size=3):
    """Create dummy data for testing the model with different input sizes."""
    # Create HSI data: [B, C, T, H, W] with larger size
    # Updated to 500x500 spatial size and 91 spectral bands
    hsi_data = torch.randn(batch_size, 1, 91, 500, 500)  # Test resizing

    # Create auxiliary data with different sizes - NOW 1 CHANNEL EACH
    aux_data = {
        'ir': torch.randn(batch_size, 1, 128, 128),
        'af': torch.randn(batch_size, 1, 256, 256),
        'thickness': torch.randn(batch_size, 1, 200, 200)
    }

    # Create batch indices for contrastive learning
    batch_indices = torch.arange(batch_size)

    return hsi_data, aux_data, batch_indices


def test_model():
    """Test the MultiModalSpectralGPT model with dummy data."""

    # Model parameters
    analysis_dim = 500  # Spatial dimension matching HSI input
    patch_size = (25, 25)  # Specify as a tuple
    t_patch_size = 5  # Adjust temporal patch size to divide spectral bands evenly
    embed_dim = 768

    # Verify that analysis_dim is divisible by patch_size
    assert analysis_dim % patch_size[0] == 0, f"{analysis_dim} must be divisible by {patch_size[0]}"

    # Create model with updated parameters
    model = MultiModalSpectralGPT(
        analysis_dim=analysis_dim,  # Common spatial dimension
        patch_size=patch_size,  # Use tuple
        embed_dim=embed_dim,
        t_patch_size=t_patch_size,
        in_chans=1,  # HSI channels
        aux_chans=1,  # NOW 1 CHANNEL FOR AUXILIARY CHANNELS
        aux_encoder_type='vit'  # Choose auxiliary encoder type
    )

    # Generate dummy data with varied input sizes to test spatial registration
    hsi_data, aux_data, batch_indices = create_dummy_data(batch_size=3)

    print("\nInput data shapes before spatial registration:")
    print(f"HSI data shape: {hsi_data.shape}")
    for modality, data in aux_data.items():
        print(f"{modality} data shape: {data.shape}")

    # Move everything to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = model.to(device)
    hsi_data = hsi_data.to(device)
    aux_data = {k: v.to(device) for k, v in aux_data.items()}
    batch_indices = batch_indices.to(device)

    # Get the registered dimensions by manually applying spatial registration
    with torch.no_grad():
        hsi_registered, aux_registered = model.spatial_registration(hsi_data, aux_data)

    print("\nData shapes after spatial registration:")
    print(f"HSI data shape: {hsi_registered.shape}")
    for modality, data in aux_registered.items():
        print(f"{modality} data shape: {data.shape}")

    # Manually set up patch embedding
    model._setup_patch_embedding(hsi_registered)

    # Dynamically calculate number of patches after preprocessing
    num_patches = calculate_num_patches(
        hsi_registered,
        patch_size,
        t_patch_size
    )
    print(f"\nNumber of patches that will be generated: {num_patches}")

    # Verify positional embedding size
    print(f"Position embedding shape: {model.pos_embed.shape}")
    print(f"Expected shape: [1, {num_patches}, {embed_dim}]")

    print("\nTesting model with dummy data...")

    # Test forward pass
    try:
        model.train()
        output = model(hsi_data, aux_data, batch_indices)

        print("\nForward pass successful!")
        print("\nOutput dictionary contains:")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"{key}: {value}")

        # Check if losses are reasonable
        print("\nLoss values:")
        print(f"Reconstruction loss: {output['loss_recon']:.4f}")
        print(f"Contrastive loss: {output['loss_contrast']:.4f}")
        print(f"Total loss: {output['loss']:.4f}")

        # Check mask ratio
        mask = output['mask']
        actual_mask_ratio = mask.float().mean()
        print(f"\nActual mask ratio: {actual_mask_ratio:.4f}")

    except Exception as e:
        print(f"\nError during forward pass: {str(e)}")
        raise


if __name__ == '__main__':
    test_model()