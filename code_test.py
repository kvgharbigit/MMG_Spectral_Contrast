import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from base import MultiModalSpectralGPT


def calculate_num_patches(img_size, patch_size, num_frames, t_patch_size):
    """Calculate the number of patches that will be generated."""
    return (img_size // patch_size) * (img_size // patch_size) * (num_frames // t_patch_size)


def create_dummy_data(batch_size=3):
    """Create dummy data for testing the model."""
    # Create HSI data: [B, C, T, H, W]
    hsi_data = torch.randn(batch_size, 1, 12, 224, 224)

    # Create auxiliary data with different sizes
    aux_data = {
        'ir': torch.randn(batch_size, 3, 128, 128),
        'af': torch.randn(batch_size, 3, 128, 128),
        'thickness': torch.randn(batch_size, 1, 128, 128)
    }

    # Create batch indices for contrastive learning
    batch_indices = torch.arange(batch_size)

    return hsi_data, aux_data, batch_indices


def test_model():
    """Test the MultiModalSpectralGPT model with dummy data."""

    # Model parameters
    hsi_img_size = 224  # HSI image size
    aux_img_size = 128  # Auxiliary image size
    patch_size = 16
    num_frames = 12
    t_patch_size = 3
    embed_dim = 768

    # Calculate the actual number of patches for HSI
    num_patches = calculate_num_patches(hsi_img_size, patch_size, num_frames, t_patch_size)
    print(f"Number of patches that will be generated: {num_patches}")

    # Create model with correct dimensions
    model = MultiModalSpectralGPT(
        hsi_img_size=hsi_img_size,     # Specify HSI image size
        aux_img_size=aux_img_size,      # Specify auxiliary image size
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_frames=num_frames,
        t_patch_size=t_patch_size,
        in_chans=1,                     # HSI channels
        aux_chans=3,                    # Auxiliary channels
        aux_encoder_type='vit'          # Choose auxiliary encoder type
    )

    # Verify positional embedding size
    print(f"Position embedding shape: {model.pos_embed.shape}")
    print(f"Expected shape: [1, {num_patches}, {embed_dim}]")

    # Generate dummy data
    hsi_data, aux_data, batch_indices = create_dummy_data(batch_size=3)

    # Move everything to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = model.to(device)
    hsi_data = hsi_data.to(device)
    aux_data = {k: v.to(device) for k, v in aux_data.items()}
    batch_indices = batch_indices.to(device)

    print("\nTesting model with dummy data...")
    print(f"HSI data shape: {hsi_data.shape}")
    for modality, data in aux_data.items():
        print(f"{modality} data shape: {data.shape}")

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