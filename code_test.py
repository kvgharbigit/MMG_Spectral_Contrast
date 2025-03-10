import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from timm.layers import to_2tuple
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


def test_model_contrastive_modes():
    """Test both contrastive learning modes of the MultiModalSpectralGPT model."""

    # Basic model parameters
    analysis_dim = 500  # Spatial dimension
    patch_size = (25, 25)  # Spatial patch size
    t_patch_size = 5  # Temporal/spectral patch size
    embed_dim = 768

    # Generate dummy data
    hsi_data, aux_data, batch_indices = create_dummy_data(batch_size=3)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Move data to device
    hsi_data = hsi_data.to(device)
    aux_data = {k: v.to(device) for k, v in aux_data.items()}
    batch_indices = batch_indices.to(device)

    # Test global contrastive mode first
    print("\n==== Testing Global Contrastive Mode ====")
    global_model = MultiModalSpectralGPT(
        analysis_dim=analysis_dim,
        patch_size=patch_size,
        embed_dim=embed_dim,
        t_patch_size=t_patch_size,
        in_chans=1,
        aux_chans=1,
        contrastive_mode='global'  # Global mode
    ).to(device)

    # Run forward pass with global mode
    with torch.no_grad():
        global_output = global_model(hsi_data, aux_data, batch_indices)

    print(f"Global contrastive loss: {global_output['loss_contrast'].item():.4f}")

    # Now test spatial contrastive mode
    print("\n==== Testing Spatial Contrastive Mode ====")
    spatial_model = MultiModalSpectralGPT(
        analysis_dim=analysis_dim,
        patch_size=patch_size,
        embed_dim=embed_dim,
        t_patch_size=t_patch_size,
        in_chans=1,
        aux_chans=1,
        contrastive_mode='spatial'  # Spatial mode
    ).to(device)

    # Run forward pass with spatial mode
    with torch.no_grad():
        spatial_output = spatial_model(hsi_data, aux_data, batch_indices)

    print(f"Spatial contrastive loss: {spatial_output['loss_contrast'].item():.4f}")

    # Compare the reconstruction losses (should be similar)
    print("\n==== Comparing Losses ====")
    print(f"Global reconstruction loss: {global_output['loss_recon'].item():.4f}")
    print(f"Spatial reconstruction loss: {spatial_output['loss_recon'].item():.4f}")

    # Compare the overall losses
    print(f"Global total loss: {global_output['loss'].item():.4f}")
    print(f"Spatial total loss: {spatial_output['loss'].item():.4f}")

    print("\n==== Test Complete ====")
    print("Both contrastive learning modes are working as expected.")


def test_model_original():
    """Test the original functionality of the MultiModalSpectralGPT model with dummy data."""

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
        contrastive_mode='global'  # Default global mode
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


def test_masking():
    """Test the automatic masking functionality with a mock thickness image."""

    print("\n==== Testing Dynamic Masking ====")

    # Import visualization libraries
    import matplotlib.pyplot as plt
    import os

    # Create output directory for visualizations
    os.makedirs("test_visualizations", exist_ok=True)

    # Basic model parameters
    analysis_dim = 500
    patch_size = (25, 25)
    t_patch_size = 5
    embed_dim = 768

    # Create test data with a simulated masked thickness image
    batch_size = 2
    hsi_data, aux_data, batch_indices = create_dummy_data(batch_size=batch_size)

    # Create a simulated mask in the thickness image
    # Generate circular masks of different sizes for each batch item
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for b in range(batch_size):
        # Create a grid of coordinates
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, 200),
            torch.linspace(-1, 1, 200),
            indexing='ij'
        )
        # Create a circular mask (0 outside, 1 inside the circle)
        # Vary the circle radius for each batch
        radius = 0.7 if b == 0 else 0.5
        mask = ((x ** 2 + y ** 2) < radius ** 2).float()

        # Apply the simulated mask to the thickness image
        # Make pixels outside the circle black (0)
        aux_data['thickness'][b, 0] = aux_data['thickness'][b, 0] * mask

    # Visualize original thickness images with masks
    plt.figure(figsize=(10, 5))
    for b in range(batch_size):
        plt.subplot(1, 2, b + 1)
        plt.imshow(aux_data['thickness'][b, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Original Thickness Image\nRadius={0.7 if b == 0 else 0.5}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_visualizations/original_thickness_masks.png")
    plt.close()

    # Visualize a few HSI bands before masking
    plt.figure(figsize=(15, 5))
    bands_to_show = [0, 10, 20]
    for i, band in enumerate(bands_to_show):
        plt.subplot(1, 3, i + 1)
        plt.imshow(hsi_data[0, 0, band].cpu().numpy(), cmap='gray')
        plt.title(f"HSI Band {band} (Before Masking)")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_visualizations/hsi_before_masking.png")
    plt.close()

    # Move data to device
    hsi_data = hsi_data.to(device)
    aux_data = {k: v.to(device) for k, v in aux_data.items()}
    batch_indices = batch_indices.to(device)

    # Create model
    model = MultiModalSpectralGPT(
        analysis_dim=analysis_dim,
        patch_size=patch_size,
        embed_dim=embed_dim,
        t_patch_size=t_patch_size,
        in_chans=1,
        aux_chans=1,
        contrastive_mode='global'
    ).to(device)

    # Extract the spatial registration module for direct testing
    spatial_reg = model.spatial_registration

    # Test spatial registration with masking
    print("Testing spatial registration with masking...")
    with torch.no_grad():
        # Apply spatial registration
        hsi_registered, aux_registered = spatial_reg(hsi_data, aux_data)

        # Check if HSI has been masked
        print(f"HSI tensor shape: {hsi_registered.shape}")
        print(
            f"Min/max values in HSI before masking (first band): {hsi_data[:, :, 0].min().item():.4f}/{hsi_data[:, :, 0].max().item():.4f}")
        print(
            f"Min/max values in HSI after masking (first band): {hsi_registered[:, :, 0].min().item():.4f}/{hsi_registered[:, :, 0].max().item():.4f}")

        # Visualize HSI bands after masking
        plt.figure(figsize=(15, 5))
        for i, band in enumerate(bands_to_show):
            plt.subplot(1, 3, i + 1)
            plt.imshow(hsi_registered[0, 0, band].cpu().numpy(), cmap='gray')
            plt.title(f"HSI Band {band} (After Masking)")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig("test_visualizations/hsi_after_masking.png")
        plt.close()

        # Visualize all modalities after masking
        plt.figure(figsize=(15, 10))

        # Plot HSI band 0
        plt.subplot(2, 3, 1)
        plt.imshow(hsi_registered[0, 0, 0].cpu().numpy(), cmap='gray')
        plt.title("HSI (Band 0)")
        plt.axis('off')

        # Plot auxiliary modalities
        pos = 2
        for modality in aux_registered:
            if aux_registered[modality] is not None:
                plt.subplot(2, 3, pos)
                plt.imshow(aux_registered[modality][0, 0].cpu().numpy(), cmap='gray')
                plt.title(f"{modality.upper()}")
                plt.axis('off')
                pos += 1

                print(f"{modality} shape: {aux_registered[modality].shape}")
                original_min = aux_data[modality].min().item()
                original_max = aux_data[modality].max().item()
                registered_min = aux_registered[modality].min().item()
                registered_max = aux_registered[modality].max().item()

                print(f"Min/max values in {modality} before masking: {original_min:.4f}/{original_max:.4f}")
                print(f"Min/max values in {modality} after masking: {registered_min:.4f}/{registered_max:.4f}")

        plt.tight_layout()
        plt.savefig("test_visualizations/all_modalities_after_masking.png")
        plt.close()

        # Visualize mask boundaries
        plt.figure(figsize=(10, 5))
        for b in range(batch_size):
            # Create a binary version of the mask for visualization
            binary_mask = (aux_registered['thickness'][b, 0] > 0.05).float().cpu().numpy()

            plt.subplot(1, 2, b + 1)
            plt.imshow(binary_mask, cmap='gray')
            plt.title(f"Detected Mask (Batch {b})")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig("test_visualizations/detected_mask_boundaries.png")
        plt.close()

    # Test full model forward pass with masked inputs
    print("\nTesting full model with masked inputs...")
    with torch.no_grad():
        output = model(hsi_data, aux_data, batch_indices)

        print(f"Forward pass successful!")
        print(f"Reconstruction loss: {output['loss_recon'].item():.4f}")
        print(f"Contrastive loss: {output['loss_contrast'].item():.4f}")
        print(f"Total loss: {output['loss'].item():.4f}")

    print("\nVisualization images saved to 'test_visualizations/' directory")
    print("\n==== Masking Test Complete ====")


def test_variable_mask_shapes():
    """Test the model with various mask shapes for thickness images."""

    print("\n==== Testing Various Mask Shapes ====")

    # Import visualization libraries
    import matplotlib.pyplot as plt
    import os

    # Create output directory for visualizations
    os.makedirs("test_visualizations", exist_ok=True)

    # Basic model parameters
    analysis_dim = 500
    patch_size = (25, 25)
    t_patch_size = 5
    embed_dim = 768

    # Create test data
    batch_size = 3
    hsi_data, aux_data, batch_indices = create_dummy_data(batch_size=batch_size)

    # Create different mask shapes for each batch item
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Circular mask for first batch item
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, 200),
        torch.linspace(-1, 1, 200),
        indexing='ij'
    )
    circle_mask = ((x ** 2 + y ** 2) < 0.6 ** 2).float()
    aux_data['thickness'][0, 0] = aux_data['thickness'][0, 0] * circle_mask

    # 2. Square mask for second batch item
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, 200),
        torch.linspace(-1, 1, 200),
        indexing='ij'
    )
    square_mask = ((abs(x) < 0.7) & (abs(y) < 0.7)).float()
    aux_data['thickness'][1, 0] = aux_data['thickness'][1, 0] * square_mask

    # 3. Elliptical mask for third batch item
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, 200),
        torch.linspace(-1, 1, 200),
        indexing='ij'
    )
    ellipse_mask = (((x ** 2) / 0.7 ** 2 + (y ** 2) / 0.4 ** 2) < 1.0).float()
    aux_data['thickness'][2, 0] = aux_data['thickness'][2, 0] * ellipse_mask

    # Visualize original masks
    plt.figure(figsize=(15, 5))
    mask_types = ["Circular", "Square", "Elliptical"]
    for b in range(batch_size):
        plt.subplot(1, 3, b + 1)
        plt.imshow(aux_data['thickness'][b, 0].cpu().numpy(), cmap='gray')
        plt.title(f"{mask_types[b]} Mask (Original)")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_visualizations/original_masks.png")
    plt.close()

    # Move data to device
    hsi_data = hsi_data.to(device)
    aux_data = {k: v.to(device) for k, v in aux_data.items()}
    batch_indices = batch_indices.to(device)

    # Create model
    model = MultiModalSpectralGPT(
        analysis_dim=analysis_dim,
        patch_size=patch_size,
        embed_dim=embed_dim,
        t_patch_size=t_patch_size,
        in_chans=1,
        aux_chans=1,
        contrastive_mode='global'
    ).to(device)

    # Test spatial registration with different mask shapes
    print("Testing spatial registration with various mask shapes...")
    with torch.no_grad():
        # Apply spatial registration
        hsi_registered, aux_registered = model.spatial_registration(hsi_data, aux_data)

        # Visualize registered modalities for each mask type
        for b in range(batch_size):
            mask_type = mask_types[b]
            print(f"\nBatch {b} ({mask_type.lower()} mask):")

            plt.figure(figsize=(15, 10))

            # Show HSI (first channel)
            plt.subplot(2, 3, 1)
            plt.imshow(hsi_registered[b, 0, 0].cpu().numpy(), cmap='gray')
            plt.title(f"{mask_type} Mask - HSI (Band 0)")
            plt.axis('off')

            # Show all auxiliary modalities
            pos = 2
            for modality in aux_registered:
                if aux_registered[modality] is not None:
                    plt.subplot(2, 3, pos)
                    plt.imshow(aux_registered[modality][b, 0].cpu().numpy(), cmap='gray')
                    plt.title(f"{mask_type} Mask - {modality.upper()}")
                    plt.axis('off')
                    pos += 1

                    # Count non-zero elements to verify masking
                    original_nonzero = (aux_data[modality][b] > 0.05).float().sum().item()
                    registered_nonzero = (aux_registered[modality][b] > 0.05).float().sum().item()

                    # Calculate the percentage of non-zero elements
                    total_elements = aux_registered[modality][b].numel()
                    original_percent = 100 * original_nonzero / total_elements
                    registered_percent = 100 * registered_nonzero / total_elements

                    print(f"{modality}: {registered_percent:.2f}% non-zero elements (was {original_percent:.2f}%)")

            plt.tight_layout()
            plt.savefig(f"test_visualizations/registered_{mask_type.lower()}_mask.png")
            plt.close()

        # Create visualization of the mask detection
        plt.figure(figsize=(15, 5))
        for b in range(batch_size):
            # Create binary mask using same threshold as in model
            binary_mask = (aux_registered['thickness'][b, 0] > 0.05).float().cpu().numpy()

            plt.subplot(1, 3, b + 1)
            plt.imshow(binary_mask, cmap='gray')
            plt.title(f"Detected {mask_types[b]} Mask")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig("test_visualizations/detected_masks.png")
        plt.close()

    # Test full model forward pass with various mask shapes
    print("\nTesting full model with various mask shapes...")
    with torch.no_grad():
        output = model(hsi_data, aux_data, batch_indices)
        print(f"Forward pass successful!")
        print(f"Total loss: {output['loss'].item():.4f}")

    print("\nVisualization images saved to 'test_visualizations/' directory")
    print("\n==== Various Mask Shapes Test Complete ====")


def test_complex_irregular_mask():
    """Test the model with a complex, irregular mask shape."""

    print("\n==== Testing Complex Irregular Mask ====")

    # Import visualization libraries
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from scipy import ndimage

    # Create output directory for visualizations
    os.makedirs("test_visualizations", exist_ok=True)

    # Basic model parameters
    analysis_dim = 500
    patch_size = (25, 25)
    t_patch_size = 5
    embed_dim = 768

    # Create test data
    batch_size = 1
    hsi_data, aux_data, batch_indices = create_dummy_data(batch_size=batch_size)

    # Create a complex irregular mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create an irregular mask using a combination of shapes and noise
    # Start with a base circular mask
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, 200),
        torch.linspace(-1, 1, 200),
        indexing='ij'
    )

    # Create base mask with an irregular shape
    base_mask = ((x ** 2 + y ** 2) < 0.6 ** 2).float().numpy()

    # Add some random "fingers" extending from the circular mask
    for i in range(5):
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0.1, 0.3)
        width = np.random.uniform(0.05, 0.15)

        # Create a "finger" extending from the center
        finger_x = x.numpy() * np.cos(angle) - y.numpy() * np.sin(angle)
        finger_y = x.numpy() * np.sin(angle) + y.numpy() * np.cos(angle)

        finger_mask = (np.abs(finger_y) < width) & (finger_x > 0) & (finger_x < length)
        base_mask = np.logical_or(base_mask, finger_mask).astype(np.float32)

    # Add some noise and blur the edges
    noise = np.random.normal(0, 0.5, size=(200, 200))
    noisy_mask = base_mask + 0.1 * noise

    # Smooth the mask
    smooth_mask = ndimage.gaussian_filter(noisy_mask, sigma=2)

    # Threshold to create final binary mask
    final_mask = (smooth_mask > 0.5).astype(np.float32)

    # Create a "hole" inside the mask
    hole_mask = ((x.numpy() + 0.3) ** 2 + (y.numpy() - 0.2) ** 2 < 0.15 ** 2)
    final_mask[hole_mask] = 0

    # Convert to tensor
    irregular_mask = torch.tensor(final_mask, dtype=torch.float32)

    # Apply the irregular mask to the thickness image
    aux_data['thickness'][0, 0] = aux_data['thickness'][0, 0] * irregular_mask

    # Visualize the irregular mask
    plt.figure(figsize=(10, 10))
    plt.imshow(irregular_mask.numpy(), cmap='gray')
    plt.title("Complex Irregular Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_visualizations/complex_irregular_mask.png")
    plt.close()

    # Visualize the masked thickness image
    plt.figure(figsize=(10, 10))
    plt.imshow(aux_data['thickness'][0, 0].numpy(), cmap='gray')
    plt.title("Thickness Image with Irregular Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_visualizations/thickness_irregular_mask.png")
    plt.close()

    # Move data to device
    hsi_data = hsi_data.to(device)
    aux_data = {k: v.to(device) for k, v in aux_data.items()}
    batch_indices = batch_indices.to(device)

    # Create model
    model = MultiModalSpectralGPT(
        analysis_dim=analysis_dim,
        patch_size=patch_size,
        embed_dim=embed_dim,
        t_patch_size=t_patch_size,
        in_chans=1,
        aux_chans=1,
        contrastive_mode='global'
    ).to(device)

    # Test spatial registration with irregular mask
    print("Testing spatial registration with irregular mask...")
    with torch.no_grad():
        # Apply spatial registration
        hsi_registered, aux_registered = model.spatial_registration(hsi_data, aux_data)

        # Visualize all modalities after masking
        plt.figure(figsize=(15, 10))

        # Show a few HSI bands
        bands_to_show = [0, 10, 20]
        for i, band in enumerate(bands_to_show):
            plt.subplot(2, 3, i + 1)
            plt.imshow(hsi_registered[0, 0, band].cpu().numpy(), cmap='gray')
            plt.title(f"HSI Band {band}")
            plt.axis('off')

        # Show auxiliary modalities
        modalities = list(aux_registered.keys())
        for i, modality in enumerate(modalities[:3]):  # Show up to 3 modalities
            if aux_registered[modality] is not None:
                plt.subplot(2, 3, i + 4)
                plt.imshow(aux_registered[modality][0, 0].cpu().numpy(), cmap='gray')
                plt.title(f"{modality.upper()}")
                plt.axis('off')

        plt.tight_layout()
        plt.savefig("test_visualizations/irregular_mask_all_modalities.png")
        plt.close()

        # Visualize binary mask detection
        binary_mask = (aux_registered['thickness'][0, 0] > 0.05).float().cpu().numpy()

        plt.figure(figsize=(15, 5))

        # Original thickness with irregular mask
        plt.subplot(1, 3, 1)
        plt.imshow(aux_data['thickness'][0, 0].cpu().numpy(), cmap='gray')
        plt.title("Original Thickness w/ Mask")
        plt.axis('off')

        # Detected binary mask
        plt.subplot(1, 3, 2)
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Detected Binary Mask")
        plt.axis('off')

        # Overlay of mask on HSI
        plt.subplot(1, 3, 3)
        plt.imshow(hsi_registered[0, 0, 0].cpu().numpy(), cmap='gray')
        plt.title("HSI with Applied Mask")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("test_visualizations/irregular_mask_detection.png")
        plt.close()

        # Print mask coverage statistics
        for modality in aux_registered:
            if aux_registered[modality] is not None:
                # Count non-zero elements to verify masking
                original_nonzero = (aux_data[modality][0] > 0.05).float().sum().item()
                registered_nonzero = (aux_registered[modality][0] > 0.05).float().sum().item()

                # Calculate the percentage of non-zero elements
                total_elements = aux_registered[modality][0].numel()
                original_percent = 100 * original_nonzero / total_elements
                registered_percent = 100 * registered_nonzero / total_elements

                print(f"{modality}: {registered_percent:.2f}% active region (was {original_percent:.2f}%)")

    # Test full model forward pass with irregular mask
    print("\nTesting full model with irregular mask...")
    with torch.no_grad():
        output = model(hsi_data, aux_data, batch_indices)
        print(f"Forward pass successful!")
        print(f"Reconstruction loss: {output['loss_recon'].item():.4f}")
        print(f"Contrastive loss: {output['loss_contrast'].item():.4f}")
        print(f"Total loss: {output['loss'].item():.4f}")

    print("\nVisualization images saved to 'test_visualizations/' directory")
    print("\n==== Complex Irregular Mask Test Complete ====")


if __name__ == '__main__':
    # Test original functionality
    print("====== TESTING ORIGINAL FUNCTIONALITY ======")
    test_model_original()

    print("\n\n====== TESTING CONTRASTIVE LEARNING MODES ======")
    test_model_contrastive_modes()

    print("\n\n====== TESTING DYNAMIC MASKING ======")
    test_masking()

    print("\n\n====== TESTING VARIABLE MASK SHAPES ======")
    test_variable_mask_shapes()

    print("\n\n====== TESTING COMPLEX IRREGULAR MASK ======")
    test_complex_irregular_mask()

