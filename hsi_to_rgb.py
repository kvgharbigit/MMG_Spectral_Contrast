import numpy as np
import torch


def hsi_to_rgb(hsi_data, wavelength_range=(450, 905)):
    """
    Convert a hyperspectral image to RGB format using CIE color matching functions.

    Args:
        hsi_data: Hyperspectral image tensor of shape [B, C, T, H, W] or [T, H, W]
                 where T is the number of spectral bands
        wavelength_range: Tuple of (min_wavelength, max_wavelength) in nanometers

    Returns:
        RGB image tensor with the same batch and spatial dimensions as input
    """
    # Handle different input shapes
    orig_shape = hsi_data.shape
    device = hsi_data.device if isinstance(hsi_data, torch.Tensor) else None

    # Convert to numpy if it's a torch tensor
    if isinstance(hsi_data, torch.Tensor):
        hsi_np = hsi_data.detach().cpu().numpy()
    else:
        hsi_np = hsi_data

    # Determine the shape and reshape if necessary
    if len(orig_shape) == 5:  # [B, C, T, H, W]
        B, C, T, H, W = orig_shape
        hsi_reshaped = hsi_np.reshape(B * C, T, H, W)
    elif len(orig_shape) == 4:  # [B, T, H, W] or [C, T, H, W]
        if orig_shape[1] <= 3:  # Likely [B, C, H, W]
            raise ValueError("Input appears to be RGB already or doesn't have spectral dimension")
        # Assume [B, T, H, W]
        B, T, H, W = orig_shape
        hsi_reshaped = hsi_np.reshape(B, T, H, W)
    elif len(orig_shape) == 3:  # [T, H, W]
        T, H, W = orig_shape
        hsi_reshaped = hsi_np.reshape(1, T, H, W)
    else:
        raise ValueError(f"Unsupported input shape: {orig_shape}")

    # Calculate wavelengths (equally spaced)
    num_bands = hsi_reshaped.shape[1]
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_bands)

    # Define simplified CIE 1931 color matching functions (sampled at 10nm intervals)
    # Source: http://www.cvrl.org/cmfs.htm (CIE 1931 2-deg)
    cie_wavelengths = np.arange(380, 781, 10)

    # Color matching function values (x̄, ȳ, z̄)
    xyz_cmf = np.array([
        # x̄ values (red response)
        [0.0014, 0.0042, 0.0143, 0.0435, 0.1344, 0.2839, 0.3483, 0.3362, 0.2908, 0.1954,
         0.0956, 0.0320, 0.0049, 0.0093, 0.0633, 0.1655, 0.2904, 0.4334, 0.5945, 0.7621,
         0.9163, 1.0263, 1.0622, 1.0026, 0.8544, 0.6424, 0.4479, 0.2835, 0.1649, 0.0874,
         0.0468, 0.0227, 0.0114, 0.0058, 0.0029, 0.0014, 0.0007, 0.0003, 0.0002, 0.0001],

        # ȳ values (green response)
        [0.0000, 0.0001, 0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.0910,
         0.1390, 0.2080, 0.3230, 0.5030, 0.7100, 0.8620, 0.9540, 0.9950, 0.9950, 0.9520,
         0.8700, 0.7570, 0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1070, 0.0610, 0.0320,
         0.0170, 0.0082, 0.0041, 0.0021, 0.0010, 0.0005, 0.0002, 0.0001, 0.0001, 0.0000],

        # z̄ values (blue response)
        [0.0065, 0.0201, 0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876,
         0.8130, 0.4652, 0.2720, 0.1582, 0.0782, 0.0422, 0.0203, 0.0087, 0.0039, 0.0021,
         0.0017, 0.0011, 0.0008, 0.0003, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    ])

    # Interpolate color matching functions to the wavelengths of our HSI data
    x_interp = np.interp(wavelengths, cie_wavelengths, xyz_cmf[0], left=0, right=0)
    y_interp = np.interp(wavelengths, cie_wavelengths, xyz_cmf[1], left=0, right=0)
    z_interp = np.interp(wavelengths, cie_wavelengths, xyz_cmf[2], left=0, right=0)

    # Normalize the interpolated values to ensure proper conversion
    max_val = max(np.sum(x_interp), np.sum(y_interp), np.sum(z_interp))
    x_norm = x_interp / max_val
    y_norm = y_interp / max_val
    z_norm = z_interp / max_val

    # Initialize XYZ image
    bs = hsi_reshaped.shape[0]
    xyz_img = np.zeros((bs, 3, H, W))

    # Project HSI to XYZ color space
    for b in range(bs):
        for i, (x, y, z) in enumerate(zip(x_norm, y_norm, z_norm)):
            xyz_img[b, 0] += hsi_reshaped[b, i] * x
            xyz_img[b, 1] += hsi_reshaped[b, i] * y
            xyz_img[b, 2] += hsi_reshaped[b, i] * z

    # XYZ to RGB conversion matrix (sRGB D65)
    xyz_to_rgb = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])

    # Convert from XYZ to RGB
    rgb_img = np.zeros_like(xyz_img)
    for b in range(bs):
        for h in range(H):
            for w in range(W):
                rgb_img[b, :, h, w] = xyz_to_rgb @ xyz_img[b, :, h, w]

    # Apply gamma correction (assume linear input)
    rgb_img = np.maximum(rgb_img, 0)  # Clip negative values
    mask = rgb_img <= 0.0031308
    rgb_img[mask] = 12.92 * rgb_img[mask]
    rgb_img[~mask] = 1.055 * np.power(rgb_img[~mask], 1.0 / 2.4) - 0.055

    # Clip to [0, 1] range
    rgb_img = np.clip(rgb_img, 0, 1)

    # Handle reshaping back to original dimensions
    if len(orig_shape) == 5:  # [B, C, T, H, W]
        rgb_img = rgb_img.reshape(B, C, 3, H, W)
    elif len(orig_shape) == 3:  # [T, H, W]
        rgb_img = rgb_img[0]  # Remove batch dimension

    # Convert back to torch tensor if input was a tensor
    if device is not None:
        rgb_img = torch.tensor(rgb_img, device=device, dtype=torch.float32)

    return rgb_img


def simple_hsi_to_rgb(hsi_data):
    """
    A simpler alternative HSI to RGB conversion using three representative bands.

    Args:
        hsi_data: Hyperspectral image tensor of shape [B, C, T, H, W] or [T, H, W]
                 where T is the number of spectral bands (assumed to be 30 bands)

    Returns:
        RGB image tensor with the same batch and spatial dimensions as input
    """
    # Handle different input shapes
    orig_shape = hsi_data.shape
    device = hsi_data.device if isinstance(hsi_data, torch.Tensor) else None

    # Convert to numpy if it's a torch tensor
    if isinstance(hsi_data, torch.Tensor):
        hsi_np = hsi_data.detach().cpu().numpy()
    else:
        hsi_np = hsi_data

    # Determine the shape
    if len(orig_shape) == 5:  # [B, C, T, H, W]
        B, C, T, H, W = orig_shape
        if T < 30:
            raise ValueError(f"Expected at least 30 spectral bands, got {T}")

        # Select bands for RGB approximation
        # Red: ~650nm (band 13-15), Green: ~550nm (band 7-9), Blue: ~450nm (band 0-2)
        r_idx = min(14, T - 1)  # Red band index
        g_idx = min(8, T - 1)  # Green band index
        b_idx = 0  # Blue band index

        # Create RGB image
        rgb_img = np.zeros((B, C, 3, H, W), dtype=hsi_np.dtype)
        rgb_img[:, :, 0] = hsi_np[:, :, r_idx]  # Red channel
        rgb_img[:, :, 1] = hsi_np[:, :, g_idx]  # Green channel
        rgb_img[:, :, 2] = hsi_np[:, :, b_idx]  # Blue channel

    elif len(orig_shape) == 4:  # [B, T, H, W]
        B, T, H, W = orig_shape
        if T < 30:
            raise ValueError(f"Expected at least 30 spectral bands, got {T}")

        r_idx = min(14, T - 1)
        g_idx = min(8, T - 1)
        b_idx = 0

        rgb_img = np.zeros((B, 3, H, W), dtype=hsi_np.dtype)
        rgb_img[:, 0] = hsi_np[:, r_idx]  # Red channel
        rgb_img[:, 1] = hsi_np[:, g_idx]  # Green channel
        rgb_img[:, 2] = hsi_np[:, b_idx]  # Blue channel

    elif len(orig_shape) == 3:  # [T, H, W]
        T, H, W = orig_shape
        if T < 30:
            raise ValueError(f"Expected at least 30 spectral bands, got {T}")

        r_idx = min(14, T - 1)
        g_idx = min(8, T - 1)
        b_idx = 0

        rgb_img = np.zeros((3, H, W), dtype=hsi_np.dtype)
        rgb_img[0] = hsi_np[r_idx]  # Red channel
        rgb_img[1] = hsi_np[g_idx]  # Green channel
        rgb_img[2] = hsi_np[b_idx]  # Blue channel

    else:
        raise ValueError(f"Unsupported input shape: {orig_shape}")

    # Normalize each channel for better visualization
    rgb_min = np.min(rgb_img, axis=tuple(range(len(rgb_img.shape)))[:-2] + tuple(range(len(rgb_img.shape)))[-2:],
                     keepdims=True)
    rgb_max = np.max(rgb_img, axis=tuple(range(len(rgb_img.shape)))[:-2] + tuple(range(len(rgb_img.shape)))[-2:],
                     keepdims=True)
    rgb_range = rgb_max - rgb_min

    # Avoid division by zero
    rgb_range[rgb_range == 0] = 1.0

    # Normalize
    rgb_img = (rgb_img - rgb_min) / rgb_range

    # Convert back to torch tensor if input was a tensor
    if device is not None:
        rgb_img = torch.tensor(rgb_img, device=device, dtype=torch.float32)

    return rgb_img