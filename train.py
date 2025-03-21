import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import time
import glob
import torch
import numpy as np
import math
import pandas as pd
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import hydra.utils
from datetime import datetime
from pathlib import Path


# Import custom modules
from MultiModalSpectralGPT import MultiModalSpectralGPT
from dataset import PatientDataset, custom_collate_fn, create_patient_dataloader
from hsi_to_rgb import hsi_to_rgb, simple_hsi_to_rgb  # These are already imported elsewhere
from dataset import MultiModalTransforms

from visualisation import visualize_batch
import matplotlib.cm as cm

# Configure matplotlib for non-GUI environments
plt.switch_backend('agg')


import torch
import os

# Optional: only necessary if you want to lock to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add a global flag to track memory reservation status
_MEMORY_RESERVED = False

# Reserve GPU memory by allocating a large dummy tensor
def reserve_gpu_memory(device_id=0, target_gb=10):
    """Allocates memory to effectively reserve the GPU."""
    global _MEMORY_RESERVED

    # If memory is already reserved, skip allocation
    if _MEMORY_RESERVED:
        print("GPU memory already reserved. Skipping.")
        return

    device = torch.device(f"cuda:{device_id}")
    tensor_list = []

    try:
        # Try allocating chunks to fill up the GPU
        chunk_size_mb = 512
        while True:
            tensor = torch.empty((chunk_size_mb * 1024 * 1024 // 4,), dtype=torch.float32, device=device)
            tensor_list.append(tensor)
            allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3
            print(f"[GPU {device_id}] Reserved ~{allocated:.2f} GB")
            if allocated >= target_gb:
                break

        # Set the flag to indicate memory has been reserved
        _MEMORY_RESERVED = True
        print("Memory reservation complete.")
    except RuntimeError as e:
        print(f"[!] Stopped allocating: {e}")




def get_warmup_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr, base_lr):
    """
    Creates a learning rate schedule with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer to use
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate at the end of training
        base_lr: Base learning rate after warmup

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step):
        # Linear warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay phase
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr / base_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def visualize_reconstruction(model, test_batch, epoch, output_dir, max_samples=2, include_aux=True):
    """
    Visualize the original and reconstructed HSI data for evaluation.

    Args:
        model: The trained model
        test_batch: A batch of test data
        epoch: Current epoch number
        output_dir: Directory to save visualizations
        max_samples: Maximum number of samples to visualize
        include_aux: Whether to include auxiliary modalities in visualization

    Returns:
        str: Path to the saved visualization file
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # Create visualization directory if it doesn't exist
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Ensure model is in evaluation mode
    model.eval()

    # Limit the number of samples to visualize
    batch_size = min(max_samples, test_batch['hsi'].shape[0])

    # Get device
    device = next(model.parameters()).device

    # Move data to device
    hsi = test_batch['hsi'][:batch_size].to(device)
    aux_data = {k: v[:batch_size].to(device) if v is not None else None
                for k, v in test_batch['aux_data'].items()}
    batch_idx = test_batch['batch_idx'][:batch_size].to(device)

    # Save patient IDs for labeling
    patient_ids = test_batch['patient_id'][:batch_size] if 'patient_id' in test_batch else [f"Sample_{i}" for i in
                                                                                            range(batch_size)]

    # Get model prediction with no gradient tracking
    with torch.no_grad():
        output = model(hsi, aux_data, batch_idx)

        # Extract relevant outputs
        pred = output['pred']  # Predicted embeddings
        mask = output['mask']  # MAE mask (1 is masked, 0 is kept)

        # Check if model has a pixel_decoder (needed for reconstruction)
        if not hasattr(model, 'pixel_decoder'):
            raise ValueError(
                "Model does not have a pixel_decoder. A trained pixel decoder is required for visualization.")

    # Determine number of columns based on what to include
    n_cols = 4  # Original, Mask, Masked, Reconstructed

    if include_aux:
        # Count available auxiliary modalities
        n_aux = sum(1 for v in aux_data.values() if v is not None)
        if n_aux > 0:
            n_cols += n_aux

    # Create a larger figure to prevent overlapping
    # Increase the figure size to give more space between columns
    fig_width = 5 * n_cols  # 5 inches per column
    fig_height = 5 * batch_size  # 5 inches per row

    # Set up the figure with more space
    fig, axes = plt.subplots(batch_size, n_cols, figsize=(fig_width, fig_height))
    fig.suptitle(f"Epoch {epoch} - HSI Reconstruction", fontsize=16)

    # If only one sample, make axes 2D
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    # Add more space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # For each sample in the batch
    for i in range(batch_size):
        col_idx = 0

        # 1. Plot original HSI as RGB
        try:
            # Get the original HSI and convert to RGB using existing function
            orig_hsi = hsi[i].unsqueeze(0)  # Add batch dimension back

            # Debug: print shape before conversion
            print(f"Original HSI shape before RGB conversion: {orig_hsi.shape}")

            orig_rgb = simple_hsi_to_rgb(orig_hsi)  # Use the imported function

            # Debug: print shape after conversion
            print(f"RGB shape after conversion: {orig_rgb.shape}")

            # Convert to numpy and fully remove all batch dimensions
            if len(orig_rgb.shape) == 5:  # If shape is [B, C, 3, H, W]
                orig_rgb_np = orig_rgb[0, 0].cpu().numpy()  # Get [3, H, W]
            elif len(orig_rgb.shape) == 4:  # If shape is [B, 3, H, W]
                orig_rgb_np = orig_rgb[0].cpu().numpy()  # Get [3, H, W]
            else:
                orig_rgb_np = orig_rgb.cpu().numpy()  # Already correct shape

            # Always transpose to [H, W, 3] if we have [3, H, W]
            if len(orig_rgb_np.shape) == 3 and orig_rgb_np.shape[0] == 3:
                orig_rgb_np = np.transpose(orig_rgb_np, (1, 2, 0))

            # Debug: print final shape
            print(f"Final RGB shape for plotting: {orig_rgb_np.shape}")

            # Ensure valid range for display
            orig_rgb_np = np.clip(orig_rgb_np, 0, 1)

            axes[i, col_idx].imshow(orig_rgb_np)
            axes[i, col_idx].set_title("Original HSI", fontsize=12)
            axes[i, col_idx].axis('off')
        except Exception as e:
            print(f"Error creating original RGB visualization: {e}")
            import traceback
            traceback.print_exc()
            axes[i, col_idx].text(0.5, 0.5, "Error visualizing HSI",
                                  ha='center', va='center', transform=axes[i, col_idx].transAxes)
        col_idx += 1

        # 2. Visualize the MAE mask
        try:
            # Get the original patch tokens
            orig_tokens = model.patch_embed(orig_hsi)
            B, T, HW, D = orig_tokens.shape
            orig_tokens = orig_tokens.reshape(B, T * HW, D)

            # Create a spatial visualization of the mask
            patch_mask = mask[i].clone().cpu().numpy()

            # Reshape the mask to match the spatial structure of patches
            patch_h, patch_w = model.patch_size
            t_patch_size = model.t_patch_size
            num_patches_h = model.patches_per_dim
            num_patches_w = model.patches_per_dim
            num_patches_t = model.spectral_patches

            # Need to reshape the 1D mask into a 3D grid (T, H, W)
            # First reshape to spectral patches × spatial patches
            if len(patch_mask) == num_patches_t * num_patches_h * num_patches_w:
                spatial_mask = patch_mask.reshape(num_patches_t, num_patches_h, num_patches_w)

                # Average across spectral dimension for visualization
                spatial_mask_2d = np.mean(spatial_mask, axis=0)
            else:
                # If mask shape doesn't match expected dimensions, just reshape to square
                side_len = int(np.sqrt(len(patch_mask)))
                spatial_mask_2d = patch_mask.reshape(side_len, side_len)

            # Plot the mask
            mask_img = axes[i, col_idx].imshow(spatial_mask_2d, cmap='hot', interpolation='nearest')
            axes[i, col_idx].set_title(f"MAE Mask ({mask[i].sum().item():.0f} masked tokens)", fontsize=12)
            axes[i, col_idx].axis('off')

            # Add a colorbar for the mask
            plt.colorbar(mask_img, ax=axes[i, col_idx], fraction=0.046, pad=0.04)

        except Exception as e:
            print(f"Error visualizing mask: {e}")
            import traceback
            traceback.print_exc()
            axes[i, col_idx].text(0.5, 0.5, "Error visualizing mask",
                                  ha='center', va='center', transform=axes[i, col_idx].transAxes)
            axes[i, col_idx].axis('off')
        col_idx += 1

        # 3. Create and plot masked version that better shows spectral masking effects
        try:
            # Apply the mask from model output to the tokens (set masked tokens to zero)
            masked_tokens = orig_tokens.clone()
            mask_idx = torch.where(mask[i] > 0.5)[0]
            masked_tokens[0, mask_idx] = 0.0

            # Reshape to [B*L, D] format expected by the pixel decoder
            B, L, D = masked_tokens.shape
            masked_tokens_flat = masked_tokens.reshape(-1, D)

            # Use the model's pixel decoder to get patch pixels
            pixels_flat = model.pixel_decoder(masked_tokens_flat)

            # Extract necessary dimensions
            C = 1  # Channels (HSI usually has 1 channel)
            T = model.spectral_patches * model.t_patch_size  # Total spectral bands
            H = model.patches_per_dim * model.patch_size[0]  # Height
            W = model.patches_per_dim * model.patch_size[1]  # Width

            # Calculate patch dimensions
            patch_h, patch_w = model.patch_size
            t_patch_size = model.t_patch_size
            num_patches_h = H // patch_h
            num_patches_w = W // patch_w
            num_patches_t = T // t_patch_size
            patch_volume = C * t_patch_size * patch_h * patch_w

            # Reshape to B, L, C, t_patch_size, patch_h, patch_w
            pixels = pixels_flat.reshape(B, L, C, t_patch_size, patch_h, patch_w)

            # Initialize output tensor
            masked_hsi = torch.zeros((B, C, T, H, W), device=device)

            # Create a mask density tensor to track which spatial locations have masked bands
            mask_density = torch.zeros((H, W), device=device)

            # Place each patch in the correct position
            patch_idx = 0
            for t_idx in range(num_patches_t):
                t_start = t_idx * t_patch_size
                for h_idx in range(num_patches_h):
                    h_start = h_idx * patch_h
                    for w_idx in range(num_patches_w):
                        w_start = w_idx * patch_w

                        if patch_idx < pixels.shape[1]:
                            # Get this patch
                            patch = pixels[:, patch_idx]

                            # Place in output tensor
                            masked_hsi[:, :, t_start:t_start + t_patch_size,
                            h_start:h_start + patch_h,
                            w_start:w_start + patch_w] = patch

                            # Check if this patch was masked and update mask density
                            if patch_idx in mask_idx:
                                mask_density[h_start:h_start + patch_h, w_start:w_start + patch_w] += 1.0

                        patch_idx += 1

            # Normalize mask density by the number of spectral patches
            mask_density = mask_density / num_patches_t

            # Debug: print shape before RGB conversion
            print(f"Masked HSI shape before RGB conversion: {masked_hsi.shape}")

            # Convert to RGB for visualization using the imported function
            masked_rgb = simple_hsi_to_rgb(masked_hsi)

            # Debug: print shape after conversion
            print(f"Masked RGB shape after conversion: {masked_rgb.shape}")

            # Convert to numpy with proper handling of dimensions
            if len(masked_rgb.shape) == 5:  # If shape is [B, C, 3, H, W]
                masked_rgb_np = masked_rgb[0, 0].cpu().numpy()
            elif len(masked_rgb.shape) == 4:  # If shape is [B, 3, H, W]
                masked_rgb_np = masked_rgb[0].cpu().numpy()
            else:
                masked_rgb_np = masked_rgb.cpu().numpy()

            # Always transpose to [H, W, 3] if we have [3, H, W]
            if len(masked_rgb_np.shape) == 3 and masked_rgb_np.shape[0] == 3:
                masked_rgb_np = np.transpose(masked_rgb_np, (1, 2, 0))

            # Debug: print final shape
            print(f"Final masked RGB shape for plotting: {masked_rgb_np.shape}")

            # Ensure values are valid for display
            masked_rgb_np = np.clip(masked_rgb_np, 0, 1)

            # Create a more informative visualization
            # Blend the original RGB with red color based on mask density
            mask_density_np = mask_density.cpu().numpy()
            enhanced_masked_rgb = masked_rgb_np.copy()

            # Areas with high mask density will appear more red-tinted
            for c in range(3):
                if c == 0:  # Red channel
                    # Keep red channel more visible
                    enhanced_masked_rgb[:, :, c] = enhanced_masked_rgb[:, :, c] * (1.0 - mask_density_np * 0.5)
                else:  # Green and Blue channels
                    # Reduce green and blue more to create red tint in masked areas
                    enhanced_masked_rgb[:, :, c] = enhanced_masked_rgb[:, :, c] * (1.0 - mask_density_np)

            # Plot the enhanced masked version
            axes[i, col_idx].imshow(enhanced_masked_rgb)
            axes[i, col_idx].set_title("Masked HSI (red = masked)", fontsize=12)
            axes[i, col_idx].axis('off')
        except Exception as e:
            print(f"Error creating masked HSI visualization: {e}")
            import traceback
            traceback.print_exc()
            axes[i, col_idx].text(0.5, 0.5, "Error visualizing masked HSI",
                                  ha='center', va='center', transform=axes[i, col_idx].transAxes)
        col_idx += 1

        # 4. Reconstruct and plot the reconstructed HSI
        try:
            # Reconstruct HSI from model prediction
            recon_tokens = pred[i].unsqueeze(0)  # Add batch dimension [1, L, D]

            # Reshape for pixel decoder
            B, L, D = recon_tokens.shape
            recon_tokens_flat = recon_tokens.reshape(-1, D)

            # Use the model's pixel decoder for reconstruction
            pixels_flat = model.pixel_decoder(recon_tokens_flat)

            # Extract necessary dimensions
            C = 1  # Channels (HSI usually has 1 channel)
            T = model.spectral_patches * model.t_patch_size  # Total spectral bands
            H = model.patches_per_dim * model.patch_size[0]  # Height
            W = model.patches_per_dim * model.patch_size[1]  # Width

            # Calculate patch dimensions
            patch_h, patch_w = model.patch_size
            t_patch_size = model.t_patch_size
            num_patches_h = H // patch_h
            num_patches_w = W // patch_w
            num_patches_t = T // t_patch_size
            patch_volume = C * t_patch_size * patch_h * patch_w

            # Reshape to B, L, patch_components
            pixels = pixels_flat.reshape(B, L, C, t_patch_size, patch_h, patch_w)

            # Initialize output tensor
            recon_hsi = torch.zeros((B, C, T, H, W), device=device)

            # Place each patch in the correct position
            patch_idx = 0
            for t_idx in range(num_patches_t):
                t_start = t_idx * t_patch_size
                for h_idx in range(num_patches_h):
                    h_start = h_idx * patch_h
                    for w_idx in range(num_patches_w):
                        w_start = w_idx * patch_w

                        if patch_idx < pixels.shape[1]:
                            # Get this patch
                            patch = pixels[:, patch_idx]

                            # Place in output tensor
                            recon_hsi[:, :, t_start:t_start + t_patch_size,
                            h_start:h_start + patch_h,
                            w_start:w_start + patch_w] = patch

                        patch_idx += 1



            # Debug: print shape before RGB conversion
            print(f"Reconstructed HSI shape before RGB conversion: {recon_hsi.shape}")

            # Convert to RGB for visualization using the imported function
            recon_rgb = simple_hsi_to_rgb(recon_hsi)

            # Debug: print shape after conversion
            print(f"Reconstructed RGB shape after conversion: {recon_rgb.shape}")

            # Convert to numpy with proper dimension handling
            if len(recon_rgb.shape) == 5:  # If shape is [B, C, 3, H, W]
                recon_rgb_np = recon_rgb[0, 0].cpu().numpy()
            elif len(recon_rgb.shape) == 4:  # If shape is [B, 3, H, W]
                recon_rgb_np = recon_rgb[0].cpu().numpy()
            else:
                recon_rgb_np = recon_rgb.cpu().numpy()

            # Always transpose to [H, W, 3] if we have [3, H, W]
            if len(recon_rgb_np.shape) == 3 and recon_rgb_np.shape[0] == 3:
                recon_rgb_np = np.transpose(recon_rgb_np, (1, 2, 0))

            # Debug: print final shape
            print(f"Final reconstructed RGB shape for plotting: {recon_rgb_np.shape}")

            # Ensure valid range for display
            recon_rgb_np = np.clip(recon_rgb_np, 0, 1)

            axes[i, col_idx].imshow(recon_rgb_np)
            axes[i, col_idx].set_title("Reconstructed HSI", fontsize=12)
            axes[i, col_idx].axis('off')
        except Exception as e:
            print(f"Error creating reconstructed HSI visualization: {e}")
            import traceback
            traceback.print_exc()
            axes[i, col_idx].text(0.5, 0.5, "Error visualizing reconstruction",
                                  ha='center', va='center', transform=axes[i, col_idx].transAxes)
        col_idx += 1

        # 5. Plot auxiliary modalities if available and requested
        if include_aux:
            for modality, data in aux_data.items():
                if data is not None:
                    try:
                        # Get auxiliary data for this sample
                        aux_img = data[i, 0].cpu().numpy()  # Assuming shape [B, 1, H, W]

                        # Display the auxiliary data
                        axes[i, col_idx].imshow(aux_img, cmap='gray')
                        axes[i, col_idx].set_title(f"{modality.upper()}", fontsize=12)
                        axes[i, col_idx].axis('off')
                    except Exception as e:
                        print(f"Error displaying {modality} data: {e}")
                        axes[i, col_idx].text(0.5, 0.5, f"Error: {modality}",
                                              ha='center', va='center', transform=axes[i, col_idx].transAxes)
                    col_idx += 1

    # Add sample identifiers
    for i, patient_id in enumerate(patient_ids):
        # Add text to the left side of the row
        plt.figtext(0.01, (batch_size - i - 0.5) / batch_size, f"ID: {patient_id}",
                    va='center', ha='left', fontsize=10)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, left=0.05, wspace=0.3, hspace=0.3)

    # Save the figure
    save_path = os.path.join(viz_dir, f"reconstruction_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Create a higher quality visualization for just one sample
    if batch_size > 0:
        try:
            # Create a detailed visualization for the first sample with more spacing
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            fig.suptitle(f"Epoch {epoch} - Detailed Reconstruction (ID: {patient_ids[0]})", fontsize=16)

            # Add more space between subplots
            plt.subplots_adjust(wspace=0.3)

            # Original HSI
            axes[0].imshow(orig_rgb_np)
            axes[0].set_title("Original HSI", fontsize=14)
            axes[0].axis('off')

            # MAE Mask
            mask_img = axes[1].imshow(spatial_mask_2d, cmap='hot', interpolation='nearest')
            axes[1].set_title(f"MAE Mask ({mask[0].sum().item():.0f} tokens)", fontsize=14)
            axes[1].axis('off')
            plt.colorbar(mask_img, ax=axes[1], fraction=0.046, pad=0.04)

            # Masked HSI with enhanced visualization
            axes[2].imshow(enhanced_masked_rgb)
            axes[2].set_title("Masked HSI (red = masked)", fontsize=14)
            axes[2].axis('off')

            # Reconstructed HSI
            axes[3].imshow(recon_rgb_np)
            axes[3].set_title("Reconstructed HSI", fontsize=14)
            axes[3].axis('off')

            # Save high-quality version
            hq_save_path = os.path.join(viz_dir, f"detailed_reconstruction_epoch_{epoch}.png")
            plt.savefig(hq_save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error creating detailed visualization: {e}")
            import traceback
            traceback.print_exc()

    print(f"Reconstructions visualized and saved to {save_path}")
    return save_path


def calculate_metrics(outputs, optimizer=None):
    """Calculate aggregate metrics from a list of model outputs."""
    metrics = {
        'loss': 0.0,
        'loss_recon': 0.0,
        'loss_contrast': 0.0,
        'num_modalities': 0.0,
    }

    # Add current learning rate if optimizer is provided
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            metrics['learning_rate'] = param_group['lr']
            break  # Just take the first group's learning rate

    batch_count = len(outputs)
    if batch_count == 0:
        return metrics

    # Sum the metrics across all batches
    for output in outputs:
        metrics['loss'] += output['loss'].item()
        metrics['loss_recon'] += output['loss_recon'].item()
        metrics['loss_contrast'] += output['loss_contrast'].item()
        metrics['num_modalities'] += output['num_modalities'].item()

    # Calculate the average for each metric (except learning_rate)
    for key in metrics:
        if key != 'learning_rate':
            metrics[key] /= batch_count

    return metrics


def log_metrics(split, metrics, epoch, writer, mlflow_logging=True):
    """
    Log metrics to both TensorBoard and MLflow.
    
    Args:
        split: String indicating 'train' or 'val'
        metrics: Dictionary of metric values
        epoch: Current epoch number
        writer: TensorBoard SummaryWriter instance
        mlflow_logging: Whether to log to MLflow
    """
    # Log to TensorBoard
    for key, value in metrics.items():
        writer.add_scalar(f"{split}/{key}", value, epoch)
    
    # Log to MLflow
    if mlflow_logging:
        for key, value in metrics.items():
            mlflow.log_metric(f"{split}_{key}", value, step=epoch)


def log_reconstruction(recon_path, epoch, writer, mlflow_logging=True):
    """
    Log reconstruction visualization to TensorBoard and MLflow.

    Args:
        recon_path: Path to reconstruction image
        epoch: Current epoch number
        writer: TensorBoard SummaryWriter instance
        mlflow_logging: Whether to log to MLflow
    """
    if recon_path is None or not os.path.exists(recon_path):
        print(f"No reconstruction image found at epoch {epoch}")
        return

    try:
        # Log to TensorBoard
        img = plt.imread(recon_path)
        if img.shape[2] == 4:  # RGBA image with alpha channel
            img = img[:, :, :3]  # Remove alpha channel

        # TensorBoard expects images in [C, H, W] format
        img = np.transpose(img, (2, 0, 1))
        writer.add_image(f"Reconstruction/epoch_{epoch}", img, epoch, dataformats='CHW')

        # Log to MLflow if enabled
        if mlflow_logging:
            try:
                import mlflow
                mlflow.log_artifact(recon_path, f"reconstructions/epoch_{epoch}")
            except Exception as e:
                print(f"Error logging reconstruction to MLflow: {e}")

    except Exception as e:
        print(f"Error logging reconstruction image: {e}")


def save_metrics_to_csv(metrics_dict, output_dir, epoch, split='train'):
    """
    Save metrics to a CSV file.
    
    Args:
        metrics_dict: Dictionary of metric values
        output_dir: Directory to save the CSV file
        epoch: Current epoch number
        split: String indicating 'train' or 'val'
    """
    # Create metrics directory if it doesn't exist
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Prepare metrics data
    metrics_data = {key: [value] for key, value in metrics_dict.items()}
    metrics_data['epoch'] = [epoch]
    metrics_df = pd.DataFrame(metrics_data)
    
    # Define file path
    csv_path = os.path.join(metrics_dir, f"{split}_metrics.csv")
    
    # If file exists, append to it; otherwise, create a new file
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    # Save to CSV
    metrics_df.to_csv(csv_path, index=False)


def update_best_metrics(val_metrics, epoch, summaries_dir, current_lr=None):
    """
    Update the best metrics file if the current epoch has better validation loss.
    Robust version that handles various file reading scenarios.
    """
    best_path = os.path.join(summaries_dir, "best_metrics.txt")
    current_val_loss = val_metrics['loss']

    # Default best loss to current loss if no previous best exists
    best_loss = current_val_loss
    best_epoch = epoch

    # Check if best metrics file exists and can be read
    try:
        if os.path.exists(best_path):
            with open(best_path, 'r') as f:
                lines = f.readlines()

                # Look for a line with 'loss:' in it
                for line in lines:
                    if 'loss:' in line.lower():
                        try:
                            best_loss = float(line.split(':')[1].strip())
                            break
                        except (ValueError, IndexError):
                            pass
    except (IOError, OSError) as e:
        print(f"Error reading best metrics file: {e}")
        best_loss = current_val_loss

    # Only update if current loss is better (lower)
    if current_val_loss < best_loss:
        try:
            with open(best_path, 'w') as f:
                f.write("Best Metrics\n")
                f.write(f"Epoch: {epoch}\n")
                if current_lr is not None:
                    f.write(f"Learning Rate: {current_lr:.8f}\n")
                f.write("Validation Metrics:\n")
                for key, value in val_metrics.items():
                    if key != 'learning_rate':
                        f.write(f"  {key}: {value:.10f}\n")
        except (IOError, OSError) as e:
            print(f"Error writing best metrics file: {e}")


def save_epoch_summary(train_metrics, val_metrics, epoch, output_dir, total_time):
    """Save a summary of the epoch's metrics to a text file."""
    # Create summaries directory if it doesn't exist
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)

    # Define file path
    summary_path = os.path.join(summaries_dir, f"epoch_{epoch}_summary.txt")
    best_path = os.path.join(summaries_dir, "best_metrics.txt")

    try:
        with open(summary_path, 'w') as f:
            f.write(f"Epoch {epoch} Summary\n")
            f.write("=" * 50 + "\n\n")

            # Print learning rate prominently if available
            if 'learning_rate' in train_metrics:
                f.write(f"Learning Rate: {train_metrics['learning_rate']:.8f}\n\n")

            f.write("Training Metrics:\n")
            for key, value in train_metrics.items():
                if key != 'learning_rate':  # Skip learning_rate as it's already shown above
                    f.write(f"  {key}: {value:.10f}\n")

            f.write("\nValidation Metrics:\n")
            for key, value in val_metrics.items():
                if key != 'learning_rate':
                    f.write(f"  {key}: {value:.10f}\n")

            f.write(f"\nTime taken: {total_time:.2f} seconds\n")

            # Calculate improvements from previous best metrics
            try:
                if os.path.exists(best_path):
                    with open(best_path, 'r') as best_file:
                        best_lines = best_file.readlines()

                        # Look for a line with loss
                        for line in best_lines:
                            if 'loss:' in line.lower():
                                try:
                                    best_loss = float(line.split(':')[1].strip())
                                    improvement = best_loss - val_metrics['loss']
                                    f.write(f"\nImprovement in validation loss: {improvement:.10f}")
                                    break
                                except (ValueError, IndexError):
                                    pass
            except (IOError, OSError) as e:
                print(f"Error reading best metrics for improvement: {e}")

        # Update best metrics if this is the best epoch so far
        update_best_metrics(val_metrics, epoch, summaries_dir,
                            current_lr=train_metrics.get('learning_rate', None))

    except Exception as e:
        print(f"Error in save_epoch_summary: {e}")
        import traceback
        traceback.print_exc()


def check_early_stopping(val_losses, patience, min_delta=0.0):
    """
    Check if training should be stopped based on validation loss.
    
    Args:
        val_losses: List of validation losses
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        
    Returns:
        Boolean indicating whether to stop training
    """
    if len(val_losses) <= patience:
        return False
    
    # Check if validation loss hasn't improved for 'patience' epochs
    best_loss = min(val_losses[:-patience])
    recent_best = min(val_losses[-patience:])
    
    # Return True if there's been no improvement
    return (best_loss - recent_best) < min_delta


def train_epoch(model, dataloader, optimizer, device, contrastive_mode=None):
    """Train the model for one epoch."""
    model.train()
    outputs = []

    # Set contrastive mode if specified
    if contrastive_mode is not None:
        model.contrastive_mode = contrastive_mode

    # Create progress bar
    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        # Move data to device
        hsi = batch['hsi'].to(device)
        aux_data = {k: v.to(device) if v is not None else None for k, v in batch['aux_data'].items()}
        batch_idx = batch['batch_idx'].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(hsi, aux_data, batch_idx)
        loss = output['loss']

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Get current learning rate for progress bar
        current_lr = optimizer.param_groups[0]['lr']

        # Update progress bar with more decimal places
        pbar.set_postfix({
            'loss': f"{loss.item():.10f}",
            'recon_loss': f"{output['loss_recon'].item():.10f}",
            'contrast_loss': f"{output['loss_contrast'].item():.10f}",
            'lr': f"{current_lr}\n"  # Add a newline character after the learning rate
        })

        # Store outputs
        outputs.append(output)

    return outputs


def validate_epoch(model, dataloader, device, contrastive_mode=None):
    """
    Validate the model on the validation set.

    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        device: Device to validate on
        contrastive_mode: Contrastive mode to use (if None, use model's default)

    Returns:
        List of outputs from each batch
    """
    model.eval()
    outputs = []

    # Set contrastive mode if specified
    if contrastive_mode is not None:
        model.contrastive_mode = contrastive_mode

    # Create progress bar
    pbar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for batch in pbar:
            # Move data to device
            hsi = batch['hsi'].to(device)
            aux_data = {k: v.to(device) if v is not None else None for k, v in batch['aux_data'].items()}
            batch_idx = batch['batch_idx'].to(device)

            # Forward pass
            output = model(hsi, aux_data, batch_idx)

            # Update progress bar with more decimal places
            pbar.set_postfix({
                'loss': f"{output['loss'].item():.10f}",
                'recon_loss': f"{output['loss_recon'].item():.10f}",
                'contrast_loss': f"{output['loss_contrast'].item():.10f}"
            })

            # Store outputs
            outputs.append(output)

    return outputs

def save_checkpoint(model, optimizer, epoch, val_loss, output_dir, is_best=False):
    """
    Save model checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        val_loss: Validation loss
        output_dir: Directory to save the checkpoint
        is_best: Whether this is the best model so far
    """
    # Create checkpoints directory if it doesn't exist
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }

    # Only save the checkpoint if it's the best so far
    if is_best:
        best_path = os.path.join(checkpoints_dir, "best_model.pth")
        torch.save(checkpoint, best_path)
        print(f"Saved new best model with validation loss: {val_loss:.10f}")

@hydra.main(config_path="configs", config_name="train",version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    # Get original working directory for path handling
    original_cwd = hydra.utils.get_original_cwd()

    # Convert data paths to absolute paths if they're relative
    if not os.path.isabs(cfg.data.parent_dir):
        cfg.data.parent_dir = os.path.join(original_cwd, cfg.data.parent_dir)
    if not os.path.isabs(cfg.data.train_dir):
        cfg.data.train_dir = os.path.join(original_cwd, cfg.data.train_dir)
    if not os.path.isabs(cfg.data.val_dir):
        cfg.data.val_dir = os.path.join(original_cwd, cfg.data.val_dir)

    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run it
    reserve_gpu_memory(device_id=0, target_gb=10.5)

    # Create output directory
    output_dir = os.getcwd()  # Hydra changes working dir to outputs/{date}/...
    print(f"Output directory: {output_dir}")

    # Set up MLflow
    if cfg.logging.use_mlflow:
        mlflow.set_experiment(cfg.logging.experiment_name)
        mlflow.start_run(run_name=cfg.logging.run_name)

        # Log Hydra config
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

    # Set up TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    # Load datasets
    if cfg.data.use_auto_split:
        print(f"Loading dataset with auto split: {cfg.data.auto_split_ratio}")

        # Load the entire dataset
        dataset = PatientDataset(
            parent_dir=cfg.data.parent_dir,
            analysis_dim=cfg.model.analysis_dim,
            target_bands=cfg.model.num_frames
        )

        # Split into train and validation sets
        val_size = int(len(dataset) * cfg.data.auto_split_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.seed)
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            drop_last=cfg.data.drop_last,
        )

        # Add augmentation transform to the train_dataset
        if cfg.data.use_augmentation:
            train_dataset.dataset.transform = MultiModalTransforms(
                prob=cfg.data.augmentation.prob,
                rotation_degrees=cfg.data.augmentation.rotation_degrees,
                scale_range=cfg.data.augmentation.scale_range
            )


        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )

        print(f"Dataset split: {train_size} training, {val_size} validation")

    else:
        print(f"Using predefined split: {cfg.data.train_dir} and {cfg.data.val_dir}")

        # For predefined split:
        train_loader = create_patient_dataloader(
            parent_dir=cfg.data.train_dir,
            analysis_dim=cfg.model.analysis_dim,
            target_bands=cfg.model.num_frames,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=True,
            augment=cfg.data.use_augmentation  # Enable augmentation for training
        )

        val_loader = create_patient_dataloader(
            parent_dir=cfg.data.val_dir,
            analysis_dim=cfg.model.analysis_dim,
            target_bands=cfg.model.num_frames,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=False,
            augment=False  # No augmentation for validation
        )

    # Create and initialize model
    model = MultiModalSpectralGPT(
        analysis_dim=cfg.model.analysis_dim,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        decoder_embed_dim=cfg.model.decoder_embed_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        num_frames=cfg.model.num_frames,
        t_patch_size=cfg.model.t_patch_size,
        in_chans=cfg.model.in_chans,
        aux_chans=cfg.model.aux_chans,
        aux_embed_dim=cfg.model.aux_embed_dim,
        temperature=cfg.model.temperature,
        mask_ratio=cfg.model.mask_ratio,
        contrastive_mode=cfg.model.contrastive_mode,
        use_thickness_mask=cfg.model.use_thickness_mask
    )

    # Move model to device
    model = model.to(device)

    for param in model.parameters():
        param.data = param.data.to(device)

    # Print model summary
    print(f"Model initialized: {model.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
    )

    # Create learning rate scheduler
    scheduler_step_frequency = "epoch"  # Default
    if cfg.scheduler.use_scheduler:
        if cfg.scheduler.type == "cosine":
            # Check if warmup is enabled through config
            if hasattr(cfg.scheduler, 'warmup_ratio') and cfg.scheduler.warmup_ratio > 0:
                # Calculate warmup steps
                warmup_ratio = cfg.scheduler.warmup_ratio
                warmup_steps = int(warmup_ratio * cfg.training.epochs * len(train_loader))
                total_steps = cfg.training.epochs * len(train_loader)

                # Create warmup-cosine scheduler
                scheduler = get_warmup_cosine_schedule(
                    optimizer,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    min_lr=cfg.scheduler.min_lr,
                    base_lr=cfg.optimizer.lr
                )
                scheduler_step_frequency = "batch"
            else:
                # Standard cosine scheduler without warmup
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cfg.training.epochs,
                    eta_min=cfg.scheduler.min_lr
                )
        elif cfg.scheduler.type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.scheduler.step_size,
                gamma=cfg.scheduler.gamma
            )
        elif cfg.scheduler.type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=cfg.scheduler.factor,
                patience=cfg.scheduler.patience,
                min_lr=cfg.scheduler.min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler type: {cfg.scheduler.type}")
    else:
        scheduler = None
        scheduler_step_frequency = None

    # Resume from checkpoint if specified
    start_epoch = 0
    val_losses = []
    best_val_loss = float('inf')

    if cfg.training.resume_from_checkpoint:
        checkpoint_path = cfg.training.checkpoint_path
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")

    # Training loop
    print(f"Starting training for {cfg.training.epochs} epochs")

    # In your main training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        epoch_start_time = time.time()

        # Training phase
        train_outputs = train_epoch(
            model, train_loader, optimizer, device,
            contrastive_mode=cfg.model.contrastive_mode
        )
        train_metrics = calculate_metrics(train_outputs, optimizer)  # Now passing optimizer

        # Validation phase
        val_outputs = validate_epoch(
            model, val_loader, device,
            contrastive_mode=cfg.model.contrastive_mode
        )
        val_metrics = calculate_metrics(val_outputs, optimizer)  # Also pass optimizer here



        # Update learning rate scheduler - only for epoch-based schedulers
        if scheduler is not None and scheduler_step_frequency == "epoch":
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()



        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Update the print statement to include learning rate
        print(f"Train Loss: {train_metrics['loss']:.10f}, "
              f"Val Loss: {val_metrics['loss']:.10f}, "
              f"LR: {train_metrics.get('learning_rate', 0):.8f}, "
              f"Time: {epoch_time:.2f}s")

        # Log to TensorBoard and MLflow
        log_metrics('train', train_metrics, epoch, writer, cfg.logging.use_mlflow)
        log_metrics('val', val_metrics, epoch, writer, cfg.logging.use_mlflow)

        # Save metrics to CSV
        save_metrics_to_csv(train_metrics, output_dir, epoch, 'train')
        save_metrics_to_csv(val_metrics, output_dir, epoch, 'val')

        # Save epoch summary
        save_epoch_summary(train_metrics, val_metrics, epoch, output_dir, epoch_time)

        # Keep track of validation losses for early stopping
        val_losses.append(val_metrics['loss'])

        # Visualize reconstructions every N epochs
        if epoch % cfg.visualization.viz_frequency == 0:
            print("Visualizing reconstructions...")
            try:
                recon_path = visualize_reconstruction(
                    model,
                    next(iter(val_loader)),  # Use a validation batch
                    epoch,
                    output_dir,
                    max_samples=cfg.visualization.num_samples,
                    include_aux=cfg.visualization.include_aux
                )
                log_reconstruction(recon_path, epoch, writer, cfg.logging.use_mlflow)
            except Exception as e:
                print(f"Error in reconstruction visualization: {e}")
                import traceback
                traceback.print_exc()

        # Save checkpoint
        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            print(f"New best validation loss: {best_val_loss:.6f}")

        # Only pass is_best=True when it's actually the best model
        save_checkpoint(model, optimizer, epoch, val_metrics['loss'], output_dir, is_best=is_best)

        # Check for early stopping
        if cfg.training.early_stopping.enabled:
            if check_early_stopping(
                val_losses,
                cfg.training.early_stopping.patience,
                cfg.training.early_stopping.min_delta
            ):
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    # Final logging
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Close MLflow run
    if cfg.logging.use_mlflow:
        mlflow.end_run()

if __name__ == "__main__":
    main()
