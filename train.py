import os

#tw
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.putenv('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
from torch.cuda.amp import autocast, GradScaler

import time
import torch.nn.functional as F
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
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
import gc


# Import custom modules
from MultiModalSpectralGPT import MultiModalSpectralGPT
from dataset import PatientDataset, custom_collate_fn, create_patient_dataloader
from hsi_to_rgb import hsi_to_rgb, simple_hsi_to_rgb  # These are already imported elsewhere
from dataset import MultiModalTransforms
from visualisation import (
    visualize_pixel_reconstruction  # Add this line
)

import matplotlib.cm as cm

# Configure matplotlib for non-GUI environments
plt.switch_backend('agg')


import torch
import os

# Optional: only necessary if you want to lock to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global variables to track memory reservation status and store tensors
_MEMORY_RESERVED = False
_RESERVED_TENSORS = []


def reserve_gpu_memory(device_id=0, target_gb=10, chunk_size_mb=512):
    """
    Allocates memory to effectively reserve the GPU, but with more conservative allocation.

    Args:
        device_id (int): CUDA device ID to use
        target_gb (float): Target amount of memory to reserve in GB
        chunk_size_mb (int): Size of individual memory chunks to allocate in MB

    Returns:
        float: Amount of memory actually reserved in GB, or 0 if reservation failed
    """
    global _MEMORY_RESERVED, _RESERVED_TENSORS

    # If memory is already reserved, skip allocation
    if _MEMORY_RESERVED:
        print(f"GPU memory already reserved ({len(_RESERVED_TENSORS)} tensors). Skipping.")
        # Calculate and return current allocation
        if _RESERVED_TENSORS:
            allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            return allocated
        return 0

    # Clear any previous reservations (shouldn't happen, but as a safeguard)
    _RESERVED_TENSORS.clear()

    # Get available memory on GPU
    torch.cuda.empty_cache()  # Clear any unused memory first
    total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
    allocated_memory = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    available_memory = total_memory - allocated_memory

    # Calculate safe target (use only 50% of available memory)
    safe_target = min(target_gb, available_memory * 0.5)  # Half of available to be safer
    print(f"[GPU {device_id}] Total memory: {total_memory:.2f} GB")
    print(f"[GPU {device_id}] Already allocated: {allocated_memory:.2f} GB")
    print(f"[GPU {device_id}] Target allocation: {safe_target:.2f} GB")

    device = torch.device(f"cuda:{device_id}")

    try:
        # Try allocating chunks to fill up the GPU
        allocated_gb = 0
        max_chunks = int(safe_target / (chunk_size_mb / 1024))
        print(f"Allocating up to {max_chunks} chunks of {chunk_size_mb} MB each")

        for i in range(max_chunks):
            # Each float32 element is 4 bytes
            tensor_size = chunk_size_mb * 1024 * 1024 // 4
            tensor = torch.empty((tensor_size,), dtype=torch.float32, device=device)
            _RESERVED_TENSORS.append(tensor)

            # Check how much we've allocated so far
            allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            allocated_gb = allocated - allocated_memory  # How much we've added

            if i % 5 == 0:  # Print status every 5 chunks
                print(f"[GPU {device_id}] Reserved ~{allocated_gb:.2f} GB with {len(_RESERVED_TENSORS)} tensors")

            # Stop when we reach the target
            if allocated_gb >= safe_target:
                break

    except RuntimeError as e:
        print(f"[!] Stopped allocating: {e}")
        # If we hit an error but still allocated some tensors, consider it partially successful
        if _RESERVED_TENSORS:
            allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            allocated_gb = allocated - allocated_memory
            print(f"[GPU {device_id}] Partially reserved ~{allocated_gb:.2f} GB")
            _MEMORY_RESERVED = True
            return allocated_gb
        return 0

    # Calculate final allocation
    final_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    allocated_gb = final_allocated - allocated_memory
    print(f"[GPU {device_id}] Successfully reserved {allocated_gb:.2f} GB with {len(_RESERVED_TENSORS)} tensors")

    # Set the flag to indicate memory has been reserved
    _MEMORY_RESERVED = True

    return allocated_gb


def release_reserved_memory():
    """
    Release any memory that was reserved by reserve_gpu_memory.
    """
    global _MEMORY_RESERVED, _RESERVED_TENSORS

    if not _MEMORY_RESERVED:
        print("No GPU memory was reserved. Nothing to release.")
        return

    tensor_count = len(_RESERVED_TENSORS)
    _RESERVED_TENSORS.clear()

    # Force garbage collection to ensure tensors are freed
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    _MEMORY_RESERVED = False
    print(f"Released memory from {tensor_count} reserved tensors.")

    # Return current allocation for verification
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    print(f"Current GPU memory allocation: {allocated:.2f} GB")
    return allocated



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



    # Calculate the average for normal metrics
    for key in ['loss', 'loss_recon', 'loss_contrast', 'num_modalities']:
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

    scaler = torch.amp.GradScaler()  # Create gradient scaler for mixed precision

    # Set contrastive mode if specified
    if contrastive_mode is not None:
        model.contrastive_mode = contrastive_mode

    # Create progress bar
    pbar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        # Clear CUDA cache between iterations if not the first batch
        if torch.cuda.is_available() and batch_idx > 0:
            torch.cuda.empty_cache()

        # Move data to device
        hsi = batch['hsi'].to(device)
        aux_data = {k: v.to(device) if v is not None else None for k, v in batch['aux_data'].items()}
        batch_idx = batch['batch_idx'].to(device)

        # Forward pass with mixed precision
        optimizer.zero_grad()
        with autocast():
            output = model(hsi, aux_data, batch_idx)
            loss = output['loss']

        # Backward pass and optimization with scaling for mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Store only necessary data from output with detach
        output_dict = {
            'loss': output['loss'].detach(),
            'loss_recon': output['loss_recon'].detach(),
            'loss_contrast': output['loss_contrast'].detach(),
            'num_modalities': output['num_modalities'].detach()
        }
        outputs.append(output_dict)



        # Get current learning rate for progress bar
        current_lr = optimizer.param_groups[0]['lr']

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.10f}",
            'recon_loss': f"{output['loss_recon'].item():.10f}",
            'contrast_loss': f"{output['loss_contrast'].item():.10f}",
            'lr': f"{current_lr}\n"
        })

        # Delete intermediate tensors
        del hsi, aux_data, batch_idx, output

        # Force garbage collection every iteration
        gc.collect()

    return outputs


def validate_epoch(model, dataloader, device, contrastive_mode=None):
    """Validate the model on the validation set."""
    model.eval()
    outputs = []


    # Set contrastive mode if specified
    if contrastive_mode is not None:
        model.contrastive_mode = contrastive_mode

    # Create progress bar
    pbar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Clear CUDA cache between iterations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Move data to device
            hsi = batch['hsi'].to(device)
            aux_data = {k: v.to(device) if v is not None else None for k, v in batch['aux_data'].items()}
            batch_idx = batch['batch_idx'].to(device)

            # Forward pass
            output = model(hsi, aux_data, batch_idx)

            # Store only necessary data with detach
            output_dict = {
                'loss': output['loss'].detach(),
                'loss_recon': output['loss_recon'].detach(),
                'loss_contrast': output['loss_contrast'].detach(),
                'num_modalities': output['num_modalities'].detach()
            }
            outputs.append(output_dict)



            # Update progress bar
            pbar.set_postfix({
                'loss': f"{output['loss'].item():.10f}",
                'recon_loss': f"{output['loss_recon'].item():.10f}",
                'contrast_loss': f"{output['loss_contrast'].item():.10f}"
            })

            # Delete intermediate tensors
            del hsi, aux_data, batch_idx, output

            # Force garbage collection
            gc.collect()

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


# Add this function to train.py to visualize reconstructions during training
# Add this function to train.py to visualize reconstructions during training
# Update the function in train.py to use the simplified visualization

def visualize_reconstruction_during_training(model, val_loader, device, epoch, output_dir):
    """
    Generate and save reconstruction visualizations during training with diversity analysis
    using the original model patches.
    """
    # Create visualization directory if it doesn't exist
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    model.eval()

    try:
        # Get a single batch from the dataloader
        batch = next(iter(val_loader))

        # Move data to device
        hsi = batch['hsi'].to(device)
        aux_data = {k: v.to(device) if v is not None else None for k, v in batch['aux_data'].items()}
        batch_idx = batch['batch_idx'].to(device)

        # Get thickness mask if available
        thickness_mask = batch.get('thickness_mask', None)
        if thickness_mask is not None:
            thickness_mask = thickness_mask.to(device)

        # Forward pass without gradient computation
        with torch.no_grad():
            output = model(hsi, aux_data, batch_idx)

        # Get the reconstructed pixels and mask
        reconstructed_pixels = output['reconstructed_pixels']
        mask = output['mask']

        # Define save path
        save_path = os.path.join(viz_dir, f'reconstruction_epoch_{epoch}.png')

        # Visualize the reconstruction with numerical visualization and patch diversity analysis
        from visualisation import visualize_pixel_reconstruction
        visualize_pixel_reconstruction(
            model=model,
            original_input=hsi,
            reconstructed_pixels=reconstructed_pixels,
            mask=mask,
            output=output,  # Pass the complete output dictionary containing pred
            thickness_mask=thickness_mask,
            save_path=save_path,
            add_numerical_viz=True
        )

        # Log the visualization to TensorBoard and MLFlow
        return save_path

    except Exception as e:
        print(f"Error generating reconstruction visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

@hydra.main(config_path="configs", config_name="train", version_base="1.1")
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
    if torch.cuda.is_available():
        reserve_gpu_memory(device_id=0)

    # Create output directory
    output_dir = os.getcwd()  # Hydra changes working dir to outputs/{date}/...
    print(f"Output directory: {output_dir}")

    # Save hyperparameter and training configuration summary
    summary_path = save_training_summary(cfg, output_dir)
    print(f"Training configuration summary saved to: {summary_path}")

    # Set up MLflow
    if cfg.logging.use_mlflow:
        mlflow.set_experiment(cfg.logging.experiment_name)
        mlflow.start_run(run_name=cfg.logging.run_name)

        # Log Hydra config
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # Log configuration summary to MLflow
        if summary_path and os.path.exists(summary_path):
            mlflow.log_artifact(summary_path)

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
            pin_memory=False,
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
            pin_memory=False,
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
    total_start_time = time.time()  # Add this line to record when training started
    early_stopped = False  # Add this line to track if early stopping occurred

    # Training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        epoch_start_time = time.time()

        # Clear memory before each epoch
        torch.cuda.empty_cache()
        gc.collect()

        # Training phase
        train_outputs = train_epoch(
            model, train_loader, optimizer, device,
            contrastive_mode=cfg.model.contrastive_mode
        )
        train_metrics = calculate_metrics(train_outputs, optimizer)

        # Validation phase
        val_outputs = validate_epoch(
            model, val_loader, device,
            contrastive_mode=cfg.model.contrastive_mode
        )
        val_metrics = calculate_metrics(val_outputs)

        #Visualise
        if (epoch + 1) % cfg.visualization.viz_frequency == 0 or epoch == cfg.training.epochs - 1:
            print("Generating reconstruction visualization with diversity analysis...")
            recon_path = visualize_reconstruction_during_training(
                model, val_loader, device, epoch, output_dir
            )

            # Log reconstruction to TensorBoard and MLflow
            if recon_path:
                log_reconstruction(recon_path, epoch, writer, cfg.logging.use_mlflow)

        # Update learning rate scheduler - only for epoch-based schedulers
        if scheduler is not None and scheduler_step_frequency == "epoch":
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print metrics summary
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
                print(f"Early stopping triggered after epoch {epoch + 1}")
                early_stopped = True  # Add this line to record that early stopping happened
                break

    # Calculate total training time
    total_training_time = time.time() - total_start_time

    # Final logging
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Update the summary file with training results
    update_summary_with_results(
        output_dir,
        best_val_loss,
        total_training_time,
        epoch + 1,  # Number of epochs completed
        early_stopped
    )

    # Log the configuration summary to MLflow if enabled
    if cfg.logging.use_mlflow:
        summary_path = os.path.join(output_dir, 'summaries', "training_configuration.txt")
        if os.path.exists(summary_path):
            mlflow.log_artifact(summary_path, "training_summary")
        mlflow.end_run()

    # Close TensorBoard writer
    writer.close()

def save_training_summary(cfg, output_dir):
    """
    Save a comprehensive summary of hyperparameters and training configuration to a text file.

    Args:
        cfg: The Hydra configuration object containing all parameters
        output_dir: Directory to save the summary file
    """
    import os
    from datetime import datetime
    import torch

    # Create summaries directory if it doesn't exist
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)

    # Define file path
    summary_path = os.path.join(summaries_dir, "training_configuration.txt")

    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get system info
    device_info = f"CPU" if not torch.cuda.is_available() else f"GPU: {torch.cuda.get_device_name(0)}"
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"

    try:
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"MULTIMODAL SPECTRALGPT TRAINING CONFIGURATION\n")
            f.write(f"Generated on: {current_time}\n")
            f.write("=" * 80 + "\n\n")

            # System Information
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Device: {device_info}\n")
            f.write(f"PyTorch Version: {torch.__version__}\n")
            f.write(f"CUDA Version: {cuda_version}\n")
            f.write(f"Random Seed: {cfg.seed}\n\n")

            # Model Configuration
            f.write("MODEL CONFIGURATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Analysis Dimension: {cfg.model.analysis_dim}\n")
            f.write(f"Patch Size: {cfg.model.patch_size}\n")
            f.write(f"Embedding Dimension: {cfg.model.embed_dim}\n")
            f.write(f"Decoder Embedding Dimension: {cfg.model.decoder_embed_dim}\n")
            f.write(f"Transformer Depth: {cfg.model.depth}\n")
            f.write(f"Decoder Depth: {cfg.model.decoder_depth}\n")
            f.write(f"Number of Attention Heads: {cfg.model.num_heads}\n")
            f.write(f"Decoder Attention Heads: {cfg.model.decoder_num_heads}\n")
            f.write(f"MLP Ratio: {cfg.model.mlp_ratio}\n")
            f.write(f"Spectral Bands: {cfg.model.num_frames}\n")
            f.write(f"Temporal Patch Size: {cfg.model.t_patch_size}\n")
            f.write(f"Input Channels: {cfg.model.in_chans}\n")
            f.write(f"Auxiliary Channels: {cfg.model.aux_chans}\n")
            f.write(f"Auxiliary Embed Dimension: {cfg.model.aux_embed_dim}\n")
            f.write(f"Temperature: {cfg.model.temperature}\n")
            f.write(f"Mask Ratio: {cfg.model.mask_ratio}\n")
            f.write(f"Contrastive Mode: {cfg.model.contrastive_mode}\n")
            f.write(f"Using Thickness Mask: {cfg.model.use_thickness_mask}\n\n")

            # Optimizer Configuration
            f.write("OPTIMIZER CONFIGURATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Optimizer: AdamW\n")  # Hardcoded as your code uses AdamW
            f.write(f"Learning Rate: {cfg.optimizer.lr}\n")
            f.write(f"Weight Decay: {cfg.optimizer.weight_decay}\n")
            f.write(f"Beta1: {cfg.optimizer.beta1}\n")
            f.write(f"Beta2: {cfg.optimizer.beta2}\n\n")

            # Scheduler Configuration
            if cfg.scheduler.use_scheduler:
                f.write("LEARNING RATE SCHEDULER\n")
                f.write("-" * 50 + "\n")
                f.write(f"Scheduler Type: {cfg.scheduler.type}\n")

                # Write scheduler-specific parameters
                if cfg.scheduler.type == "cosine":
                    f.write(f"Minimum Learning Rate: {cfg.scheduler.min_lr}\n")
                    if hasattr(cfg.scheduler, 'warmup_ratio'):
                        f.write(f"Warmup Ratio: {cfg.scheduler.warmup_ratio}\n")
                elif cfg.scheduler.type == "step":
                    f.write(f"Step Size: {cfg.scheduler.step_size}\n")
                    f.write(f"Gamma: {cfg.scheduler.gamma}\n")
                elif cfg.scheduler.type == "reduce_on_plateau":
                    f.write(f"Factor: {cfg.scheduler.factor}\n")
                    f.write(f"Patience: {cfg.scheduler.patience}\n")
                    f.write(f"Minimum Learning Rate: {cfg.scheduler.min_lr}\n")
                f.write("\n")

            # Training Configuration
            f.write("TRAINING CONFIGURATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Batch Size: {cfg.training.batch_size}\n")
            f.write(f"Number of Epochs: {cfg.training.epochs}\n")
            f.write(f"Resume from Checkpoint: {cfg.training.resume_from_checkpoint}\n")
            if hasattr(cfg.training, 'checkpoint_path') and cfg.training.resume_from_checkpoint:
                f.write(f"Checkpoint Path: {cfg.training.checkpoint_path}\n")

            # Early Stopping
            if cfg.training.early_stopping.enabled:
                f.write(f"Early Stopping Enabled: Yes\n")
                f.write(f"Patience: {cfg.training.early_stopping.patience}\n")
                f.write(f"Minimum Delta: {cfg.training.early_stopping.min_delta}\n")
            else:
                f.write(f"Early Stopping Enabled: No\n")
            f.write("\n")

            # Data Configuration
            f.write("DATA CONFIGURATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Parent Directory: {cfg.data.parent_dir}\n")
            f.write(f"Training Directory: {cfg.data.train_dir}\n")
            f.write(f"Validation Directory: {cfg.data.val_dir}\n")
            f.write(f"Using Auto Split: {cfg.data.use_auto_split}\n")
            if cfg.data.use_auto_split:
                f.write(f"Auto Split Ratio: {cfg.data.auto_split_ratio}\n")
            f.write(f"Number of Workers: {cfg.data.num_workers}\n")
            f.write(f"Drop Last Batch: {cfg.data.drop_last}\n")
            f.write(f"Using Data Augmentation: {cfg.data.use_augmentation}\n")
            if cfg.data.use_augmentation:
                f.write(f"Augmentation Probability: {cfg.data.augmentation.prob}\n")
                f.write(f"Rotation Degrees: {cfg.data.augmentation.rotation_degrees}\n")
                f.write(f"Scale Range: {cfg.data.augmentation.scale_range}\n")
            f.write("\n")

            # Logging Configuration
            f.write("LOGGING CONFIGURATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Using MLflow: {cfg.logging.use_mlflow}\n")
            if cfg.logging.use_mlflow:
                f.write(f"Experiment Name: {cfg.logging.experiment_name}\n")
                f.write(f"Run Name: {cfg.logging.run_name}\n")
            f.write(f"Visualization Frequency: {cfg.visualization.viz_frequency}\n")
            f.write("\n")

            # Additional Information
            f.write("ADDITIONAL INFORMATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write(f"Command Line/Config Overrides: {get_overrides_str(cfg)}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF CONFIGURATION SUMMARY\n")
            f.write("=" * 80 + "\n")

        print(f"Training configuration summary saved to: {summary_path}")
        return summary_path

    except Exception as e:
        print(f"Error saving training configuration summary: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_overrides_str(cfg):
    """
    Attempts to extract override information from Hydra config if available.

    Args:
        cfg: The Hydra configuration object

    Returns:
        str: String representation of overrides or "Not available"
    """
    try:
        # Try to access Hydra's override history if available
        if hasattr(cfg, '_metadata') and hasattr(cfg._metadata, 'overrides'):
            return ', '.join(cfg._metadata.overrides.items())
        else:
            return "Not available in config object"
    except:
        return "Could not extract overrides"


def update_summary_with_results(output_dir, best_val_loss, training_time, epochs_completed, early_stopped=False):
    """
    Update the training configuration summary with the results of training.

    Args:
        output_dir: Directory where the summary file is located
        best_val_loss: Best validation loss achieved during training
        training_time: Total training time in seconds
        epochs_completed: Number of epochs completed
        early_stopped: Whether training was stopped early
    """
    import os
    from datetime import timedelta

    # Path to the summary file
    summary_path = os.path.join(output_dir, 'summaries', "training_configuration.txt")

    if not os.path.exists(summary_path):
        print(f"Warning: Could not find configuration summary at {summary_path}")
        return

    try:
        # Read the existing file
        with open(summary_path, 'r') as f:
            content = f.read()

        # Format training time as hours:minutes:seconds
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        # Create results section
        results_section = (
                "\nTRAINING RESULTS\n" +
                "-" * 50 + "\n" +
                f"Best Validation Loss: {best_val_loss:.10f}\n" +
                f"Epochs Completed: {epochs_completed}\n" +
                f"Early Stopping Triggered: {'Yes' if early_stopped else 'No'}\n" +
                f"Total Training Time: {formatted_time} (HH:MM:SS)\n" +
                "\n"
        )

        # Find the end marker in the file
        end_marker = "=" * 80 + "\nEND OF CONFIGURATION SUMMARY\n" + "=" * 80

        if end_marker in content:
            # Insert results before the end marker
            updated_content = content.replace(end_marker, results_section + end_marker)
        else:
            # If end marker not found, just append to the end
            updated_content = content + "\n" + results_section

        # Write the updated content back to the file
        with open(summary_path, 'w') as f:
            f.write(updated_content)

        print(f"Training results added to configuration summary: {summary_path}")

        # Also create a separate results file for quick reference
        results_path = os.path.join(output_dir, 'summaries', "training_results.txt")
        with open(results_path, 'w') as f:
            f.write("TRAINING RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Validation Loss: {best_val_loss:.10f}\n")
            f.write(f"Epochs Completed: {epochs_completed}\n")
            f.write(f"Early Stopping Triggered: {'Yes' if early_stopped else 'No'}\n")
            f.write(f"Total Training Time: {formatted_time} (HH:MM:SS)\n")

        return True

    except Exception as e:
        print(f"Error updating training summary with results: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
