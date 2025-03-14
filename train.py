import os
import sys
import time
import glob
import torch
import numpy as np
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
from visualisation import visualize_batch
import matplotlib.cm as cm

# Configure matplotlib for non-GUI environments
plt.switch_backend('agg')






def visualize_reconstruction(model, test_batch, epoch, output_dir, max_samples=2, include_aux=True):
    """
    Placeholder visualization function that does not generate any images.

    Args:
        model: The trained model
        test_batch: A batch of test data
        epoch: Current epoch number
        output_dir: Directory to save visualizations
        max_samples: Maximum number of samples to visualize
        include_aux: Whether to include auxiliary modalities in visualization
    """
    print(f"Reconstruction visualization disabled at epoch {epoch}")
    return None

def calculate_metrics(outputs):
    """
    Calculate aggregate metrics from a list of model outputs.
    
    Args:
        outputs: List of dictionaries containing model outputs
        
    Returns:
        Dictionary of aggregated metrics
    """
    metrics = {
        'loss': 0.0,
        'loss_recon': 0.0,
        'loss_contrast': 0.0,
        'num_modalities': 0.0,
    }
    
    batch_count = len(outputs)
    if batch_count == 0:
        return metrics
    
    # Sum the metrics across all batches
    for output in outputs:
        metrics['loss'] += output['loss'].item()
        metrics['loss_recon'] += output['loss_recon'].item()
        metrics['loss_contrast'] += output['loss_contrast'].item()
        metrics['num_modalities'] += output['num_modalities'].item()
    
    # Calculate the average for each metric
    for key in metrics:
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
    Placeholder logging function that does nothing.

    Args:
        recon_path: Path to reconstruction image
        epoch: Current epoch number
        writer: TensorBoard SummaryWriter instance
        mlflow_logging: Whether to log to MLflow
    """
    print(f"Skipping reconstruction logging at epoch {epoch}")
    return


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


def save_epoch_summary(train_metrics, val_metrics, epoch, output_dir, total_time):
    """
    Save a summary of the epoch's metrics to a text file.
    
    Args:
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        epoch: Current epoch number
        output_dir: Directory to save the summary
        total_time: Total time taken for the epoch
    """
    # Create summaries directory if it doesn't exist
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Define file path
    summary_path = os.path.join(summaries_dir, f"epoch_{epoch}_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write(f"Epoch {epoch} Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Training Metrics:\n")
        for key, value in train_metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
        
        f.write("\nValidation Metrics:\n")
        for key, value in val_metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
        
        f.write(f"\nTime taken: {total_time:.2f} seconds\n")
        
        # Calculate improvements from previous epoch if available
        improvement_path = os.path.join(summaries_dir, "best_metrics.txt")
        if os.path.exists(improvement_path):
            with open(improvement_path, 'r') as best_file:
                lines = best_file.readlines()
                if len(lines) >= 4:  # Ensure we have enough lines to read
                    best_loss = float(lines[3].split(": ")[1].strip())
                    improvement = best_loss - val_metrics['loss']
                    f.write(f"\nImprovement in validation loss: {improvement:.6f}")
        
    # Update best metrics if this is the best epoch so far
    update_best_metrics(val_metrics, epoch, summaries_dir)


def update_best_metrics(val_metrics, epoch, summaries_dir):
    """
    Update the best metrics file if the current epoch has better validation loss.
    
    Args:
        val_metrics: Dictionary of validation metrics
        epoch: Current epoch number
        summaries_dir: Directory containing the summaries
    """
    best_path = os.path.join(summaries_dir, "best_metrics.txt")
    current_val_loss = val_metrics['loss']
    
    # Check if best metrics file exists
    if os.path.exists(best_path):
        with open(best_path, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 4:  # Ensure we have enough lines to read
                best_epoch = int(lines[1].split(": ")[1].strip())
                best_loss = float(lines[3].split(": ")[1].strip())
                
                # Only update if current loss is better (lower)
                if current_val_loss >= best_loss:
                    return
    
    # Write new best metrics
    with open(best_path, 'w') as f:
        f.write("Best Metrics\n")
        f.write(f"Epoch: {epoch}\n")
        f.write("Validation Metrics:\n")
        for key, value in val_metrics.items():
            f.write(f"  {key}: {value:.6f}\n")


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
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on
        contrastive_mode: Contrastive mode to use (if None, use model's default)
        
    Returns:
        List of outputs from each batch
    """
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
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'recon_loss': output['loss_recon'].item(),
            'contrast_loss': output['loss_contrast'].item()
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
            
            # Update progress bar
            pbar.set_postfix({
                'loss': output['loss'].item(),
                'recon_loss': output['loss_recon'].item(),
                'contrast_loss': output['loss_contrast'].item()
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
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(checkpoints_dir, "best_model.pth")
        torch.save(checkpoint, best_path)


@hydra.main(config_path="configs", config_name="train")
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
    if cfg.scheduler.use_scheduler:
        if cfg.scheduler.type == "cosine":
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

    for epoch in range(start_epoch, cfg.training.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.training.epochs}")
        epoch_start_time = time.time()

        # Training phase
        train_outputs = train_epoch(
            model, train_loader, optimizer, device,
            contrastive_mode=cfg.model.contrastive_mode
        )
        train_metrics = calculate_metrics(train_outputs)

        # Validation phase
        val_outputs = validate_epoch(
            model, val_loader, device,
            contrastive_mode=cfg.model.contrastive_mode
        )
        val_metrics = calculate_metrics(val_outputs)

        # Update learning rate scheduler
        if scheduler is not None:
            if cfg.scheduler.type == "reduce_on_plateau":
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Log metrics
        print(f"Train Loss: {train_metrics['loss']:.6f}, "
              f"Val Loss: {val_metrics['loss']:.6f}, "
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
            recon_path = visualize_reconstruction(
                model,
                next(iter(val_loader)),
                epoch,
                output_dir,
                max_samples=cfg.visualization.num_samples,
                include_aux=cfg.visualization.include_aux  # Access the parameter from config
            )
            log_reconstruction(recon_path, epoch, writer, cfg.logging.use_mlflow)

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            print(f"New best validation loss: {best_val_loss:.6f}")

        save_checkpoint(model, optimizer, epoch, val_metrics['loss'], output_dir, is_best)

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
