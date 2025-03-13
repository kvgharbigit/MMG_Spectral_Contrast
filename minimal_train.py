# minimal_train.py
import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import traceback

# Set up logging to a file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('training_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(cfg: DictConfig):
    try:
        logger.info("Starting minimal training script")

        # Get original working directory
        original_cwd = hydra.utils.get_original_cwd()
        logger.info(f"Original working directory: {original_cwd}")
        logger.info(f"Current working directory: {os.getcwd()}")

        # Log configuration
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        # Fix data paths
        if not os.path.isabs(cfg.data.parent_dir):
            cfg.data.parent_dir = os.path.join(original_cwd, cfg.data.parent_dir)
        logger.info(f"Using data directory: {cfg.data.parent_dir}")

        # Import modules
        logger.info("Importing modules...")
        import torch
        from dataset import PatientDataset, custom_collate_fn
        from MultiModalSpectralGPT import MultiModalSpectralGPT
        logger.info("Modules imported successfully")

        # Load dataset
        logger.info("Loading dataset...")
        dataset = PatientDataset(
            parent_dir=cfg.data.parent_dir,
            analysis_dim=cfg.model.analysis_dim,
            target_bands=cfg.model.num_frames
        )
        logger.info(f"Dataset loaded with {len(dataset)} samples")

        # Create model
        logger.info("Creating model...")
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
        )
        logger.info("Model created successfully")

        # Create data loader
        logger.info("Creating data loader...")
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )
        logger.info(f"DataLoader created with batch size {cfg.training.batch_size}")

        # Create optimizer
        logger.info("Creating optimizer...")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        )
        logger.info("Optimizer created successfully")

        # Log that we're ready to start training
        logger.info("Setup complete - ready to start training loop")
        logger.info("Script completed successfully")

    except Exception as e:
        logger.error(f"Error in training script: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        logging.error(traceback.format_exc())