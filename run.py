#!/usr/bin/env python
import os
import sys
import argparse
import subprocess
from pathlib import Path
import yaml
import datetime


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run MultiModalSpectralGPT training and evaluation")
    parser.add_argument("mode", choices=["train", "evaluate", "sweep"],
                        help="Mode to run (train, evaluate, or sweep)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for resuming training or evaluation")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to data directory")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--contrastive_mode", type=str, choices=["global", "spatial"], default=None,
                        help="Contrastive learning mode (global or spatial)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training/evaluation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs for training")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with minimal data")
    return parser.parse_args()


def prepare_env(args):
    """Prepare environment variables"""
    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # Create timestamp for run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    return timestamp


def build_command(args, timestamp):
    """Build command for running hydra"""
    cmd = ["python"]
    
    if args.mode == "train":
        cmd.append("train.py")
    elif args.mode == "evaluate":
        cmd.append("evaluate.py")
    
    # Add custom config file if specified
    if args.config:
        cmd.append(f"--config-path={Path(args.config).parent}")
        cmd.append(f"--config-name={Path(args.config).stem}")
    
    # Add overrides based on command line arguments
    overrides = []
    
    if args.checkpoint:
        if args.mode == "train":
            overrides.append(f"training.resume_from_checkpoint=true")
            overrides.append(f"training.checkpoint_path={args.checkpoint}")
        elif args.mode == "evaluate":
            overrides.append(f"evaluation.checkpoint_path={args.checkpoint}")
    
    if args.data_dir:
        if args.mode == "train":
            overrides.append(f"data.parent_dir={args.data_dir}")
            overrides.append(f"data.train_dir={os.path.join(args.data_dir, 'train')}")
            overrides.append(f"data.val_dir={os.path.join(args.data_dir, 'val')}")
        elif args.mode == "evaluate":
            overrides.append(f"data.test_dir={os.path.join(args.data_dir, 'test')}")
    
    if args.contrastive_mode:
        overrides.append(f"model.contrastive_mode={args.contrastive_mode}")
    
    if args.batch_size:
        if args.mode == "train":
            overrides.append(f"training.batch_size={args.batch_size}")
        elif args.mode == "evaluate":
            overrides.append(f"evaluation.batch_size={args.batch_size}")
    
    if args.epochs:
        overrides.append(f"training.epochs={args.epochs}")
    
    if args.experiment_name:
        overrides.append(f"experiment_name={args.experiment_name}")
    
    if args.seed:
        overrides.append(f"seed={args.seed}")
    
    if args.output_dir:
        if args.mode == "train":
            overrides.append(f"hydra.run.dir={os.path.join(args.output_dir, timestamp)}")
        elif args.mode == "evaluate":
            overrides.append(f"hydra.run.dir={os.path.join(args.output_dir, 'eval', timestamp)}")
    
    if args.debug:
        # Add debug mode settings
        overrides.append("training.epochs=2")
        overrides.append("training.batch_size=2")
        overrides.append("logging.use_mlflow=false")
        overrides.append("visualization.viz_frequency=1")
    
    # Add overrides to command
    cmd.extend(overrides)
    
    return cmd


def run_sweep(args, timestamp):
    """Run a hyperparameter sweep using Hydra multirun"""
    # Example sweep command - modify as needed
    base_cmd = ["python", "train.py", "--multirun"]
    
    # Define sweep parameters
    sweep_params = [
        "model.contrastive_mode=global,spatial",
        "optimizer.lr=1e-4,5e-4",
        "model.mask_ratio=0.65,0.75,0.85",
    ]
    
    # Add other parameters from args
    cmd = base_cmd + sweep_params
    
    if args.data_dir:
        cmd.append(f"data.parent_dir={args.data_dir}")
        cmd.append(f"data.train_dir={os.path.join(args.data_dir, 'train')}")
        cmd.append(f"data.val_dir={os.path.join(args.data_dir, 'val')}")
    
    if args.experiment_name:
        cmd.append(f"experiment_name={args.experiment_name}_sweep")
    
    if args.output_dir:
        cmd.append(f"hydra.sweep.dir={os.path.join(args.output_dir, 'sweep', timestamp)}")
    
    if args.debug:
        # Add debug mode settings for sweep
        cmd.append("training.epochs=2")
        cmd.append("training.batch_size=2")
        cmd.append("logging.use_mlflow=false")
    
    print("Running sweep with command:")
    print(" ".join(cmd))
    
    # Execute the sweep
    subprocess.run(cmd)


def main():
    args = parse_args()
    timestamp = prepare_env(args)
    
    print(f"Running in {args.mode} mode")
    print(f"GPUs: {args.gpus}")
    
    if args.mode == "sweep":
        run_sweep(args, timestamp)
    else:
        cmd = build_command(args, timestamp)
        print("Running with command:")
        print(" ".join(cmd))
        
        # Execute the command
        subprocess.run(cmd)


if __name__ == "__main__":
    main()