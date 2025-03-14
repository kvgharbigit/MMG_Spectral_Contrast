# MultiModalSpectralGPT Training Pipeline

This repository contains a comprehensive training and evaluation pipeline for the MultiModalSpectralGPT model, which integrates Masked Autoencoding (MAE) and Dual-Mode Contrastive Learning with Rim Masking for hyperspectral image analysis.

## Features

- **Flexible Training Pipeline**: Supports both automatic and predefined data splits
- **Dual-Mode Contrastive Learning**: Choose between global and spatial contrastive learning modes
- **Rim Masking**: Automatically excludes non-informative black border regions from loss calculations
- **Comprehensive Logging**: Integrates MLflow, TensorBoard, and CSV metrics tracking
- **Visualization Tools**: Includes HSI to RGB conversion and reconstruction visualization
- **Early Stopping**: Configurable early stopping to prevent overfitting
- **Hyperparameter Optimization**: Supports Hydra sweeps for hyperparameter tuning

## Installation

1. Create a conda environment (recommended):
   ```bash
   conda create -n spectralgpt python=3.9
   conda activate spectralgpt
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install hydra-core omegaconf mlflow tensorboard pandas matplotlib tqdm scikit-learn seaborn
   pip install h5py tifffile
   ```

## Project Structure

```
.
├── configs/                      # Hydra configuration files
│   ├── data/                     # Data configuration
│   ├── model/                    # Model configuration
│   ├── optimizer/                # Optimizer configuration
│   ├── scheduler/                # Learning rate scheduler configuration
│   ├── train.yaml                # Main training configuration
│   └── evaluate.yaml             # Evaluation configuration
├── MultiModalSpectralGPT.py      # Main model implementation
├── dataset.py                    # Dataset loading utilities
├── data_utils.py                 # Data preprocessing utilities
├── visualisation.py              # Visualization utilities
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
└── run.py                        # Helper script for running experiments
```

## Quick Start

### Training

To train the model with default settings:

```bash
python run.py train
```

With custom settings:

```bash
python run.py train --data_dir /path/to/data --gpus 0,1 --contrastive_mode global --batch_size 8 --epochs 100
```

Resume training from a checkpoint:

```bash
python run.py train --checkpoint /path/to/checkpoint.pth
```

### Evaluation

To evaluate a trained model:

```bash
python run.py evaluate --checkpoint /path/to/best_model.pth --data_dir /path/to/test_data
```

### Hyperparameter Sweep

To run a hyperparameter sweep:

```bash
python run.py sweep --data_dir /path/to/data --gpus 0,1
```

## Data Organization

The pipeline supports two ways of organizing your data:

1. **Auto-split mode**: A single directory with all data, which will be automatically split into train/validation sets
   ```
   /path/to/data/
   ├── patient1/
   │   ├── hsi.h5
   │   ├── IR.tiff
   │   ├── FAF.tiff
   │   └── thickness.tiff
   ├── patient2/
   ...
   ```

2. **Predefined split**: Separate directories for train, validation, and test sets
   ```
   /path/to/data/
   ├── train/
   │   ├── patient1/
   │   ├── patient2/
   │   ...
   ├── val/
   │   ├── patient3/
   │   ...
   └── test/
       ├── patient4/
       ...
   ```

## Configuration

The pipeline uses Hydra for configuration management. The main configuration files are:

- `configs/train.yaml`: Main training configuration
- `configs/evaluate.yaml`: Evaluation configuration
- `configs/model/spectral_gpt.yaml`: Model configuration
- `configs/data/default.yaml`: Data configuration
- `configs/optimizer/adamw.yaml`: Optimizer configuration
- `configs/scheduler/cosine.yaml`: Learning rate scheduler configuration

You can override configuration values via command line:

```bash
python train.py model.contrastive_mode=spatial data.use_auto_split=false training.batch_size=16
```

## Outputs

The training pipeline generates the following outputs:

- **Checkpoints**: Model weights at different epochs and the best model
- **Metrics**: CSV files with training and validation metrics
- **Summaries**: Text summaries of each epoch's performance
- **Visualizations**: Reconstructions at various epochs
- **TensorBoard Logs**: Detailed metrics and visualizations
- **MLflow Tracking**: Experiment tracking with artifacts and parameters

## Using TensorBoard

To visualize training progress:

```bash
tensorboard --logdir outputs/
```

## Using MLflow

To start the MLflow UI:

```bash
mlflow ui
```

## HSI to RGB Conversion

The pipeline includes utilities to convert hyperspectral images to RGB for visualization. The conversion assumes that the 30 wavelength bands are equally spaced across the 450-905 nm range.

## Citation

If you use this code in your research, please cite:

```
@article{yourname2025,
  title={MultiModalSpectralGPT: A Dual-Mode Contrastive Learning Approach with Rim Masking for Hyperspectral Image Analysis},
  author={Your Name and Coauthors},
  journal={Journal Name},
  year={2025}
}
```

## License

[Insert your license information here]