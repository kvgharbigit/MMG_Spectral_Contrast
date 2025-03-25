import os
import torch
import matplotlib.pyplot as plt
from dataset import create_patient_dataloader
from MultiModalSpectralGPT import MultiModalSpectralGPT
from verification import verify_reconstruction_pipeline, evaluate_reconstruction_pipeline


def main():
    # Configure these settings directly
    checkpoint_path = "path/to/your/checkpoint.pth"  # CHANGE THIS
    data_dir = "path/to/validation/data"  # CHANGE THIS
    output_dir = "reconstruction_verification"
    full_evaluation = True  # Set to False if you only want a single batch test

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load your trained model
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same configuration as training
    model = MultiModalSpectralGPT(
        analysis_dim=500,
        patch_size=(25, 25),
        embed_dim=768,
        depth=16,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        num_frames=30,
        t_patch_size=5,
        in_chans=1,
        aux_chans=1,
        aux_embed_dim=256,
        temperature=0.07,
        mask_ratio=0.75,
        contrastive_mode='global'
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Create data loader for validation data
    print(f"Loading validation data from {data_dir}")
    val_loader = create_patient_dataloader(
        parent_dir=data_dir,
        analysis_dim=500,
        target_bands=30,
        batch_size=1
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to {output_dir}")

    # Run single-batch verification
    print("Running single-batch verification...")
    batch = next(iter(val_loader))
    fig, metrics = verify_reconstruction_pipeline(
        model,
        batch['hsi'],
        device,
        save_path=os.path.join(output_dir, 'single_batch_verification.png')
    )
    plt.close(fig)

    print(f"Single batch reconstruction metrics:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")

    # Run comprehensive evaluation if requested
    if full_evaluation:
        print("\nRunning comprehensive evaluation...")
        avg_metrics = evaluate_reconstruction_pipeline(
            model,
            val_loader,
            device,
            output_dir=output_dir
        )


if __name__ == '__main__':
    main()