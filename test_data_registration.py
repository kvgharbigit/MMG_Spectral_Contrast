import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import your modules
from dataset import PatientDataset, custom_collate_fn
from MultiModalSpectralGPT import SpatialRegistration

def test_dataset_with_spatial_registration(data_dir, output_dir='test_data_reg_outputs'):
    """
    Test integration between the PatientDataset and SpatialRegistration module.
    
    Args:
        data_dir (str): Directory containing patient data
        output_dir (str): Directory to save test results
    """
    print(f"Testing PatientDataset and SpatialRegistration integration...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    try:
        dataset = PatientDataset(data_dir)
        print(f"✓ Dataset initialized successfully with {len(dataset)} patients")
    except Exception as e:
        print(f"✗ Failed to initialize dataset: {e}")
        return
        
    # Skip test if no data found
    if len(dataset) == 0:
        print("✗ No patient data found in the directory")
        return
    
    # Create DataLoader with custom collate function
    batch_size = min(2, len(dataset))  # Use at most 2 samples to keep visualization simple
    try:
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            collate_fn=custom_collate_fn,
            shuffle=False
        )
        print(f"✓ DataLoader created with batch size {batch_size}")
    except Exception as e:
        print(f"✗ Failed to create DataLoader: {e}")
        return
    
    # Get a single batch
    try:
        batch = next(iter(dataloader))
        print(f"✓ Successfully loaded a batch of data")
        
        # Print information about the batch
        print(f"  HSI shape: {batch['hsi'].shape}")
        for key, value in batch['aux_data'].items():
            if value is not None:
                print(f"  {key} shape: {value.shape}")
            else:
                print(f"  {key}: None")
    except Exception as e:
        print(f"✗ Failed to get batch from DataLoader: {e}")
        return
    
    # Initialize SpatialRegistration module
    try:
        spatial_reg = SpatialRegistration(analysis_dim=500, target_bands=30)
        print(f"✓ SpatialRegistration module initialized")
    except Exception as e:
        print(f"✗ Failed to initialize SpatialRegistration: {e}")
        return
    
    # Process data through SpatialRegistration
    try:
        hsi_registered, aux_registered, thickness_mask = spatial_reg(batch['hsi'], batch['aux_data'])
        print(f"✓ Successfully processed batch through SpatialRegistration")
        
        # Print information about the processed data
        print(f"  Registered HSI shape: {hsi_registered.shape}")
        for key, value in aux_registered.items():
            if value is not None:
                print(f"  Registered {key} shape: {value.shape}")
            else:
                print(f"  Registered {key}: None")
        
        if thickness_mask is not None:
            print(f"  Thickness mask shape: {thickness_mask.shape}")
        else:
            print("  No thickness mask was created")
    except Exception as e:
        print(f"✗ Failed to process through SpatialRegistration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Visualize results
    try:
        visualize_results(batch, hsi_registered, aux_registered, thickness_mask, output_dir)
        print(f"✓ Visualizations saved to {output_dir}")
    except Exception as e:
        print(f"✗ Failed to visualize results: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed!")

def visualize_results(batch, hsi_registered, aux_registered, thickness_mask, output_dir):
    """Visualize original and processed data for comparison."""
    
    patient_ids = batch['patient_id']
    
    for i in range(min(len(patient_ids), 2)):  # Visualize up to 2 patients
        patient_id = patient_ids[i]
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Patient: {patient_id} - Original vs. Registered", fontsize=16)
        
        # Row labels
        axes[0, 0].set_ylabel("Original", fontsize=14)
        axes[1, 0].set_ylabel("Registered", fontsize=14)
        
        # Column labels
        col_labels = ["HSI (mid band)", "IR", "AF", "Thickness/Mask"]
        for j, label in enumerate(col_labels):
            axes[0, j].set_title(label, fontsize=14)
        
        # Plot original HSI data (middle band)
        hsi_orig = batch['hsi'][i]
        mid_band = hsi_orig.shape[1] // 2
        axes[0, 0].imshow(hsi_orig[0, mid_band].numpy(), cmap='viridis')
        
        # Plot original auxiliary data
        aux_keys = ['ir', 'af', 'thickness']
        for j, key in enumerate(aux_keys):
            if batch['aux_data'][key] is not None:
                axes[0, j+1].imshow(batch['aux_data'][key][i, 0].numpy(), cmap='gray')
                axes[0, j+1].set_title(f"{col_labels[j+1]} (Original)", fontsize=12)
            else:
                axes[0, j+1].text(0.5, 0.5, f"No {key} data", 
                                 ha='center', va='center', transform=axes[0, j+1].transAxes)
        
        # Plot registered HSI data (middle band)
        mid_band_reg = hsi_registered.shape[2] // 2
        axes[1, 0].imshow(hsi_registered[i, 0, mid_band_reg].numpy(), cmap='viridis')
        
        # Plot registered auxiliary data
        for j, key in enumerate(aux_keys[:2]):  # IR and AF
            if aux_registered[key] is not None:
                axes[1, j+1].imshow(aux_registered[key][i, 0].numpy(), cmap='gray')
                axes[1, j+1].set_title(f"{col_labels[j+1]} (Registered)", fontsize=12)
            else:
                axes[1, j+1].text(0.5, 0.5, f"No {key} data", 
                                 ha='center', va='center', transform=axes[1, j+1].transAxes)
        
        # Plot thickness data and mask
        if aux_registered['thickness'] is not None:
            axes[1, 3].imshow(aux_registered['thickness'][i, 0].numpy(), cmap='gray')
            axes[1, 3].set_title("Thickness (Registered)", fontsize=12)
        elif thickness_mask is not None:
            axes[1, 3].imshow(thickness_mask[i, 0].numpy(), cmap='gray')
            axes[1, 3].set_title("Generated Mask", fontsize=12)
        else:
            axes[1, 3].text(0.5, 0.5, "No thickness/mask data", 
                           ha='center', va='center', transform=axes[1, 3].transAxes)
        
        # Remove axis ticks for cleaner visualization
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Save figure
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        output_path = os.path.join(output_dir, f"{patient_id}_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "dummydata"
    test_dataset_with_spatial_registration(data_dir)
