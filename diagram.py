"""
Spatially Registered Multi-Modal Guided SpectralGPT Architecture
===============================================================

Overarching Model Architecture Diagram:
---------------------------------------
                  Input Modalities
    HSI             IR             AF         Thickness
[B,1,91,500,500] [B,1,128,128] [B,1,256,256] [B,1,200,200]
    |               |               |             |
    v               v               v             v
+-----------------------------------------------------------+
|                Spatial Registration                        |
|  - Resize to common dimension (500x500)                   |
|  - Select spectral bands for HSI                          |
+-----------------------------------------------------------+
    |               |               |             |
    v               v               v             v
[B,1,30,500,500] [B,1,500,500] [B,1,500,500] [B,1,500,500]
    |               |               |             |
    v               v               v             v
+-----------------------------------------------------------+
|                Patch Embedding                             |
|  - 3D Patch Embedding (HSI)                               |
|  - 2D Patch Embedding (Auxiliary)                         |
|  - Global Averaging for Auxiliary Modalities               |
+-----------------------------------------------------------+
    |               |               |             |
    v               v               v             v
[B,2400,768]   [B,256]         [B,256]       [B,256]
    |               |               |             |
    v               v               v             v
+-----------------------------------------------------------+
|            Cross-Attention Conditioning                    |
|  - Project auxiliary features                             |
|  - Condition HSI tokens with auxiliary global vectors     |
+-----------------------------------------------------------+
    |
    v
+-----------------------------------------------------------+
|                Transformer Blocks                          |
|  - Process conditioned HSI tokens                         |
|  - Apply self-attention and feed-forward layers           |
+-----------------------------------------------------------+
    |
    v
+-----------------------------------------------------------+
|                 Decoding Path                              |
|  - Reconstruct masked tokens                              |
|  - Compute reconstruction loss                            |
+-----------------------------------------------------------+
    |
    v
+-----------------------------------------------------------+
|            Contrastive Learning                            |
|  - Global averaging of HSI features                       |
|  - Project to common embedding space                      |
|  - Compute inter-modality similarities                    |
+-----------------------------------------------------------+
    |
    v
   Loss
 (Reconstruction
   + Contrastive)

"""

"""
Input Processing Diagram:
------------------------
Original HSI Input       Original IR Input        Original AF Input        Original Thickness Input
[B,1,91,500,500]         [B,1,128,128]            [B,1,256,256]           [B,1,200,200]
      |                         |                         |                    |
      v                         v                         v                    v
+------------------+   +------------------+   +------------------+   +------------------+
|Spectral/Spatial  |  |Spatial Registration|  |Spatial Registration|  |Spatial Registration|
|Registration     |  |(analysis_dim=500) |  |(analysis_dim=500) |  |(analysis_dim=500) |
+------------------+   +------------------+   +------------------+   +------------------+
      |                         |                         |                    |
      v                         v                         v                    v
[B,1,30,500,500]         [B,1,500,500]             [B,1,500,500]        [B,1,500,500]
      |                         |                         |                    |
      v                         v                         v                    v
+------------+          +--------------+          +--------------+    +--------------+
|3D PatchEmbed|         |ViT Encoder   |          |ViT Encoder   |    |ViT Encoder   |
|(500,25x25,5)|         |(500,25x25,16)|          |(500,25x25,16)|    |(500,25x25,16)|
+------------+          |+mean(dim=1)  |          |+mean(dim=1)  |    |+mean(dim=1)  |
      |                 +--------------+          +--------------+    +--------------+
      v                         |                         |                    |
      v                         v                         v                    v
[B,2400,768]             [B,256]                  [B,256]               [B,256]
(patch tokens)     (global vector after       (global vector after  (global vector after
                   explicit averaging)        explicit averaging)   explicit averaging)

Project Overview:
----------------
A transformer-based architecture for processing hyperspectral imagery (HSI)
with multi-modal auxiliary inputs, combining:
1. Masked Autoencoder (MAE) for self-supervised learning
2. Cross-modal conditioning via cross-attention
3. Contrastive learning for multi-modal representation alignment

Key Architectural Components:
----------------------------
- Spatial Registration Module
- 3D Patch Embedding
- Cross-Attention Mechanism
- Transformer Encoder
- Masked Autoencoder Decoder
- Contrastive Learning Path

Input Processing Configuration:
------------------------------
Input Dimensions:
- HSI: [B, 1, 91, 500, 500]
  * Batch dimension: B
  * Channels: 1
  * Spectral bands: 91
  * Spatial: 500 × 500

Auxiliary Modalities:
- IR: [B, 1, 128, 128]
- AF: [B, 1, 256, 256]
- Thickness: [B, 1, 200, 200]

Preprocessing Steps:
-------------------
1. Spectral Band Selection:
   - Reduce 91 spectral bands to 30
   - Selection strategy:
     * Indices: [0, 2, 4, 6, ..., 56, 80]
     * 29 bands from 0-56 range (every 2nd band)
     * Additional band at index 80
   - Reduction Ratio: ~33% of original spectral information

2. Spatial Registration:
   - Resize all modalities to common spatial dimension (500 × 500)
   - Preserve key spectral and spatial characteristics

Cross-Attention Mechanism Diagram:
---------------------------------
   HSI Patch Tokens          Auxiliary Global Vectors
   [B, n_visible, 768]        [B, 3, 768]
        |                       |
        |                       |
        v                       v
   +------------------------------------------------------+
   |               Cross-Attention Block                  |
   |                                                      |
   |   1. Concatenate HSI tokens with auxiliary tokens    |
   |      [B, n_visible, 768] + [B, 3, 768]               |
   |      → [B, n_visible+3, 768]                         |
   |                                                      |
   |   2. Self-attention between HSI and auxiliary        |
   |                                                      |
   |   3. Extract updated HSI tokens                      |
   |      [B, n_visible, 768]                             |
   |                                                      |
   +------------------------------------------------------+
                  |
                  v
       Global Averaging
       [B, 768]

Contrastive Learning Stage Diagram:
----------------------------------
    HSI Features          Auxiliary Features
    [B,2400,768]           [B,256]
        |                    |
        v                    v
    Global Avg           Already Global
        |                    |
        v                    v
    [B,768]              [B,768]
        |                    |
        v                    v
    Proj Head           Proj Head
        |                    |
        v                    v
    [B,768]              [B,768]
        |                   /
        |                  /
        v                 v
    +------------------------------------------------------------+
    |                 Similarity Matrices                         |
    |  sim = torch.matmul(z_hsi, z_aux.T)/temperature            |
    +------------------------------------------------------------+
                              |
                              v
    +------------------------------------------------------------+
    |              Cross-Entropy Loss                             |
    |  (Encouraging same-patient matches)                         |
    +------------------------------------------------------------+
                              |
                              v
                        Average Losses

Patch Embedding Calculations:
---------------------------
1. HSI Patch Embedding:
   - Input Size: 500 × 500 × 30 (after band selection)
   - Patch Size: 25 × 25 spatial, 5 temporal
   - Grid Size: (500/25) × (500/25) × (30/5)
   - Patch Grid: 20 × 20 × 6
   - Total Patches: 20 × 20 × 6 = 2,400 patches
   - Output Shape: [B, 2400, 768]

2. Auxiliary Patch Embedding (ViT):
   - Input Size: 500 × 500 (after registration)
   - Patch Size: 25 × 25
   - Grid Size: 20 × 20
   - Total Patches: 20 × 20 = 400 patches
   - Output Shape: [B, 400, 256]
   - CRITICAL: Final averaging step (x.mean(dim=1)) → [B, 256]

Configuration Parameters:
----------------------
model = MultiModalSpectralGPT(
    analysis_dim=500,           # Common spatial dimension
    patch_size=(25, 25),        # Spatial patch size
    embed_dim=768,              # Main transformer embedding dimension
    t_patch_size=5,             # Temporal/spectral patch size
    in_chans=1,                 # HSI input channels
    aux_chans=1,                # Auxiliary modality channels
    num_frames=30,              # Selected spectral bands
    aux_encoder_type='vit',     # Auxiliary encoder type
    depth=16,                   # Transformer layers
    num_heads=12,               # Attention heads
    mask_ratio=0.75,            # Proportion of masked tokens
    temperature=0.07            # Contrastive loss temperature
)

Training Objectives:
------------------
1. Reconstruction Loss:
   - L2 loss on masked tokens
   - Computed in 768-dimensional embedding space
   - Weighted by masking pattern
   - Mask Ratio: 0.75 (75% of tokens masked)

2. Contrastive Loss:
   - Align global HSI representations with auxiliary modality representations
   - Temperature-scaled similarity calculation
   - Encourages same-patient matches across modalities
   - Handles missing modalities gracefully

Misregistration Robustness Strategies:
-------------------------------------
1. Global Averaging
   - ViT Auxiliary Encoder: Explicit mean() operation
   - Contrastive Path: Mean pooling across patch tokens
   - Reduces sensitivity to precise spatial alignment

2. Cross-Attention Conditioning
   - Allows information flow between HSI patches and global auxiliary features
   - Patch-level feature integration independent of spatial position

3. Flexible Modality Handling
   - Supports variable availability of auxiliary modalities
   - Scales contrastive loss based on available modalities

Computational Characteristics:
----------------------------
- Input Tensor Size: ~100 MB
- Model Parameters: ~100M
- Forward Pass Computation: ~15 GFLOPs

Implementation Workflow:
----------------------
1. Spatial Registration
   - Resize all input modalities
   - Select spectral bands for HSI

2. Patch Embedding
   - Convert images to patch tokens
   - Apply positional embeddings

3. Random Masking
   - Mask 75% of HSI patch tokens
   - Preserve token order for reconstruction

4. Cross-Modal Conditioning
   - Project auxiliary features
   - Apply cross-attention to condition HSI features

5. Transformer Processing
   - Apply transformer blocks
   - Normalize features

6. Decoding
   - Project to decoder dimension
   - Reconstruct masked tokens

7. Contrastive Learning
   - Compute global representations
   - Calculate inter-modality similarities

Extensibility Considerations:
---------------------------
- Configurable encoder types (ViT/CNN)
- Adaptable to different input modalities
- Flexible patch size and embedding dimensions
- Handles variable input sizes through spatial registration
"""