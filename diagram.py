"""
Multi-Modal Guided SpectralGPT Architecture with Dual-Mode Contrastive Learning
==============================================================================

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
|  - Position Embeddings                                    |
+-----------------------------------------------------------+
    |               |               |             |
    v               v               v             v
[B,2400,768]   [B,400,256]     [B,400,256]   [B,400,256]
    |               |               |             |
    v               v               v             v
+---------------------------------+  +-------------------------+
|     MAE Reconstruction Path     |  | Contrastive Learning    |
|  - Random Masking (75%)         |  | Path (Unmasked)         |
|  - Cross-Attention Conditioning |  | - Global or Spatial Mode|
|  - Transformer Processing       |  +-------------------------+
|  - Decoding & Reconstruction    |          |
+---------------------------------+          v
     |                          +-------------------------+
     |                          | Mode Selection:         |
     |     +--------------------|  1. Global Contrastive  |
     |     |                    |  2. Spatial Contrastive |
     |     |                    +-------------------------+
     v     v                               |
 +------------------------------------------+
 |  Combined Loss = Reconstruction + Contrastive |
 +------------------------------------------+

Dual-Mode Contrastive Learning Diagram:
--------------------------------------

 Mode 1: Global Contrastive Learning
 -----------------------------------
   HSI Features              Auxiliary Features
   [B,2400,768]                 [B,256]
       |                           |
       v                           v
   Global Mean Pooling     Modality Projection
       |                           |
       v                           v
     [B,768]                     [B,768]
       |                           |
       v                           v
   Global Projection Head    Global Projection Head
       |                           |
       v                           v
     [B,768]                     [B,768]
       |                           |
       +-------------+-------------+
                     |
                     v
           Cross-Modal Similarity
           (Patient-Level Labels)


 Mode 2: Spatial Contrastive Learning
 -----------------------------------
   HSI Features             Auxiliary Features
   [B,2400,768]              [B,400,256]
       |                          |
       v                          v
   Group by Spatial         Aux Spatial Projection
   Location                       |
       |                          v
       v                       [B,400,768]
   [B,400,4608]                   |
       |                          v
       v                     Global Projection Head
   Spatial Projection Head        |
       |                          v
       v                       [B,400,768]
   [B,400,768]                    |
       |                          |
       +-------------+------------+
                     |
                     v
          Patch-Level Similarities
           (Spatial Location Labels)

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

HSI Patch Group Transformation:
-----------------------------
1. Start with token sequence: [B, 2400, 768]
   - 2400 = 6 spectral chunks × 400 spatial patches

2. Reshape to: [B, 6, 400, 768]
   - Explicitly separate spectral and spatial dimensions

3. Transpose to: [B, 400, 6, 768]
   - Group by spatial location

4. Reshape to: [B, 400, 4608]
   - Concatenate spectral features (6×768 = 4608)
   - Each row contains complete spectral information for one spatial location

Projection Heads:
---------------
1. Global Projection Head:
   - Input: 768 dimensions (Standard embedding dimension)
   - Output: 768 dimensions (Contrastive space)
   - Used for: Global HSI features and all auxiliary features

2. Spatial Projection Head:
   - Input: 4608 dimensions (Concatenated spectral information)
   - Output: 768 dimensions (Contrastive space)
   - Used for: Spatial HSI features only

Contrastive Loss Modes:
---------------------
1. Global Mode:
   - Aligns patient-level HSI and auxiliary representations
   - Uses mean pooling across all patches
   - Similarity matrix: [B, B] for each modality
   - Labels: Batch indices (diagonal matches)

2. Spatial Mode:
   - Aligns patch-level HSI and auxiliary representations
   - Groups HSI patches by spatial location across spectral bands
   - Similarity matrix: [B×400, B×400] for each modality
   - Labels: Patch indices (same spatial location matches)

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
    depth=16,                   # Transformer layers
    num_heads=12,               # Attention heads
    mask_ratio=0.75,            # Proportion of masked tokens
    temperature=0.07,           # Contrastive loss temperature
    contrastive_mode='global'   # or 'spatial' - Contrastive learning mode
)

Training Process Flow:
--------------------
1. Registration & Preprocessing
   - Resize all modalities to 500×500
   - Select 30 spectral bands from HSI

2. Patch Embedding
   - Create tokenized representations for all modalities

3. Dual Learning Path
   a. MAE Path:
      - Apply random masking (75%)
      - Apply cross-attention conditioning
      - Reconstruct masked tokens

   b. Contrastive Path:
      - Use unmasked embeddings
      - Select contrastive mode (global/spatial)
      - Project to contrastive space
      - Compute contrastive loss

4. Combined Loss
   - Total Loss = Reconstruction Loss + Contrastive Loss

Implementation Notes:
-------------------
- Unmasked embeddings are used for contrastive learning
- ViT-only for auxiliary encoders (no CNN option)
- Separate projection heads for global and spatial modes
- Dynamic patch embedding based on input dimensions
- Handles varied input sizes through spatial registration

Spatial Contrastive Mode Advantage:
---------------------------------
- Aligns features at the spatial patch level
- Captures local correspondence between modalities
- Groups spectral information for each spatial location
- Preserves spatial context in contrastive representation

Global Contrastive Mode Advantage:
--------------------------------
- Creates patient-level alignment between modalities
- More robust to misregistration
- Lower computational complexity
- Simpler implementation
"""