"""
Spatially Registered Multi-Modal Guided SpectralGPT Architecture
===============================================================

Key Updates:
------------
- Uniform spatial dimensions across all modalities using analysis_dim
- Spatial registration preprocessing module
- Single configurable dimension for all modalities
- Flexible patch embedding
- Configurable auxiliary encoder types (ViT or CNN)

Input Processing with Spatial Registration:
------------------------------------------
Original HSI Input       Original IR Input        Original AF Input        Original Thickness Input
[B,C,T,H,W]              [B,aux_chans,H,W]         [B,aux_chans,H,W]        [B,1,H,W]
      |                         |                         |                    |
      v                         v                         v                    v
+------------------+   +------------------+   +------------------+   +------------------+
|Spatial Registration|  |Spatial Registration|  |Spatial Registration|  |Spatial Registration|
|(analysis_dim)     |  |(analysis_dim)     |  |(analysis_dim)     |  |(analysis_dim)     |
+------------------+   +------------------+   +------------------+   +------------------+
      |                         |                         |                    |
      v                         v                         v                    v
Registered HSI        Registered IR           Registered AF           Registered Thickness
[B,C,T,analysis_dim,  [B,aux_chans,          [B,aux_chans,          [B,1,analysis_dim,
 analysis_dim]         analysis_dim,          analysis_dim,          analysis_dim]
                       analysis_dim]          analysis_dim]
      |                         |                         |                    |
      v                         v                         v                    v
+------------+          +--------------+          +--------------+    +--------------+
|3D PatchEmbed|         |Aux Encoder   |          |Aux Encoder   |    |Aux Encoder   |
|(analysis_dim|         |(CNN or ViT)  |          |(CNN or ViT)  |    |(CNN or ViT)  |
| patch_size) |         |              |          |              |    |              |
+------------+          +--------------+          +--------------+    +--------------+
      |                         |                         |                    |
      v                         v                         v                    v
[B,num_patches,     [B,aux_embed_dim]          [B,aux_embed_dim]      [B,aux_embed_dim]
 embed_dim]                     |                         |                    |
      |                         |                         |                    |
      v                         v                         v                    v
+Position Embed        Auxiliary LayerNorm       Auxiliary LayerNorm  Auxiliary LayerNorm
      |                         |                         |                    |
      v                         |                         |                    |
Random Masking                  |                         |                    |
(mask_ratio=0.75)              |                         |                    |
      |                         \                        |                   /
      |                          \                       |                  /
      |                           \                      |                 /
      |                        Modality Projection (aux_embed_dim -> embed_dim)
      |                                       |
      |              /-------------------------/
      v             v
+------------------+
|Cross-Attn Blocks |
|(depth layers)    |
+------------------+
         |
         v
+------------------+
|Transformer Blocks|
|(depth=12)        |
+------------------+
         |
    /----|----\
    |         |
    v         v
Decoder     Contrastive
Path        Path

Configuration Parameters:
----------------------
- analysis_dim: Common spatial dimension for all modalities (e.g., 224)
- patch_size: Spatial patch size for tokenization
- in_chans: Input channels for HSI (typically 1)
- aux_chans: Channels for auxiliary modalities (e.g., 3 for RGB)
- embed_dim: Main transformer embedding dimension (e.g., 768)
- aux_embed_dim: Embedding dimension for auxiliary features (e.g., 256)
- depth: Number of transformer layers
- num_heads: Number of attention heads
- aux_encoder_type: Type of auxiliary encoder ('cnn' or 'vit')

Detailed Forward Pass:
--------------------
1. Spatial Registration
   - Resize all modalities to same spatial dimensions (analysis_dim × analysis_dim)
   - Preserve temporal dimension in HSI

2. Patch Embedding
   - HSI: 3D patch embedding with spatial and temporal patches
   - Auxiliary: 2D patch embedding (ViT or CNN)

3. Positional Embedding
   - Add learnable position embeddings to patch tokens

4. Random Masking
   - Randomly mask a proportion of tokens (default 75%)
   - Maintain original token order for reconstruction

5. Cross-Modal Conditioning
   - Project auxiliary features to main embedding dimension
   - Apply cross-attention to condition HSI features

6. Transformer Processing
   - Apply transformer blocks to process tokens
   - Normalize features

7. Decoding Path
   - Project features to decoder dimension
   - Append mask tokens
   - Reconstruct original token embeddings

8. Contrastive Learning
   - Mean pool features
   - Project to contrastive embedding space
   - Compute similarity across modalities

Training Objectives:
-----------------
1. Reconstruction Loss
   - L2 loss on masked tokens
   - Computed in embedding space
   - Weighted by masking pattern

2. Contrastive Loss
   - Align representations across modalities
   - Temperature-scaled similarity
   - Cross-entropy loss with batch indices

Example Configuration:
--------------------
model = MultiModalSpectralGPT(
    analysis_dim=224,
    patch_size=16,
    in_chans=1,
    aux_chans=3,
    embed_dim=768,
    aux_embed_dim=256,
    depth=12,
    num_heads=12,
    aux_encoder_type='vit',
    mask_ratio=0.75
)
"""


"""
Spatially Registered Multi-Modal Guided SpectralGPT Architecture (Concrete Implementation)
=======================================================================================

Concrete Model Configuration:
----------------------------
- analysis_dim: 224 (common spatial dimension for all modalities)
- patch_size: 16 (spatial patch size)
- in_chans: 1 (HSI input channels)
- aux_chans: 3 (Auxiliary modalities channels, e.g., RGB)
- embed_dim: 768 (main transformer embedding dimension)
- aux_embed_dim: 256 (auxiliary features embedding dimension)
- depth: 16 (number of transformer layers)
- num_heads: 12 (number of attention heads)
- num_frames: 12 (spectral bands in HSI)
- t_patch_size: 3 (temporal/spectral patch size)
- aux_encoder_type: 'vit' (auxiliary encoder type)
- mask_ratio: 0.75 (proportion of tokens masked)

Input Processing Specifics:
-------------------------
Original HSI Input       Original IR Input        Original AF Input        Original Thickness Input
[B,1,12,256,256]        [B,3,128,128]            [B,3,192,192]           [B,1,100,100]
      |                         |                         |                    |
      v                         v                         v                    v
+------------------+   +------------------+   +------------------+   +------------------+
|Spatial Registration|  |Spatial Registration|  |Spatial Registration|  |Spatial Registration|
|(analysis_dim=224) |  |(analysis_dim=224) |  |(analysis_dim=224) |  |(analysis_dim=224) |
+------------------+   +------------------+   +------------------+   +------------------+
      |                         |                         |                    |
      v                         v                         v                    v
[B,1,12,224,224]         [B,3,224,224]             [B,3,224,224]        [B,1,224,224]
      |                         |                         |                    |
      v                         v                         v                    v
+------------+          +--------------+          +--------------+    +--------------+
|3D PatchEmbed|         |ViT Encoder   |          |ViT Encoder   |    |ViT Encoder   |
|(224x224,16)|          |(224x224,16)  |          |(224x224,16)  |    |(224x224,16)  |
+------------+          +--------------+          +--------------+    +--------------+
      |                         |                         |                    |
      v                         v                         v                    v
[B,784,768]         [B,196,256]           [B,196,256]        [B,196,256]
      |                         |                         |                    |
      v                         v                         v                    v
+Position Embed   Aux LayerNorm  Aux LayerNorm   Aux LayerNorm
      |                |              |               |
      v                v              v               v
Random Masking    Modality Projection (aux_embed_dim -> embed_dim)
(mask_ratio=0.75)         |
      |                   |
      |    /---------------/
      v   v
+------------------+
|Cross-Attn Blocks |
|(16 layers)       |
+------------------+
         |
         v
+------------------+
|Transformer Blocks|
|(16 layers)       |
+------------------+
         |
    /----|----\
    |         |
    v         v
Decoder     Contrastive
Path        Path

Patch Embedding Calculations:
---------------------------
1. HSI Patch Embedding:
   - Input Size: 224 × 224 × 12 (after registration)
   - Patch Size: 16 × 16 spatial, 3 temporal
   - Grid Size: (224/16) × (224/16) × (12/3)
   - Patch Grid: 14 × 14 × 4
   - Total Patches: 14 × 14 × 4 = 784 patches
   - Output Shape: [B, 784, 768]

2. Auxiliary Patch Embedding:
   - Input Size: 224 × 224 (after registration)
   - Patch Size: 16 × 16
   - Grid Size: 14 × 14
   - Total Patches: 14 × 14 = 196 patches
   - Output Shape per Modality: [B, 196, 256]

Spatial Registration Process:
---------------------------
1. HSI Registration:
   - For each temporal slice t in [0, T-1]:
     - Resize spatial dimensions from original to [analysis_dim × analysis_dim]
   - Preserve all spectral/temporal information

2. Auxiliary Registration:
   - Resize spatial dimensions from original to [analysis_dim × analysis_dim]
   - Preserve channel information

Masking Scenario:
----------------
- Mask Ratio: 0.75
- Total Patches: 784
- Visible Tokens: 25% of 784 = 196 tokens
- Masked Tokens: 588 tokens

Computational Characteristics:
----------------------------
- Input Tensor Size: ~60 MB
- Model Parameters: ~100M
- Forward Pass Computation: ~12 GFLOPs

Training Objectives:
------------------
1. Reconstruction Loss
   - L2 loss on 588 masked tokens
   - Computed in 768-dimensional embedding space
   - Weighted by masking pattern

2. Contrastive Loss
   - Align HSI representations with 3 auxiliary modalities
   - Temperature: 0.07
   - Similarity computed in 768-dimensional space

Example Code:
------------
model = MultiModalSpectralGPT(
    analysis_dim=224,
    patch_size=16,
    in_chans=1,
    aux_chans=3,
    embed_dim=768,
    aux_embed_dim=256,
    depth=16,
    num_heads=12,
    num_frames=12,
    t_patch_size=3,
    aux_encoder_type='vit',
    mask_ratio=0.75
)
"""