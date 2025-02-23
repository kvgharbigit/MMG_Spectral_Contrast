"""
Multi-Modal Guided SpectralGPT Architecture (Updated)
===================================================

Key Updates:
------------
- Separate image sizes for HSI and auxiliary modalities
- Flexible patch embedding
- Configurable auxiliary encoder types (ViT or CNN)

Input Processing:
---------------
HSI Input                 IR Input                  AF Input              Thickness Input
[B,C,T,H,W]              [B,aux_chans,H,W]         [B,aux_chans,H,W]    [B,1,H,W]
      |                         |                         |                    |
      v                         v                         v                    v
+------------+          +--------------+          +--------------+    +--------------+
|3D PatchEmbed|         |Aux Encoder   |          |Aux Encoder   |    |Aux Encoder   |
|(hsi_img_size|         |(CNN or ViT)  |          |(CNN or ViT)  |    |(CNN or ViT)  |
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
- hsi_img_size: Spatial dimensions for HSI (e.g., 500x500)
- aux_img_size: Spatial dimensions for auxiliary modalities (e.g., 128x128)
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
1. Patch Embedding
   - HSI: 3D patch embedding with spatial and temporal patches
   - Auxiliary: 2D patch embedding (ViT or CNN)

2. Positional Embedding
   - Add learnable position embeddings to patch tokens

3. Random Masking
   - Randomly mask a proportion of tokens (default 75%)
   - Maintain original token order for reconstruction

4. Cross-Modal Conditioning
   - Project auxiliary features to main embedding dimension
   - Apply cross-attention to condition HSI features

5. Transformer Processing
   - Apply transformer blocks to process tokens
   - Normalize features

6. Decoding Path
   - Project features to decoder dimension
   - Append mask tokens
   - Reconstruct original token embeddings

7. Contrastive Learning
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
    hsi_img_size=500,
    aux_img_size=128,
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
Multi-Modal Guided SpectralGPT Architecture (Concrete Implementation)
===================================================================

Concrete Model Configuration:
----------------------------
- hsi_img_size: 224 (default spatial dimensions for HSI)
- aux_img_size: 128 (default auxiliary modalities dimensions)
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
HSI Input                 IR Input                  AF Input              Thickness Input
[B,1,12,224,224]         [B,3,128,128]             [B,3,128,128]        [B,1,128,128]
      |                         |                         |                    |
      v                         v                         v                    v
+------------+          +--------------+          +--------------+    +--------------+
|3D PatchEmbed|         |ViT Encoder   |          |ViT Encoder   |    |ViT Encoder   |
|(224x224,16)|          |(128x128,16)  |          |(128x128,16)  |    |(128x128,16)  |
+------------+          +--------------+          +--------------+    +--------------+
      |                         |                         |                    |
      v                         v                         v                    v
[B,784,768]         [B,64,256]           [B,64,256]        [B,64,256]
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
   - Input Size: 224 × 224 × 12
   - Patch Size: 16 × 16 spatial, 3 temporal
   - Grid Size: (224/16) × (224/16) × (12/3)
   - Patch Grid: 14 × 14 × 4
   - Total Patches: 14 × 14 × 4 = 784 patches
   - Output Shape: [B, 784, 768]

2. Auxiliary Patch Embedding:
   - Input Size: 128 × 128
   - Patch Size: 16 × 16
   - Grid Size: 8 × 8
   - Total Patches: 8 × 8 = 64 patches
   - Output Shape per Modality: [B, 64, 256]

Masking Scenario:
----------------
- Mask Ratio: 0.75
- Total Patches: 784
- Visible Tokens: 25% of 784 = 196 tokens
- Masked Tokens: 588 tokens

Computational Characteristics:
----------------------------
- Input Tensor Size: ~50 MB
- Model Parameters: ~100M
- Forward Pass Computation: ~10 GFLOPs

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
    hsi_img_size=224,
    aux_img_size=128,
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