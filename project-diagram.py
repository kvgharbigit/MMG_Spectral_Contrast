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
|  - Create thickness mask for rim exclusion                |
+-----------------------------------------------------------+
    |               |               |             |
    v               v               v             v          +----------+
[B,1,30,500,500] [B,1,500,500] [B,1,500,500] [B,1,500,500]  | Thickness|
    |               |               |             |          |   Mask   |
    v               v               v             v          |[B,1,500,500]
+-----------------------------------------------------------+    |
|                Patch Embedding                             |    |
|  - 3D Patch Embedding (HSI)                               |    |
|  - 2D Patch Embedding (Auxiliary)                         |    |
|  - Position Embeddings                                    |    |
+-----------------------------------------------------------+    |
    |               |               |             |               |
    v               v               v             v               v
[B,2400,768]   [B,400,256]     [B,400,256]   [B,400,256]  [B,2400] Patch Mask
    |               |               |             |               |
    v               v               v             v               |
+---------------------------------+  +-------------------------+  |
|     MAE Reconstruction Path     |  | Contrastive Learning    |  |
|  - Random Masking (75%)         |  | Path (Unmasked)         |<-+
|  - Cross-Attention Conditioning |  | - Global or Spatial Mode|
|  - Transformer Processing       |  +-------------------------+
|  - Direct Pixel Prediction      |          |
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

Dual-Mode Contrastive Learning Diagram (With Rim Masking):
--------------------------------------

 Mode 1: Global Contrastive Learning
 -----------------------------------
   HSI Features              Auxiliary Features       Thickness Mask
   [B,2400,768]                 [B,256]                 [B,2400]
       |                           |                        |
       v                           v                        v
   Masked Features           Modality Projection           |
   (rim excluded)                  |                        |
       |                           v                        |
       v                         [B,768]                    |
Weighted Average Pooling           |                        |
(mask_sum normalization)           |                        |
       |                           |                        |
       v                           v                        |
     [B,768]                     [B,768]                    |
       |                           |                        |
       v                           v                        |
   Global Projection Head    Global Projection Head         |
       |                           |                        |
       v                           v                        |
     [B,768]                     [B,768]                    |
       |                           |                        |
       +-------------+-------------+                        |
                     |                                      |
                     v                                      |
           Cross-Modal Similarity                           |
           (Patient-Level Labels)                           |
                     |                                      |
                     +--------------------------------------+
                                      |
                                      v
                           Loss on Valid Regions Only


 Mode 2: Spatial Contrastive Learning
 -----------------------------------
   HSI Features             Auxiliary Features        Thickness Mask
   [B,2400,768]              [B,400,256]               [B,2400]
       |                          |                        |
       v                          v                        v
   Group by Spatial         Aux Spatial Projection    Group by Spatial
   Location                       |                    Location
       |                          v                        |
       v                       [B,400,768]                 v
   [B,400,4608]                   |                   [B,400] Spatial Mask
       |                          v                        |
       v                     Global Projection Head        |
   Spatial Projection Head        |                        |
       |                          v                        |
       v                       [B,400,768]                 |
   [B,400,768]                    |                        |
       |                          |                        |
       +-------------+------------+                        |
                     |                                     |
                     v                                     |
      Filtered Valid Patch Features <---------------------+
       (only use patches where mask > 0.5)
                     |
                     v
          Patch-Level Similarities
           (Spatial Location Labels)
                     |
                     v
            Loss on Valid Patches

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

Thickness Mask Transformation:
----------------------------
1. Start with pixel mask: [B, 1, 500, 500]
   - Binary mask (1 for valid pixels, 0 for black rim)

2. Convert to patch-level mask through average pooling:
   - Apply avg_pool2d with kernel_size=patch_size
   - Results in spatial mask of shape [B, 1, 20, 20]

3. Reshape and expand across spectral dimension:
   - Expand to shape [B, 20*20, 6]
   - Reshape to [B, 2400] for full patch-level mask

4. Apply threshold (0.3) to determine valid patches:
   - A patch is valid if at least 30% of pixels are valid
   - Creates binary patch-level mask

5. For spatial contrastive learning:
   - Reshape to [B, 400, 6] (group by spatial location)
   - Average across spectral dimension to get [B, 400]
   - Only retain spatial patches with mask value > 0.5

Cross-Attention Conditioning Mechanism:
----------------------------------
The cross-attention mechanism allows auxiliary modalities to guide the HSI encoder by providing conditioning signals.

Tensor Dimensions and Transformations:
1. HSI Encoder Features (after masking): [B, num_unmasked, embed_dim]
   - B: Batch size
   - num_unmasked: Number of unmasked tokens (~600 with 75% masking)
   - embed_dim: 768

2. Auxiliary Global Features: [B, aux_embed_dim]
   - B: Batch size
   - aux_embed_dim: 256

3. Projection and Preparation:
   - Project auxiliary: [B, aux_embed_dim] → [B, embed_dim]
   - Add dimension: [B, embed_dim] → [B, 1, embed_dim]

4. Concatenation for Cross-Attention:
   - HSI tokens: [B, num_unmasked, embed_dim]
   - Auxiliary token: [B, 1, embed_dim]
   - Concatenated: [B, num_unmasked+1, embed_dim]

5. Apply Cross-Attention Block:
   - Input: [B, num_unmasked+1, embed_dim]
   - Self-attention allows HSI tokens to attend to auxiliary token
   - Output: [B, num_unmasked+1, embed_dim]

6. Extract Conditioned HSI Tokens:
   - Take first num_unmasked tokens: [B, num_unmasked, embed_dim]
   - Add residual connection to original HSI tokens
   - Process through next cross-attention layer with next modality

The Implementation Code:
```python
# For each auxiliary modality
for modality, embedding in aux_embeddings.items():
    # Project auxiliary embedding to the HSI embedding dimension
    cond_tokens = self.modality_proj(embedding).unsqueeze(1)  # Shape: [B, 1, embed_dim]

    # Apply cross-attention for each layer
    for block in self.cross_attn:
        # Concatenate HSI tokens with auxiliary token
        concat_tokens = torch.cat([x, cond_tokens], dim=1)  # Shape: [B, num_unmasked+1, embed_dim]

        # Apply self-attention block (allows HSI tokens to attend to auxiliary)
        attended = block(concat_tokens)  # Shape: [B, num_unmasked+1, embed_dim]

        # Take only the HSI tokens and add residual connection
        x = x + attended[:, :-1, :]  # Shape: [B, num_unmasked, embed_dim]
```

This approach allows:
1. Each HSI token to selectively attend to auxiliary information
2. Soft conditioning rather than hard fusion of modalities
3. Auxiliary guidance of the HSI encoder
4. Processing multiple auxiliary modalities sequentially

For thickness modality, the encoder can be modified to focus only on valid regions by using a weighted average pooling approach, where black rim areas are excluded from the global representation.

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

Contrastive Loss Modes (with Rim Masking):
---------------------
1. Global Mode:
   - Applies thickness mask before computing global representation
   - Uses weighted average pooling (sum of masked features / sum of mask)
   - Similarity matrix: [B, B] for each modality
   - Labels: Batch indices (diagonal matches)

2. Spatial Mode:
   - Filters out patches from black rim areas before computing similarities
   - Collects valid patches based on thickness mask threshold
   - Creates similarity matrix only for valid patches
   - Labels: Valid patch indices (same spatial location matches)

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

Updated Reconstruction Path Flow:
-----------------------------
1. Encoder and Conditioning (same as before)
   - Apply random masking (75%)
   - Process with transformer blocks and cross-attention
   - Output encoded latent features

2. Decoder Path (Updated)
   - Project latent to decoder dimension: decoder_embed
   - Add mask tokens for masked positions
   - Process through decoder transformer blocks
   - Previously: Predict embeddings, then convert to pixels via pixel_projection
   - Now: Directly predict pixels with decoder_pred
   - Shape: [B, L, patch_pixels] (patch_pixels = patch_h * patch_w * t_patch * in_chans)

3. Unpatchify (Updated)
   - Take predicted patch pixels (no separate projection needed)
   - Reshape to organize patches: [B, spectral_patches, spatial_patches, patch_pixels]
   - Further reshape to separate dimensions: [B, spectral_patches, spatial_h, spatial_w, t_patch, patch_h, patch_w, C]
   - Permute dimensions to align with original format
   - Final shape: [B, C, T, H, W]

4. Loss Calculation (Same)
   - Apply token mask to pixel mask
   - Only compute loss on masked pixels
   - If thickness mask is used, combine with MAE mask
   - Normalize by count of valid pixels

Training Process Flow (with Rim Masking):
--------------------
1. Registration & Preprocessing
   - Resize all modalities to 500×500
   - Select 30 spectral bands from HSI
   - Create thickness mask (1 for valid regions, 0 for black rim)

2. Patch Embedding & Mask Conversion
   - Create tokenized representations for all modalities
   - Convert pixel-level mask to patch-level mask

3. Dual Learning Path with Rim Masking
   a. MAE Reconstruction Path:
      - Apply random masking (75%)
      - Apply cross-attention conditioning
      - Transformer processing of latent features
      - Decoder directly predicts pixels (updated)
      - Only compute loss on valid regions (not in rim)

   b. Contrastive Path with Rim Masking:
      - Use unmasked embeddings
      - Exclude black rim areas using the thickness mask
      - Select contrastive mode (global/spatial)
      - Compute contrastive loss on valid regions only

4. Combined Loss
   - Total Loss = Reconstruction Loss + Contrastive Loss
   - Both losses exclude rim areas

Advantages of Direct Pixel Prediction:
----------------------------------
1. Simplified Architecture
   - Removes an intermediate transformation layer (pixel_projection)
   - More direct learning signal from loss to decoder

2. More Efficient Training
   - Fewer parameters to train
   - More direct gradient flow

3. Better Reconstructions
   - Decoder learns to directly predict the pixel values
   - No need to learn a separate embedding-to-pixel mapping

4. Compatible with Existing Benefits
   - Maintains all advantages of rim masking
   - Preserves both global and spatial contrastive learning modes
   - Still supports cross-modal conditioning
"""