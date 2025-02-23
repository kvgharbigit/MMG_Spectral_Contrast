"""
Multi-Modal Guided SpectralGPT Architecture
=========================================

Input Processing:
---------------
HSI Input                 IR Input                  AF Input              Thickness Input
[B,C,T,H,W]              [B,aux_chans,H,W]         [B,aux_chans,H,W]    [B,1,H,W]
      |                         |                         |                    |
      v                         v                         v                    v
+------------+          +--------------+          +--------------+    +--------------+
|3D PatchEmbed|         |Aux Encoder   |          |Aux Encoder   |    |Aux Encoder   |
|(img_size=128|         |(CNN or ViT)  |          |(CNN or ViT)  |    |(CNN or ViT)  |
|patch_size=8)|         |              |          |              |    |              |
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

Decoder Path:               Contrastive Path:
------------               ---------------
    |                          |
    v                          v
Decoder Embed           Mean Pooling
[B,L*(1-mask_ratio),    [B,embed_dim]
 decoder_embed_dim]           |
    |                         v
    v                   Projection Head
Append Mask Tokens      Linear->ReLU->Linear
    |                         |
    v                         v
+Decoder Pos Embed      Contrastive Loss (against the other modalities in the contrastive space)
    |                   (temp=0.07)
    v
Decoder Blocks
(depth=8)
    |
    v
Decoder Norm
    |
    v
Decoder Prediction
[B,num_patches,
 embed_dim]

Example with batch_size=32, img_size=128, num_frames=12:
===============================================

Input Processing:
---------------
HSI Input                 IR Input                  AF Input              Thickness Input
[32,1,12,128,128]        [32,3,128,128]           [32,3,128,128]       [32,1,128,128]
      |                         |                         |                    |
      v                         v                         v                    v
Patch Embed             Aux Encoder              Aux Encoder            Aux Encoder
[32,1024,768]           [32,256]                [32,256]               [32,256]
      |                         |                         |                    |
      v                         v                         v                    v
+Position Embed        Auxiliary LayerNorm       Auxiliary LayerNorm  Auxiliary LayerNorm
[32,1024,768]          [32,256]                 [32,256]              [32,256]
      |                         |                         |                    |
      v                         |                         |                    |
Random Masking                  |                         |                    |
[32,256,768]                   |                         |                    |
      |                         \                        |                   /
      |                          \                       |                  /
      |                           \                      |                 /
      |                        Modality Projection [32,256] -> [32,768]
      |                                       |
      |              /-------------------------/
      v             v
Cross-Attention Blocks
[32,256,768]
      |
      v
Transformer Blocks
[32,256,768]
      |
    /----|----\
    |         |
    v         v
Decoder     Contrastive
Path        Path

Decoder Path (HSI data):   Contrastive Path (HSI data):
------------               ---------------
    |                          |
    v                          v
Decoder Embed            Mean Pooling
[32,256,512]             [32,768]
    |                         |
    v                         v
Append Mask Tokens       Projection Head
[32,1024,512]           [32,768]
    |                         |
    v                         v
+Decoder Pos Embed      Contrastive Loss
[32,1024,512]
    |
    v
Decoder Blocks
[32,1024,512]
    |
    v
Decoder Norm
[32,1024,512]
    |
    v
Decoder Prediction
[32,1024,768]

Dimension Calculations:
--------------------
num_patches = (H/patch_size) * (W/patch_size) * (T/t_patch_size)
            = (128/8) * (128/8) * (12/3)
            = 16 * 16 * 4
            = 1024 tokens

visible_tokens = num_patches * (1 - mask_ratio)
               = 1024 * (1 - 0.75)
               = 256 tokens

Configuration Parameters:
----------------------
embed_dim: 768          # Main embedding dimension
aux_embed_dim: 256      # Auxiliary embedding dimension
depth: 12              # Number of transformer blocks
num_heads: 12          # Number of attention heads
decoder_embed_dim: 512  # Decoder embedding dimension
decoder_depth: 8       # Number of decoder layers
decoder_num_heads: 16  # Number of decoder attention heads
mlp_ratio: 4.0        # MLP hidden dimension ratio
mask_ratio: 0.75      # Ratio of tokens to mask
aux_encoder_type: cnn  # Type of auxiliary encoder

Training Objectives:
-----------------
1. Reconstruction Loss:
   - L2 loss on masked tokens
   - Computed in embedding space
   - Weighted by mask

2. Contrastive Loss:
   - Between HSI and auxiliaries
   - Temperature-scaled similarity
   - Cross entropy with batch indices
   - Averaged across modalities

Key Features:
-----------
- Modular architecture
- Flexible auxiliary encoders
- Handles missing modalities
- Configurable masking
- Multi-objective training
- Layer normalization
- Dropout support
- Separate positional embeddings

Note: All dimensions shown are for example case with:
- batch_size = 32
- img_size = 128x128
- num_frames = 12
- t_patch_size = 3
- patch_size = 8x8
"""


"""Contrastive Loss Workflow - ASCII Visualization

+---------------------------------------------------+
|                Input Modalities                   |
+-------------------+-------------------+-----------+
| HSI               | Auxiliary         | Batch     |
| [B,C,T,H,W]       | Modalities        | Indices   |
|                   | [B,aux_chans,H,W] | [B]       |
+-------------------+-------------------+-----------+
             |               |               |
             v               v               |
    +--------+-------+  +----+------+        |
    | HSI Feature    |  | Aux       |        |
    | Extraction     |  | Encoders  |        |
    | - Patch Embed  |  | - IR      |        |
    | - Transformers |  | - AF      |        |
    | - Mean Pooling |  | - Thick.  |        |
    +--------+-------+  +----+------+        |
             |               |               |
             v               v               |
    +--------+---------------+-------+       |
    |    Feature Projection          |       |
    | - Contrastive Space Embedding  |       |
    | - Normalize Dimensions         |       |
    +--------+---------------+-------+       |
             |               |               |
             v               v               |
    +--------+---------------+-------+       |
    |    Similarity Matrix           |       |
    | - Dot Product                  |       |
    | - Temperature Scaling (τ=0.07) |       |
    +--------+---------------+-------+       |
             |               |               |
             v               v               |
    +--------+---------------+-------+       |
    |    Cross-Entropy Loss          |       |
    | - Batch Indices as Labels      |       |
    | - Per-Modality Loss Computation|       |
    +--------+---------------+-------+       |
             |               |               |
             v               v               |
    +--------+---------------+-------+       |
    |    Loss Aggregation            |       |
    | - Average Across Modalities    |       |
    +--------------------------------+       |
             |                               |
             v                               |
    +--------+-------------------------------+
    |    Final Contrastive Loss              |
    +----------------------------------------+

Mathematical Formulation:
L_contrast = (1/|M|) * ∑(m ∈ M) 
             -log(exp(sim(z_HSI, z_m) / τ) / ∑(k ≠ i) exp(sim(z_HSI, z_k) / τ))

Key Characteristics:
- Modality-specific processing
- Temperature-scaled similarities
- Batch-aware positive pair definition
- Handles missing modalities
"""

