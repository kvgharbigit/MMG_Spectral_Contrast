# MultiModalSpectralGPT Architecture Diagrams
#
## 1. Current Configuration (use_multimodal=False, use_cross_attention=False)

```
                                INPUT LAYER
┌─────────────────────────────────────────────────────────────────────────┐
│ HSI Image                    Auxiliary Modalities                       │
│ [B, 1, 30, 224, 224]        ┌────────────┐ ┌────────────┐              │
│       │                     │    IR      │ │    AF      │              │
│       │                     │  (IGNORED) │ │  (IGNORED) │              │
│       │                     └────────────┘ └────────────┘              │
└────────────────┬────────────────────────────────┬───────────────────────┘
                 │                                │
                 │                                X  <- Aux data blocked
┌────────────────┴────────────────┐               │
│     SpatialRegistration         │               │
│   - Already 224x224             │               │
│   - Already 30 spectral bands   │               │
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴────────────────┐               │
│         PatchEmbed              │               │
│  HSI → Patches (7x7, t=5)      │               │
│   [B, 32*32*6, 1024]           │               │
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴────────────────┐               │
│        Position Embed           │               │
│   Add positional information    │               │
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴────────────────┐               │
│          MAE Masking            │               │
│   Random mask 50% of tokens     │               │
│   [B, ~3000, 1024]             │               │
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴────────────────┐               │
│      Transformer Blocks         │               │
│         10 layers @ 1024D       │               │
│         16 attention heads      │               │
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴────────────────┐               │
│           LayerNorm             │               │
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴────────────────┐               │
│         Decoder Embed           │               │
│  Linear(1024, 768)              │               │ 
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴────────────────┐               │
│            Decoder              │               │
│   - Mask tokens restored        │               │
│   - 8 blocks @ 768D             │               │
│   - 8 attention heads           │               │
│   - Reconstruct patches         │               │
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴─────────────────────────────────┐
│                  OUTPUT LOSSES                   │
│ ┌──────────────┐   ┌──────────────┐             │
│ │   MSE Loss   │   │  Diversity   │             │
│ │  (weight=1.0)│   │    Losses    │             │
│ │              │   │  - Intra-p:  │             │
│ │              │   │    0.2       │             │
│ │              │   │  - Inter-p:  │             │
│ │              │   │    0.2       │             │
│ └──────────────┘   └──────────────┘             │
└──────────────────────────────────────────────────┘
```

## 2. Multimodal Configuration (use_multimodal=True, use_cross_attention=True)

```
                                INPUT LAYER
┌─────────────────────────────────────────────────────────────────────────┐
│ HSI Image                    Auxiliary Modalities                       │
│ [B, 1, 30, 224, 224]        ┌────────────┐ ┌────────────┐              │
│       │                     │    IR      │ │    AF      │              │
│       │                     │            │ │            │              │
│       │                     └────────────┘ └────────────┘              │
└────────────────┬────────────────────────────────┬───────────────────────┘
                 │                                │
                 │                                │
┌────────────────┴────────────────┐               │
│     SpatialRegistration         │               │
│   - Already 224x224             │               │
│   - Already 30 spectral bands   │               │
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴────────────────┐               │
│         PatchEmbed              │               │
│  HSI → Patches (7x7, t=5)      │               │
│   [B, 32*32*6, 1024]           │               │
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴────────────────┐               │
│        Position Embed           │               │
│   Add positional information    │               │
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴────────────────┐ ┌─────────────┴─────────────┐
│          MAE Masking            │ │       AuxViTEncoder       │
│   Random mask 50% of tokens     │ │  Process each modality    │
│   [B, ~3000, 1024]             │ │  through ViT              │
└────────────────┬────────────────┘ │  [B, 256]                 │
                 │                  └─────────────┬─────────────┘
                 │                                │
┌────────────────┴────────────────┐               │
│      Transformer Blocks         │               │
│         10 layers @ 1024D       │               │
│         16 attention heads      │               │
└────────────────┬────────────────┘               │
                 │                                │
┌────────────────┴────────────────┐               │
│        Cross-Attention          │               │
│         3 layers @ 1024D        │               │
│   Conditions on aux features    │◀──────────────┘
└────────────────┬────────────────┘               
                 │                                
┌────────────────┴────────────────┐               
│           LayerNorm             │               
└────────────────┬────────────────┘               
                 │                                
┌────────────────┴────────────────┐               
│         Decoder Embed           │               
│  Linear(1024, 768)              │               
└────────────────┬────────────────┘               
                 │                                
         ┌───────┴───────┐                     
         │               │                     
┌────────┴────────┐ ┌────┴────────────────┐    
│   Decoder       │ │  Contrastive Loss   │    
│ - Mask tokens   │ │  - Global mode      │    
│ - 8 blocks @    │ │  - Temperature=0.2  │    
│   768D          │ │  - HSI vs Aux       │    
│ - Reconstruct   │ │    matching         │    
└────────┬────────┘ └─────────────────────┘
         │
┌────────┴─────────────────────────────────────────┐
│                  OUTPUT LOSSES                   │
│ ┌──────────────┐   ┌──────────────┐             │
│ │   MSE Loss   │   │  Diversity   │             │
│ │  (weight=1.0)│   │    Losses    │             │
│ │              │   │  - Intra-p:  │             │
│ │              │   │    0.2       │             │
│ │              │   │  - Inter-p:  │             │
│ │              │   │    0.2       │             │
│ └──────────────┘   └──────────────┘             │
└──────────────────────────────────────────────────┘
```

## Key Configuration Parameters from spectral_gpt.yaml

- **Input Size**: 224x224 pixels with 30 spectral bands
- **Spatial Patches**: 7x7 patch size → 32x32 patches
- **Temporal Patches**: t_patch_size=5 → 6 temporal patches
- **Total Tokens**: 32×32×6 = 6,144 tokens per HSI
- **Encoder Embed Dim**: 1024
- **Decoder Embed Dim**: 768
- **Encoder**: 10 transformers @ 16 heads each
- **Cross-Attention**: 3 layers when enabled
- **Decoder**: 8 transformers @ 8 heads each
- **Mask Ratio**: 50% of tokens masked
- **Contrastive**: Global mode, temperature=0.2, weight=0
- **Diversity Loss**: Both intra and inter set to 0.2

## Architecture Flow

### With Multimodal Disabled (Current):
1. HSI input → Spatial Registration → Patch Embedding
2. Position Embedding → MAE Masking
3. Transformer Blocks (10 layers @ 1024D)
4. LayerNorm → Decoder Embed (1024→768)
5. Decoder → Only MSE Loss + Diversity Losses output

### With Multimodal Enabled:
1. HSI + Aux inputs → Spatial Registration → Patch Embedding
2. HSI: Position Embedding → MAE Masking
3. Aux: Each through AuxViTEncoder to 256-dim embeddings
4. Transformer Blocks (10 layers @ 1024D)
5. Cross-Attention (3 layers @ 1024D) - HSI features conditioned on aux embeddings
6. LayerNorm → Decoder Embed (1024→768)
7. Decoder → Contrastive Loss computation
8. MSE Loss + Diversity Losses + Contrastive Loss (if weighted)

## Key Design Decisions

1. **Sequential Architecture**: Unlike typical transformers, cross-attention happens AFTER all main encoder blocks rather than being interspersed within them.

2. **Dimension Transition**: The `decoder_embed` layer transitions from 1024-dim encoder features to 768-dim decoder features.

3. **Aux Encoder Design**: Each auxiliary modality gets its own ViT encoder (patch conv → transformer blocks → global pool).

4. **Cross-Attention Scope**: Cross-attention is limited to 3 blocks (separate from the 10 main encoder blocks) to control computational cost.

5. **Contrastive Loss**: Set to weight=0 in current config but architecture supports global-mode contrastive loss between HSI and aux modalities.

6. **Diversity Losses**: Both intra-patch and inter-patch diversity losses are applied to prevent patch collapse (both weighted at 0.2).

7. **Conditional Architecture**: The `use_multimodal` and `use_cross_attention` flags allow the same codebase to run as either pure MAE or multimodal model.