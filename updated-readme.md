# **Model Architecture: Multi-Modal Guided SpectralGPT with Dual-Mode Contrastive Learning and Rim Masking**

This model builds on a **Masked Autoencoder (MAE)** framework with **modality-guided encoding** rather than hard fusion of auxiliary images, and introduces **dual-mode contrastive learning** to align auxiliary image embeddings with HSI representations at either patient-level or spatial patch-level. It also incorporates **rim masking** to exclude non-informative black border regions from loss calculations.

## **Key Features**

- **Channel Configuration**: All input modalities (HSI and auxiliary) use single-channel (grayscale) inputs
- **ViT-style Encoders**: All auxiliary modalities use Vision Transformer encoders for consistent processing
- **Robust Spatial Registration**: Enhanced preprocessing to handle varied input sizes
- **Dynamic Patch Embedding**: Patch embedding adapts to input dimensions dynamically
- **Dual-Mode Contrastive Learning**: Choice between global (patient-level) and spatial (patch-level) contrastive learning
- **Rim Masking**: Automatically excludes black rim areas from loss calculations in both reconstruction and contrastive learning

## **Overview of the Pipeline**

1. **Spatial Registration**:
   - Uniform spatial dimensions across all modalities using a common `analysis_dim`
   - Preprocessing module ensures consistent spatial dimensions before model processing
   - Preserves spectral/temporal information in HSI data
   - Selects specific spectral bands from the HSI image
   - Resizes inputs to a consistent 500x500 spatial dimension
   - Creates thickness mask to identify valid regions vs. black rim areas

2. **HSI Encoder** (SpectralGPT-style):
   - Uses a Vision Transformer (ViT)-like spatial-spectral encoder
   - Processes **3D spatial-spectral patches** from the **HSI image**
   - Dynamically divides HSI volumes into patches based on input dimensions
   - Supports variable input sizes with adaptive patch embedding

3. **Auxiliary Encoders**:
   - ViT-style Encoder: Transformer-based patch embedding and processing
   - Each auxiliary image (IR, AF, Thickness) is **separately tokenized**
   - Creates both global and patch-level representations
   - Enables **robustness to misregistration** between modalities

4. **Cross-Attention Conditioning**:
   - Auxiliary embeddings condition HSI tokens via cross-attention layers
   - Each HSI patch can attend to global auxiliary features
   - Ensures HSI remains the **primary representation** while benefiting from auxiliary guidance

5. **Dual Learning Paths**:
   - **MAE Reconstruction Path**: Uses masked tokens (75%) for reconstruction task
   - **Contrastive Learning Path**: Uses unmasked tokens for modality alignment
   - **Both paths incorporate rim masking** to focus on meaningful regions

6. **Dual-Mode Contrastive Learning with Rim Masking**:
   - **Global Mode**: Aligns patient-level representations across modalities
     - Weighted average pooling (excluding rim areas)
     - Patient-level similarity comparison
   - **Spatial Mode**: Aligns patch-level representations across modalities
     - Groups HSI patches by spatial location
     - Filters out patches from rim areas
     - Patch-by-patch similarity comparison using only valid patches

## **Rim Masking Mechanism**

### **Mask Creation and Propagation**

- **Thickness-Based Detection**: The model automatically detects black rim areas using the thickness modality
- **Pixel to Patch Conversion**: Converts pixel-level mask to patch-level using average pooling
- **Thresholding**: A patch is considered valid if at least 30% of its pixels are valid
- **Dynamic Application**: The mask is applied differently based on contrastive mode

### **Application in Loss Calculations**

1. **Reconstruction Loss**:
   - Only computed on valid (non-rim) regions
   - Combined with MAE mask (loss only on masked tokens in valid regions)
   - Normalized by count of tokens that are both masked and in valid regions

2. **Global Contrastive Loss**:
   - Features are masked before computing global representation
   - Weighted average pooling with proper normalization
   - Prevents black rim from "poisoning" the global representation

3. **Spatial Contrastive Loss**:
   - Only valid patches are collected for similarity calculation
   - Maintains batch information for proper contrastive learning
   - Creates cleaner spatial correspondences between modalities

## **Detailed Dual-Mode Contrastive Learning**

### **1️⃣ Global Contrastive Learning (with Rim Masking)**

- **Feature Pooling**:
  - Masked weighted averaging of HSI features (excluding rim areas)
  - Global auxiliary features from ViT encoder
  
- **Projection**:
  - Both HSI and auxiliary features project to contrastive space using `proj_head_global`
  
- **Similarity Computation**:
  - Computes cross-modality similarity between patients
  - Labels correspond to batch indices (diagonal matches)
  
- **Advantages**:
  - More robust to spatial misregistration
  - Purer signal without rim area "poisoning"
  - Clean global representations for downstream tasks

### **2️⃣ Spatial Contrastive Learning (with Rim Masking)**

- **HSI Feature Organization**:
  - Reorganizes HSI patches to group by spatial location
  - Concatenates spectral features for each spatial location
  - Filters out patches from black rim areas
  
- **Projection Pipeline**:
  - HSI spatial features: Project with specialized `proj_head_spatial` to handle spectral concatenation
  - Auxiliary patch features: First project to `embed_dim`, then use `proj_head_global`
  
- **Similarity Computation**:
  - Compares only valid patches at corresponding spatial locations
  - Labels identify spatial correspondence (same location across modalities)
  
- **Advantages**:
  - Preserves spatial correspondence between modalities
  - Focuses on meaningful regions only
  - Creates cleaner spatial-spectral representations

## **🧩 Implementation Details**

- **Patch Organization**: 
  - HSI: 6 spectral chunks × 400 spatial patches = 2,400 total patches
  - Each spatial location contains 6 spectral patches that are grouped together
  
- **Mask Transformation**:
  - Pixel-level: [B, 1, 500, 500] → Patch-level: [B, 2400]
  - For spatial mode: Reshape to [B, 400] (average across spectral dimension)
  
- **Projection Heads**:
  - `proj_head_global`: Standard projection for embedding dimension (768→768)
  - `proj_head_spatial`: Handles concatenated spectral features (4608→768)
  
- **Patch Processing**:
  - Spatial dimensions: 500×500 → 20×20 patches (25×25 each)
  - Spectral dimensions: 30 bands → 6 chunks (5 bands each)

## **🚀 Advantages of the Full Model**

✅ **Adaptive Input Handling**: Works with varied input sizes and modalities
✅ **Configurable Contrastive Learning**: Choose between global and spatial alignment
✅ **Robust Spatial Registration**: Consistent preprocessing across modalities
✅ **Unmasked Contrastive Learning**: Uses complete representations for alignment
✅ **Modality Agnostic**: Functions with any combination of available auxiliary modalities
✅ **Focused Learning**: Rim masking ensures model focuses on meaningful data regions
✅ **Improved Signal Quality**: Prevents black rim areas from diluting feature representations
✅ **Resource Efficiency**: Computational resources target informative regions

## **🔬 Technical Specifications**

- **Parameters**: ~100M parameters
- **Computational Complexity**: ~12 GFLOPs for forward pass
- **Input Dimension**: 500x500 spatial, variable spectral/temporal dimensions
- **Encoder Types**: Transformer-based with cross-attention conditioning
- **Loss Functions**: 
  - Reconstruction Loss (L2) with rim masking
  - Contrastive Alignment Loss (Global or Spatial mode) with rim masking

## **🛠 Configuration Options**

- `analysis_dim`: Target spatial dimension (default: 500)
- `patch_size`: Spatial patch size for tokenization
- `contrastive_mode`: Type of contrastive learning ('global' or 'spatial')
- `mask_ratio`: Proportion of tokens to mask (default: 0.75)
- `temperature`: Scaling factor for contrastive loss (default: 0.07)

## **🔄 Model Usage**

For downstream tasks:
1. Extract core HSI encoder components
2. Fine-tune to adapt to specific distribution shifts
3. Operate with or without auxiliary modalities
4. Choose the contrastive mode that best suits your application needs
5. Benefit from automatic rim masking to focus on meaningful regions

This model provides a flexible, robust framework for multi-modal hyperspectral image processing with self-supervised learning capabilities, configurable contrastive learning approaches, and automatic exclusion of non-informative border regions.