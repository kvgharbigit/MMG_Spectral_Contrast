# **Model Architecture: Multi-Modal Guided SpectralGPT with Contrastive Learning**

This model builds on a **Masked Autoencoder (MAE)** framework with **modality-guided encoding** rather than hard fusion of auxiliary images, and introduces **contrastive learning** to align auxiliary image embeddings with HSI representations.

## **Key Updates**

- **Channel Configuration**: All input modalities (HSI and auxiliary) now use single-channel (grayscale) inputs
- **Flexible Encoder Types**: Auxiliary modalities can now use either CNN or ViT-style encoders
- **Robust Spatial Registration**: Enhanced preprocessing to handle varied input sizes
- **Dynamic Patch Embedding**: Patch embedding now adapts to input dimensions dynamically

## **Overview of the Pipeline**

1. **Spatial Registration**:
   - Uniform spatial dimensions across all modalities using a common `analysis_dim`
   - Preprocessing module ensures consistent spatial dimensions before model processing
   - Preserves spectral/temporal information in HSI data
   - Selects specific spectral bands from the HSI image
   - Resizes inputs to a consistent 500x500 spatial dimension

2. **HSI Encoder** (SpectralGPT-style):
   - Uses a Vision Transformer (ViT)-like spatial-spectral encoder
   - Processes **3D spatial-spectral patches** from the **HSI image**
   - Dynamically divides HSI volumes into patches based on input dimensions
   - Supports variable input sizes with adaptive patch embedding

3. **Auxiliary Encoders**:
   - Supports two encoder types for auxiliary modalities:
     a. CNN Encoder: Convolutional layers with global average pooling
     b. ViT-style Encoder: Transformer-based patch embedding and processing
   - Each auxiliary image (IR, AF, Thickness) is **separately tokenized**
   - **Global averaging** applied to create modality-specific global vectors
   - Enables **robustness to misregistration** between modalities

4. **Cross-Attention Conditioning**:
   - Auxiliary embeddings condition HSI tokens via cross-attention layers
   - Each HSI patch can attend to global auxiliary features
   - Ensures HSI remains the **primary representation** while benefiting from auxiliary guidance

5. **Main Transformer Processing**:
   - Conditioned tokens pass through primary transformer blocks
   - Maintains the pure HSI representation while leveraging auxiliary information

6. **Dual Learning Objectives**:
   - **MAE Reconstruction**: Masked HSI tokens (75%) are reconstructed from visible tokens (25%)
   - **Contrastive Learning**: Global HSI embeddings are aligned with auxiliary modality embeddings

## **Detailed Modifications**

### **1️⃣ Input Processing**

- **Channel Configuration**:
  - HSI Input: Now uses single-channel input
  - Auxiliary Inputs: All use single-channel (1 channel) images
  - Supports flexible encoding strategies (CNN or ViT)

- **Spatial Registration**:
  - Fixed target dimension of 500x500 for all modalities
  - Spectral band selection: Selected indices include every 2nd index from 0 to 57, plus wavelength at index 80
  - Bilinear interpolation for resizing spatial dimensions

### **2️⃣ Auxiliary Encoding**

- **Encoder Flexibility**:
  - `aux_encoder_type` parameter allows switching between:
    - 'cnn': Convolutional encoder with pooling
    - 'vit': Vision Transformer-style encoder
  - Robust to different input modalities
  - Consistent global feature extraction

### **3️⃣ Contrastive Learning Enhancements**

- Dynamic loss scaling based on available modalities
- Handles scenarios with missing auxiliary data
- Standardized loss computation across modalities
- Temperature-scaled similarity matrix for robust representation alignment

## **🧩 Handling Missing Modalities**

- Gracefully handles scenarios with partial or missing auxiliary data
- Contrastive loss dynamically adjusts based on available modalities
- Can operate with any subset of auxiliary modalities during training or inference

## **🚀 Advantages**

✅ **Adaptive Input Handling**: Works with varied input sizes and modalities
✅ **Encoder Flexibility**: Choice between CNN and ViT-style auxiliary encoders
✅ **Robust Spatial Registration**: Consistent preprocessing across modalities
✅ **Dynamic Patch Embedding**: Adapts to input dimensions
✅ **Modality Agnostic**: Functions with any combination of available auxiliary modalities

## **🔬 Technical Specifications**

- **Parameters**: ~100M parameters
- **Computational Complexity**: ~12 GFLOPs for forward pass
- **Input Dimension**: 500x500 spatial, variable spectral/temporal dimensions
- **Encoder Types**: Transformer-based with cross-attention conditioning
- **Loss Functions**: 
  - Reconstruction Loss (L2)
  - Contrastive Alignment Loss

## **🛠 Configuration Options**

- `analysis_dim`: Target spatial dimension (default: 500)
- `patch_size`: Spatial patch size for tokenization
- `aux_encoder_type`: Encoder type for auxiliary modalities ('cnn' or 'vit')
- `mask_ratio`: Proportion of tokens to mask (default: 0.75)
- `temperature`: Scaling factor for contrastive loss (default: 0.07)

## **🔄 Model Usage**

For downstream tasks:
1. Extract core HSI encoder components
2. Fine-tune to adapt to specific distribution shifts
3. Operate with or without auxiliary modalities

Provides a flexible, robust framework for multi-modal hyperspectral image processing with self-supervised learning capabilities.