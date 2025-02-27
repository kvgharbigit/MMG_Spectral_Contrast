# **Model Architecture: Multi-Modal Guided SpectralGPT with Contrastive Learning**

This model builds on a **Masked Autoencoder (MAE)** framework with **modality-guided encoding** rather than hard fusion of auxiliary images, and introduces **contrastive learning** to align auxiliary image embeddings with HSI representations.

## **Overview of the Pipeline**

1. **Spatial Registration**:
   - Uniform spatial dimensions across all modalities using a common `analysis_dim`
   - Preprocessing module ensures consistent spatial dimensions before model processing
   - Preserves spectral/temporal information in HSI data

2. **HSI Encoder** (SpectralGPT-style):
   - Uses a Vision Transformer (ViT)-like spatial-spectral encoder
   - Processes **3D spatial-spectral patches** from the **HSI image**
   - Divides 224×224×12 HSI volumes into 14×14×4 patches (total 784 tokens)

3. **Auxiliary Encoders**:
   - Each auxiliary image (IR, AF, Thickness) is **separately tokenized** using either ViT or CNN encoders
   - **Global averaging** applied to create modality-specific global vectors
   - These global vectors enable **robustness to misregistration** between modalities

4. **Cross-Attention Conditioning**:
   - Auxiliary embeddings do not go into the main encoder directly but act as **conditioning inputs**
   - HSI tokens are conditioned via **cross-attention layers** with auxiliary global vectors
   - Each HSI patch can attend to global auxiliary features regardless of spatial position
   - Ensures HSI remains the **primary representation** while still benefiting from auxiliary guidance

5. **Main Transformer Processing**:
   - Conditioned tokens pass through primary transformer blocks for deep feature extraction
   - Maintains the pure HSI representation while leveraging auxiliary information

6. **Dual Learning Objectives**:
   - **MAE Reconstruction**: Masked HSI tokens (75%) are reconstructed from visible tokens (25%)
   - **Contrastive Learning**: Global HSI embeddings are aligned with auxiliary modality embeddings

## **Detailed Architecture**

### **1️⃣ Input Processing & Spatial Registration**

- HSI Input: `[B, C, T, H, W]` → Resized to `[B, C, T, 224, 224]`
- Auxiliary Inputs: Various sizes → All resized to `[B, C, 224, 224]`
- Consistent spatial dimensions for all modalities ensure proper alignment

### **2️⃣ Tokenization & Embedding**

- **HSI Tokenization**:
  - 3D patch embedding with 16×16 spatial and 3-band spectral patches
  - Results in 784 tokens per HSI volume (14×14×4 grid)
  - Output shape: `[B, 784, 768]`

- **Auxiliary Tokenization**:
  - Each modality processed through dedicated ViT/CNN encoder
  - Critical: Global averaging produces a single feature vector per modality
  - Output shape: `[B, 256]` per auxiliary modality

### **3️⃣ Cross-Attention Conditioning**

- HSI tokens `[B, n_visible, 768]` and auxiliary global vectors `[B, 1, 768]` are concatenated
- Self-attention allows information flow between HSI patches and global auxiliary features
- Only the updated HSI tokens are retained after attention
- Residual connections preserve the original HSI signal
- This approach provides fine-grained conditioning while maintaining robustness to misregistration

### **4️⃣ Main Transformer Processing**

- Conditioned HSI tokens are processed through 16 transformer blocks
- Self-attention and MLP layers extract rich spatial-spectral representations
- Final output: `[B, n_visible, 768]`

### **5️⃣ Decoding & Reconstruction**

- Visible tokens are projected to decoder dimension
- Mask tokens are appended for reconstruction
- Decoder reconstructs original token embeddings in embedding space
- L2 loss applied only on masked tokens (75% of total)

### **6️⃣ Contrastive Learning**

- Global averaging across HSI tokens: `[B, 784, 768]` → `[B, 768]`
- Computing similarity matrices between HSI and each auxiliary modality
- Cross-entropy loss encourages same-patient matches across modalities
- Temperature scaling (0.07) sharpens similarity distributions

## **🔬 Loss Functions**

1. **Reconstruction Loss**: L2 loss on masked HSI tokens
   ```
   loss_recon = ((pred - target)²).mean(dim=-1)
   loss_recon = (loss_recon * mask).sum() / mask.sum()
   ```

2. **Contrastive Loss**: Alignment between global representations
   ```
   # For each modality
   sim_matrix = torch.matmul(z_hsi, z_aux.T) / temperature
   loss = CrossEntropyLoss()(sim_matrix, batch_indices)
   # Average across available modalities with scaling
   ```

## **🧩 Handling Missing Modalities**

- If an auxiliary modality is missing, its conditioning step is skipped
- Contrastive loss scales dynamically based on available modalities
- The model learns to operate with any subset of auxiliary modalities
- During inference for downstream tasks, the encoder can operate solely on HSI images

## **🔄 Using the Pretrained HSI Encoder**

For downstream tasks:
1. Extract the core HSI encoder components:
   - Spatial registration, patch embedding, positional embedding
   - Main transformer blocks (excluding cross-attention)
   - Apply fine-tuning to adapt to the distribution shift

2. The standalone HSI encoder processes input without auxiliary modalities:
   ```
   HSI Input → Spatial Registration → Patch Embedding → 
   Positional Embedding → Transformer Blocks → Final Features
   ```

## **🚀 Advantages of This Approach**

✅ **Robustness to Misregistration**: Global auxiliary features enable conditioning without precise spatial alignment

✅ **Retains HSI as the primary feature space**: Cross-attention provides soft guidance without fusion biases

✅ **Dynamic Modality Handling**: Functions with any combination of available auxiliary modalities

✅ **Transferability**: HSI encoder can be extracted and fine-tuned for downstream tasks

✅ **Strong Generalization**: Contrastive learning and MAE pretraining combine to create robust representations

✅ **Computational Efficiency**: ~100M parameters, ~12 GFLOPs for forward pass