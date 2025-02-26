# **Model Architecture: Multi-Modal Guided SpectralGPT with Contrastive Learning**

This model builds on a **Masked Autoencoder (MAE)** framework with **modality-guided encoding** rather than hard fusion of auxiliary images, and introduces **contrastive learning** to align auxiliary image embeddings with HSI representations.

## **Overview of the Pipeline**

1. **HSI Encoder** (SpectralGPT-style):
    - Uses a Vision Transformer (ViT)-like spatial-spectral encoder.
    - Processes **3D spatial-spectral patches** from the **HSI image**.
    - Learns high-level spectral-spatial representations.
2. **Modality Conditioners ( Cross-Attention)**
    - Auxiliary images (IR, AF, Thickness) are embedded separately into a **modality-specific token space**.
    - These embeddings **do not go into the encoder directly** but instead act as **conditioning inputs** via cross-attention layers
    - Ensures HSI remains the **primary representation** while still benefiting from auxiliary guidance.
3. **Decoding & Pretraining Objectives**
    - **Main Task: MAE Reconstruction of HSI** (masked tokens are reconstructed from latent embeddings).
    - **Contrastive Learning: Auxiliary Image Alignment**
        - Instead of reconstructing auxiliary images, we use contrastive learning to enforce **embedding alignment** between the HSI latent space and the auxiliary modality embeddings.

## **Detailed Architecture**

### **1️⃣ Input Processing**

### **(a) HSI Input - Spectral Tokenization**

- Divide HSI image into non **overlapping 3D patches** (spatial-spectral cubes).
- Feed into a **Transformer-based encoder**.

### **(b) Auxiliary Images - Modality-Specific Tokens**

- Each auxiliary image (IR, AF, Thickness) is **separately tokenized**.
- Images are either 1 channel (grayscale) or 3 channel (rgb)
- Instead of encoding them fully, extract **global embeddings** using ViT blocks.
- Pass them as **learned conditioning tokens** (not direct fusion).
- These tokens are injected via **cross-attention layers** during HSI encoding.

---

### **2️⃣ SpectralGPT Encoder (Spatial-Spectral Transformer)**

- A **ViT-style transformer with spectral attention** processes HSI patches.
- **Masked Autoencoding:** Randomly mask a portion of the patches during training.
- **Cross-Attention (Optional):** If auxiliary data is present, attention heads query modality tokens.
- Outputs a set of **latent representations**.

---

### **3️⃣ Decoding & Contrastive Learning**

- **Masked Token Reconstruction (HSI)**
    - A ViT decoder reconstructs missing HSI patches from the learned latent representations.
- **Contrastive Learning for Multi-Modal Alignment**
    - Instead of reconstructing auxiliary images, we enforce **embedding-level alignment** using contrastive learning.
    - The HSI latent embeddings and auxiliary modality embeddings should be **closer for the same patient** and **distant for different patients**.
    - This encourages the HSI encoder to extract **features that correlate well** with auxiliary modalities without forcing direct reconstruction.

---

## **🔬 Loss Functions**

1. **Main Loss:** **MAE Loss (L2) on Masked HSI Tokens**
2. **Contrastive Loss for Modality Alignment:**
    - **Sim()** is a similarity function (e.g., cosine similarity) that maximizes intra-patient similarity and minimizes inter-patient similarity.
    - values can be tuned to balance contrastive learning with the MAE objective.

---

## **🧩 Handling Missing Modalities**

- If an auxiliary modality is missing, simply **skip its token embedding**.
- The model learns to **not rely on any single modality** for HSI encoding.
- During inference, the encoder operates **solely on HSI images** (pure spatial-spectral generalization).

---

## **🚀 Advantages of This Approach**

✅ **Retains HSI as the primary feature space** → No fusion biases.

✅ **Auxiliary images act as soft guidance** → They improve pretraining but don’t degrade spectral learning.

✅ **Handles missing modalities dynamically** → Works even with partial auxiliary data.

✅ **Uses MAE-style training for robust pretraining** → Captures robust spectral-spatial dependencies.

✅ **Contrastive Learning improves feature generalization** → Ensures the HSI encoder aligns with auxiliary modalities **without overfitting to them**.

---
