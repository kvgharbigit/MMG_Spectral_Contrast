import torch
import torch.nn as nn
from functools import partial
from util.video_vit import PatchEmbed, Block
from einops import rearrange


class MultiModalSpectralGPT(nn.Module):
    """Multi-Modal Guided SpectralGPT with MAE and Contrastive Learning.

    This model implements a transformer-based architecture for processing hyperspectral imagery (HSI)
    with auxiliary modalities (IR, AF, thickness). It combines three key approaches:
    1. Masked Autoencoder (MAE) for self-supervised learning of HSI data
    2. Soft conditioning with auxiliary modalities via cross-attention
    3. Contrastive learning to align representations across modalities

    The architecture consists of:
    - HSI Encoder: Processes 3D HSI data using patch embedding and transformer blocks
    - Auxiliary Encoders: Process additional modalities (IR, AF, thickness)
    - Cross-Attention: Conditions HSI features on auxiliary information
    - MAE Decoder: Reconstructs masked tokens in embedding space
    - Contrastive Head: Aligns features across different modalities
    """

    def __init__(
            self,
            hsi_img_size=224,  # Input spatial dimensions for HSI (H=W)
            aux_img_size=128,  # Input spatial dimensions for auxiliary modalities
            patch_size=16,  # Spatial patch size for tokenization
            in_chans=1,  # Input channels for HSI (typically 1)
            aux_chans=3,  # Channels for auxiliary modalities (e.g., RGB=3)
            embed_dim=768,  # Main transformer embedding dimension
            depth=16,  # Number of transformer layers i.e. in hsi encoder
            num_heads=12,  # Number of attention heads per transformer
            decoder_embed_dim=512,  # Embedding dimension in decoder
            decoder_depth=4,  # Number of decoder transformer layers
            decoder_num_heads=16,  # Number of attention heads in decoder
            mlp_ratio=4.0,  # Expansion ratio for MLP in transformer blocks
            num_frames=12,  # Number of spectral bands in HSI
            t_patch_size=3,  # Temporal/spectral patch size
            norm_layer=partial(nn.LayerNorm, eps=1e-6),  # Normalization layer
            aux_embed_dim=256,  # Embedding dimension for auxiliary features
            temperature=0.07,  # Temperature for contrastive loss scaling
            mask_ratio=0.75,  # Proportion of tokens to mask in MAE
            aux_encoder_type='vit',  # Type of auxiliary encoder ('cnn' or 'vit')
            **kwargs
    ):
        super().__init__()

        # Store configuration parameters
        self.hsi_img_size = hsi_img_size
        self.aux_img_size = aux_img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.aux_encoder_type = aux_encoder_type

        # Set the expected number of modalities for loss standardization
        self.num_expected_modalities = 3  # ir, af, thickness

        # Initialize HSI encoder with 3D patch embedding
        # This converts the input HSI volume into a sequence of tokens
        self.patch_embed = PatchEmbed(
            img_size=hsi_img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            frames=num_frames,
            t_patch_size=t_patch_size
        )

        # Create learnable position embeddings for each patch
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Create encoders for each auxiliary modality
        self.aux_encoder = nn.ModuleDict({
            'ir': self._make_aux_encoder(aux_chans, aux_embed_dim),
            'af': self._make_aux_encoder(aux_chans, aux_embed_dim),
            'thickness': self._make_aux_encoder(1, aux_embed_dim)
        })

        # Normalization layer for auxiliary features
        self.aux_norm = nn.LayerNorm(aux_embed_dim)

        # Learnable mask token for MAE decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Project auxiliary features to main embedding dimension
        self.modality_proj = nn.Linear(aux_embed_dim, embed_dim)

        # Cross-attention blocks for conditioning on auxiliary features
        self.cross_attn = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Main transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Decoder components
        # Projects encoder features to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        # Position embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim)
        )
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(decoder_depth)
        ])

        # Final normalization layers
        self.norm = norm_layer(embed_dim)
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Decoder prediction head (predicts in embedding space)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim)

        # Contrastive learning components
        # Projects features to space for contrastive learning
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.temperature = temperature

        # Initialize weights
        self.initialize_weights()

    def _make_aux_encoder(self, in_channels, embed_dim):
        """Creates an encoder for auxiliary modalities with separate image size and improved error handling."""
        if self.aux_encoder_type == 'cnn':
            return nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(embed_dim, embed_dim)
            )
        elif self.aux_encoder_type == 'vit':
            return self.AuxViTEncoder(
                img_size=self.aux_img_size,  # Use auxiliary-specific image size
                patch_size=self.patch_size,
                in_chans=in_channels,
                embed_dim=embed_dim
            )
        else:
            raise ValueError(f"Unknown auxiliary encoder type: {self.aux_encoder_type}")

    class AuxViTEncoder(nn.Module):
        """Custom ViT-style encoder for auxiliary modalities with robust 2D patch embedding."""

        def __init__(
                self,
                img_size=128,
                patch_size=8,
                in_chans=3,
                embed_dim=256
        ):
            super().__init__()

            # Validate patch size
            assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"

            # Custom 2D patch embedding for auxiliary modalities
            self.patch_embed = nn.Conv2d(
                in_channels=in_chans,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )

            # Transformer blocks
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)
                ) for _ in range(4)
            ])

            # Final layer norm
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x):
            # Ensure input is 4D (B, C, H, W)
            if x.ndim == 3:
                x = x.unsqueeze(0)  # Add batch dimension if missing

            # Patch embedding
            x = self.patch_embed(x)  # Output: [B, embed_dim, H/patch_size, W/patch_size]

            # Flatten spatial dimensions
            B, C, H, W = x.shape
            x = x.flatten(2)  # Output: [B, embed_dim, H*W]
            x = x.transpose(1, 2)  # Output: [B, H*W, embed_dim]

            # Process through transformer blocks
            for blk in self.blocks:
                x = blk(x)

            # Layer norm
            x = self.norm(x)

            # Global average pooling
            x = x.mean(dim=1)  # Mean across patch tokens

            return x

    def random_masking(self, x, mask_ratio):
        """Perform per-sample random masking by per-sample shuffling."""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        len_keep = max(1, len_keep)  # Keep at least one token

        # Generate random noise and use it to shuffle indices
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        # Generate binary mask (1 is masked, 0 is keep)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, hsi_img, aux_data=None):
        """Forward pass through encoder with masking and auxiliary conditioning."""
        # Convert input to sequence of embedded patches
        x = self.patch_embed(hsi_img)  # Shape: [B, T, HW, D]

        # Reshape from [B, T, HW, D] to [B, T*HW, D]
        B, T, HW, D = x.shape
        x = x.reshape(B, T * HW, D)  # Shape: [B, 1024, D]

        # Add positional embeddings
        x = x + self.pos_embed  # Shape: [B, 1024, D]

        # Apply random masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # Process auxiliary modalities if present
        if aux_data is not None:
            aux_embeddings = self.encode_auxiliary(aux_data)
            # Apply cross-attention for each modality
            for modality, embedding in aux_embeddings.items():
                cond_tokens = self.modality_proj(embedding).unsqueeze(1)
                for block in self.cross_attn:
                    x = x + block(torch.cat([x, cond_tokens], dim=1))[:, :-1, :]

        # Apply main transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Decoder to reconstruct masked tokens in embedding space."""
        # Project to decoder dimension
        x = self.decoder_embed(x)

        # Create and append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0],
            ids_restore.shape[1] - x.shape[1],
            1
        )
        x_ = torch.cat([x, mask_tokens], dim=1)

        # Restore original sequence order
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        # Add position embeddings and apply decoder
        x = x_ + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # Predict original embeddings
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """Compute reconstruction loss in embedding space."""
        # Get target embeddings [B, T, HW, D]
        target = self.patch_embed(imgs)

        # Reshape target from [B, T, HW, D] to [B, T*HW, D]
        B, T, HW, D = target.shape
        target = target.reshape(B, T * HW, D)

        # Compute MSE loss
        loss_recon = (pred - target) ** 2
        loss_recon = loss_recon.mean(dim=-1)

        # Only compute loss on masked tokens
        loss_recon = (loss_recon * mask).sum() / (mask.sum() + 1e-6)

        return loss_recon

    def contrastive_loss(self, hsi_features, aux_embeddings, batch_idx):
        """Calculate contrastive loss with fixed normalization."""
        device = hsi_features.device

        # Project features to contrastive space
        z_hsi = self.proj_head(hsi_features.mean(dim=1))

        # Calculate individual losses for available modalities
        individual_losses = []

        for modality, embeddings in aux_embeddings.items():
            if embeddings is not None:
                # Project embeddings
                embeddings = self.modality_proj(embeddings)
                z_aux = self.proj_head(embeddings)

                # Calculate similarity and loss
                sim_matrix = torch.matmul(z_hsi, z_aux.T) / self.temperature
                labels = batch_idx.to(device)
                loss = nn.CrossEntropyLoss()(sim_matrix, labels)

                individual_losses.append(loss)

        # If no modalities available, return zero loss
        if not individual_losses:
            return torch.tensor(0.0, device=device)

        # Average the available losses
        avg_loss = torch.stack(individual_losses).mean()

        # Scale to expected total (to maintain consistent magnitude)
        # This ensures total loss doesn't decrease when modalities are missing
        num_available_modalities = len(individual_losses)
        scaling_factor = self.num_expected_modalities / num_available_modalities
        scaled_loss = avg_loss * scaling_factor

        return scaled_loss

    def forward(self, hsi_img, aux_data=None, batch_idx=None):
        """Forward pass through the full model with standardized contrastive loss."""
        # Move auxiliary data to correct device
        device = hsi_img.device
        if aux_data is not None:
            aux_data = {k: v.to(device) if v is not None else None
                        for k, v in aux_data.items()}

        # Encode with masking
        latent, mask, ids_restore = self.forward_encoder(hsi_img, aux_data)

        # Decode and reconstruct
        pred = self.forward_decoder(latent, ids_restore)

        # Calculate reconstruction loss
        loss_recon = self.forward_loss(hsi_img, pred, mask)

        # Calculate contrastive loss if auxiliary data present
        loss_contrast = torch.tensor(0.0, device=device)
        num_available = 0
        if aux_data is not None and batch_idx is not None:
            # Count available modalities for logging
            num_available = sum(1 for v in aux_data.values() if v is not None)

            # Only compute contrastive loss if at least one modality is available
            if num_available > 0:
                aux_embeddings = self.encode_auxiliary(aux_data)
                loss_contrast = self.contrastive_loss(latent, aux_embeddings, batch_idx)

        # Calculate total loss
        loss = loss_recon + loss_contrast

        return {
            'loss': loss,
            'loss_recon': loss_recon,
            'loss_contrast': loss_contrast,
            'num_modalities': torch.tensor(num_available, device=device),
            'pred': pred,
            'mask': mask
        }

    def encode_auxiliary(self, aux_data):
        """Encode auxiliary modalities.

        Processes each auxiliary modality through its respective encoder
        and applies normalization.

        Args:
            aux_data: Dictionary of auxiliary images

        Returns:
            dict: Encoded and normalized auxiliary embeddings
        """
        aux_embeddings = {}
        for modality, data in aux_data.items():
            if data is not None:
                aux_embeddings[modality] = self.aux_encoder[modality](data)
                aux_embeddings[modality] = self.aux_norm(aux_embeddings[modality])
        return aux_embeddings

    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings with correct size
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)

        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Print shapes for verification
        print(f"Position embedding shape: {self.pos_embed.shape}")
        print(f"Decoder position embedding shape: {self.decoder_pos_embed.shape}")
        print(f"Mask token shape: {self.mask_token.shape}")