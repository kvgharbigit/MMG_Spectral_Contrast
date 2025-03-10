import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from util.video_vit import PatchEmbed, Block
from timm.models.layers import to_2tuple

from einops import rearrange


class SpatialRegistration(nn.Module):
    """
    Preprocessing module to ensure all modalities are spatially registered
    and spectral bands are selected as specified.
    """

    def __init__(self, analysis_dim=500, target_bands=30):
        """
        Initialize the spatial registration module.

        Args:
            analysis_dim (int): The target spatial dimension for all modalities.
            target_bands (int): The target number of spectral bands.
        """
        super().__init__()
        self.analysis_dim = analysis_dim
        self.target_bands = target_bands
        self.selected_indices = None  # To track selected indices for use in other methods

    def detect_mask(self, image):
        """Detect black mask in an image tensor.

        Args:
            image (torch.Tensor): Image tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Binary mask (1 for valid regions, 0 for masked regions)
        """
        # Assuming black regions have pixel values near zero
        threshold = 0.05

        # Create binary mask (1 for valid pixels, 0 for masked/black pixels)
        if image.shape[1] > 1:
            mask = (image.mean(dim=1, keepdim=True) > threshold).float()
        else:
            mask = (image > threshold).float()

        return mask

    def select_spectral_bands(self, hsi_img):
        """
        Select specific spectral bands from the HSI image.

        Args:
            hsi_img (torch.Tensor): HSI image tensor of shape [B, C, T, H, W]

        Returns:
            torch.Tensor: Processed HSI image with selected bands
        """
        # Original spectral dimensions
        B, C, T, H, W = hsi_img.shape

        # Check if already has the target number of bands
        if T == self.target_bands:
            # No need to select bands, return as is
            self.selected_indices = list(range(T))
            return hsi_img

        # Create indices for band selection
        # 0:58:2 → Every 2nd index from 0 to 57
        # 80 → Add wavelength at index 80
        selected_indices = list(range(0, 58, 2)) + [80]

        # Sort indices to maintain order
        selected_indices = sorted(set(selected_indices))

        # Store selected indices for reference
        self.selected_indices = selected_indices

        # Select the specified bands
        selected_bands = hsi_img[:, :, selected_indices, :, :]

        return selected_bands

    def forward(self, hsi_img, aux_data):
        """
        Preprocesses HSI image by:
        1. Selecting specific spectral bands
        2. Resizing spatial dimensions
        3. Applying consistent masking from thickness modality to all modalities

        Args:
            hsi_img (torch.Tensor): HSI image tensor of shape [B, C, T, H, W]
            aux_data (dict): Dictionary of auxiliary modalities

        Returns:
            tuple: (spatially_registered_hsi, spatially_registered_aux_data)
        """
        # First, select specified spectral bands
        hsi_img = self.select_spectral_bands(hsi_img)

        # Get updated dimensions after band selection
        B, C, T, H, W = hsi_img.shape

        # Check if spatial dimensions already match the target dimension
        if H == self.analysis_dim and W == self.analysis_dim:
            # If already 500x500, no need to resize
            hsi_registered = hsi_img
        else:
            # Reshape for resizing if spatial dimensions are not 500x500
            # First reshape to [B*T, C, H, W] for batch-compatible resizing
            hsi_reshaped = hsi_img.view(B * T, C, H, W)

            # Resize spatial dimensions
            hsi_resized = F.interpolate(
                hsi_reshaped,
                size=(self.analysis_dim, self.analysis_dim),
                mode='bilinear',
                align_corners=False
            )

            # Reshape back to [B, C, T, analysis_dim, analysis_dim]
            hsi_registered = hsi_resized.view(B, C, T, self.analysis_dim, self.analysis_dim)

        # Process auxiliary modalities (resize first)
        aux_registered = {}
        for modality, data in aux_data.items():
            if data is not None:
                # Check if auxiliary data already has the target dimensions
                if data.shape[2] == self.analysis_dim and data.shape[3] == self.analysis_dim:
                    aux_registered[modality] = data
                else:
                    # Resize to target dimension
                    aux_registered[modality] = F.interpolate(
                        data,
                        size=(self.analysis_dim, self.analysis_dim),
                        mode='bilinear',
                        align_corners=False
                    )
            else:
                aux_registered[modality] = None

            # Detect mask from thickness image if available
            thickness_mask = None
            if 'thickness' in aux_registered and aux_registered['thickness'] is not None:
                thickness_mask = self.detect_mask(aux_registered['thickness'])

                # Apply mask to HSI (across all spectral bands)
                for t in range(T):
                    # Use torch.where instead of multiplication
                    hsi_registered[:, :, t] = torch.where(
                        thickness_mask > 0.03,  # Condition: where mask is above threshold
                        hsi_registered[:, :, t],  # True: keep original values
                        torch.zeros_like(hsi_registered[:, :, t])  # False: set to zero
                    )

                # Apply mask to all auxiliary modalities
                for modality in aux_registered:
                    if aux_registered[modality] is not None:
                        # Use torch.where instead of multiplication
                        aux_registered[modality] = torch.where(
                            thickness_mask > 0.05,  # Condition: where mask is above threshold
                            aux_registered[modality],  # True: keep original values
                            torch.zeros_like(aux_registered[modality])  # False: set to zero
                        )

        return hsi_registered, aux_registered


class MultiModalSpectralGPT(nn.Module):
    """Multi-Modal Guided SpectralGPT with MAE and Contrastive Learning.

    This model implements a transformer-based architecture for processing hyperspectral imagery (HSI)
    with auxiliary modalities (IR, AF, thickness). It combines three key approaches:
    1. Masked Autoencoder (MAE) for self-supervised learning of HSI data
    2. Soft conditioning with auxiliary modalities via cross-attention
    3. Contrastive learning to align representations across modalities
    """

    def __init__(
            self,
            analysis_dim=500,  # Common spatial dimension for all modalities
            patch_size=(25, 25),  # Spatial patch size for tokenization
            in_chans=1,  # Input channels for HSI (typically 1)
            aux_chans=1,  # Channels for auxiliary modalities (all now 1 channel/grayscale)
            embed_dim=768,  # Main transformer embedding dimension
            depth=16,  # Number of transformer layers i.e. in hsi encoder
            num_heads=12,  # Number of attention heads per transformer
            decoder_embed_dim=512,  # Embedding dimension in decoder
            decoder_depth=4,  # Number of decoder transformer layers
            decoder_num_heads=16,  # Number of attention heads in decoder
            mlp_ratio=4.0,  # Expansion ratio for MLP in transformer blocks
            num_frames=30,  # Number of frames/spectral bands
            t_patch_size=5,  # Temporal/spectral patch size
            norm_layer=partial(nn.LayerNorm, eps=1e-6),  # Normalization layer
            aux_embed_dim=256,  # Embedding dimension for auxiliary features
            temperature=0.07,  # Temperature for contrastive loss scaling
            mask_ratio=0.75,  # Proportion of tokens to mask in MAE
            contrastive_mode='global',  # Type of contrastive learning ('global' or 'spatial')
            **kwargs
    ):
        super().__init__()

        # Convert patch_size to tuple if it isn't already
        patch_size = to_2tuple(patch_size)

        # Store configuration parameters
        self.analysis_dim = analysis_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.t_patch_size = t_patch_size
        self.contrastive_mode = contrastive_mode  # Parameter for contrastive learning mode

        # Derived parameters for patch organization
        self.patches_per_dim = analysis_dim // patch_size[0]  # Number of patches in one spatial dimension
        self.spatial_patches = self.patches_per_dim * self.patches_per_dim  # Total spatial patches
        self.spectral_patches = num_frames // t_patch_size  # Number of spectral/temporal patches

        # Set the expected number of modalities for loss standardization
        self.num_expected_modalities = 3  # ir, af, thickness

        # Add spatial registration module for consistent dimensions
        self.spatial_registration = SpatialRegistration(analysis_dim, num_frames)

        # Create ViT encoders for each auxiliary modality
        self.aux_encoder = nn.ModuleDict({
            'ir': self._make_aux_encoder(aux_chans, aux_embed_dim),
            'af': self._make_aux_encoder(aux_chans, aux_embed_dim),
            'thickness': self._make_aux_encoder(aux_chans, aux_embed_dim)
        })

        # Add normalization layers for auxiliary encoders
        self.aux_norm = nn.LayerNorm(aux_embed_dim)

        # Normalization layer for auxiliary features
        self.modality_proj = nn.Linear(aux_embed_dim, embed_dim)

        # Learnable mask token for MAE decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

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
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
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
        self.temperature = temperature

        # Calculate spectral dim for spatial contrastive learning
        spectral_dim = self.spectral_patches * embed_dim

        # For global contrastive learning mode
        self.proj_head_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # For spatial contrastive learning mode
        self.proj_head_spatial = nn.Sequential(
            nn.Linear(spectral_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Project auxiliary patch embeddings to fixed embedding dim
        self.aux_spatial_proj = nn.Linear(aux_embed_dim, embed_dim)

        # Patch embedding will be dynamically set in forward method
        self.patch_embed = None
        self.pos_embed = None
        self.num_patches = None

        # Initialize weights
        self.initialize_weights()

    def _make_aux_encoder(self, in_channels, embed_dim):
        """Creates a ViT encoder for auxiliary modalities."""
        return self.AuxViTEncoder(
            img_size=self.analysis_dim,  # Use common analysis dimension
            patch_size=self.patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim
        )

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

            # Convert patch_size to tuple if it isn't already
            patch_size = to_2tuple(patch_size)

            # Validate patch size
            assert img_size % patch_size[
                0] == 0, f"Image size {img_size} must be divisible by patch size {patch_size[0]}"

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

    def _setup_patch_embedding(self, hsi_img):
        """
        Dynamically set up patch embedding based on preprocessed HSI image.

        Args:
            hsi_img (torch.Tensor): Preprocessed HSI image tensor
        """
        # Get dimensions of preprocessed HSI image
        _, _, T, H, W = hsi_img.shape

        # Dynamically create patch embedding
        self.patch_embed = PatchEmbed(
            img_size=H,  # Use actual height after preprocessing
            patch_size=self.patch_size,
            in_chans=1,  # Assuming single channel after preprocessing
            embed_dim=self.embed_dim,
            frames=T,  # Use actual number of frames/bands
            t_patch_size=self.t_patch_size
        )

        # Dynamically calculate and set number of patches
        self.num_patches = self.patch_embed.num_patches

        # Create learnable position embeddings for each patch
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        # Re-initialize position embeddings
        torch.nn.init.normal_(self.pos_embed, std=0.02)

    def initialize_weights(self):
        """Initialize model weights."""
        # Note: Actual positional embedding initialization is deferred to forward pass
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

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
        x = x.reshape(B, T * HW, D)  # Shape: [B, T*HW, D]

        # Add positional embeddings
        x = x + self.pos_embed  # Shape: [B, T*HW, D]

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

    def group_hsi_patches_by_spatial_location(self, features):
        """
        Reorganizes HSI patch tokens to group them by spatial location.

        Args:
            features (torch.Tensor): HSI patch features of shape [B, T*HW, D]
                where T is the number of spectral chunks, HW is spatial patches

        Returns:
            torch.Tensor: Reorganized features of shape [B, HW, T*D]
                where each row contains all spectral information for a spatial patch
        """
        B, L, D = features.shape

        # Calculate the number of spatial and spectral patches
        spatial_patches = self.spatial_patches
        spectral_patches = self.spectral_patches

        # Reshape into [B, spectral_patches, spatial_patches, D]
        features = features.reshape(B, spectral_patches, spatial_patches, D)

        # Transpose to [B, spatial_patches, spectral_patches, D]
        features = features.transpose(1, 2)

        # Concatenate spectral features for each spatial patch
        features = features.reshape(B, spatial_patches, spectral_patches * D)

        return features

    def encode_auxiliary_patches(self, aux_data):
        """
        Encode auxiliary modalities into patch-level features for spatial contrastive learning.

        Args:
            aux_data: Dictionary of auxiliary images

        Returns:
            dict: Dictionary of encoded auxiliary patch embeddings
        """
        aux_patch_embeddings = {}

        for modality, data in aux_data.items():
            if data is not None:
                # Patch embed using the auxiliary encoder's patch_embed
                x = self.aux_encoder[modality].patch_embed(data)

                # Flatten spatial dimensions
                B, C, H, W = x.shape
                x = x.flatten(2)  # Output: [B, embed_dim, H*W]
                x = x.transpose(1, 2)  # Output: [B, H*W, embed_dim]

                # Process through transformer blocks
                for blk in self.aux_encoder[modality].blocks:
                    x = blk(x)

                # Apply normalization
                x = self.aux_encoder[modality].norm(x)

                # Skip the global pooling step to keep patch tokens
                aux_patch_embeddings[modality] = x

        return aux_patch_embeddings

    def contrastive_loss_spatial(self, hsi_features, aux_embeddings, batch_idx):
        """
        Calculate contrastive loss at the spatial patch level.

        Args:
            hsi_features (torch.Tensor): HSI token features [B, T*HW, D]
            aux_embeddings (dict): Dictionary of auxiliary patch embeddings
            batch_idx (torch.Tensor): Batch indices for cross-batch negatives

        Returns:
            torch.Tensor: Spatial contrastive loss
        """
        device = hsi_features.device
        B = hsi_features.shape[0]

        # Group HSI patches by spatial location
        hsi_spatial = self.group_hsi_patches_by_spatial_location(hsi_features)

        # The grouped shape is [B, HW, T*D]
        # Project with spatial-specific head
        z_hsi = self.proj_head_spatial(hsi_spatial)  # [B, HW, T*D] -> [B, HW, D]

        # Calculate individual losses for available modalities
        individual_losses = []

        for modality, embeddings in aux_embeddings.items():
            if embeddings is not None:
                # First project to embedding dim
                projected_aux = self.aux_spatial_proj(embeddings)

                # Then use global projection head (works with embed_dim)
                z_aux = self.proj_head_global(projected_aux)

                # Calculate patch-wise similarity matrix
                # For each sample in batch and each patch location
                # Shape: [B*HW, B*HW]
                z_hsi_flat = z_hsi.reshape(B * self.spatial_patches, -1)
                z_aux_flat = z_aux.reshape(B * self.spatial_patches, -1)

                sim_matrix = torch.matmul(z_hsi_flat, z_aux_flat.T) / self.temperature

                # Create labels: patches at same spatial location and same batch are positives
                # We need labels of shape [B*HW] where the value indicates the positive match index
                labels = torch.arange(B * self.spatial_patches, device=device)

                # Calculate loss
                loss = nn.CrossEntropyLoss()(sim_matrix, labels)
                individual_losses.append(loss)

        # If no modalities available, return zero loss
        if not individual_losses:
            return torch.tensor(0.0, device=device)

        # Average the available losses and scale
        avg_loss = torch.stack(individual_losses).mean()
        scaling_factor = self.num_expected_modalities / len(individual_losses)
        scaled_loss = avg_loss * scaling_factor

        return scaled_loss

    def contrastive_loss_global(self, hsi_features, aux_embeddings, batch_idx):
        """Calculate contrastive loss with global representations (original method)."""
        device = hsi_features.device

        # Get global HSI representation (mean across all patches)
        global_hsi = hsi_features.mean(dim=1)

        # Project features to contrastive space
        z_hsi = self.proj_head_global(global_hsi)

        # Calculate individual losses for available modalities
        individual_losses = []

        for modality, embeddings in aux_embeddings.items():
            if embeddings is not None:
                # Project embeddings
                embeddings = self.modality_proj(embeddings)
                z_aux = self.proj_head_global(embeddings)

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

    def forward(self, hsi_img, aux_data=None, batch_idx=None):
        """Forward pass through the full model with configurable contrastive loss mode."""
        # Apply spatial registration to ensure all modalities have the same dimensions
        hsi_img, aux_data = self.spatial_registration(hsi_img, aux_data)

        # Dynamically set up patch embedding based on preprocessed image
        self._setup_patch_embedding(hsi_img)

        # Create unmasked embeddings for contrastive learning
        unmasked_features = None
        if aux_data is not None and batch_idx is not None:
            # Get the unmasked patch embeddings for contrastive learning
            unmasked_features = self.patch_embed(hsi_img)  # Shape: [B, T, HW, D]
            B, T, HW, D = unmasked_features.shape
            unmasked_features = unmasked_features.reshape(B, T * HW, D)  # Shape: [B, T*HW, D]
            unmasked_features = unmasked_features + self.pos_embed  # Add positional embeddings

        # Move auxiliary data to correct device
        device = hsi_img.device
        if aux_data is not None:
            aux_data = {k: v.to(device) if v is not None else None
                        for k, v in aux_data.items()}

        # Encode with masking for reconstruction
        latent, mask, ids_restore = self.forward_encoder(hsi_img, aux_data)

        # Decode and reconstruct
        pred = self.forward_decoder(latent, ids_restore)

        # Calculate reconstruction loss
        loss_recon = self.forward_loss(hsi_img, pred, mask)

        # Calculate contrastive loss if auxiliary data present
        loss_contrast = torch.tensor(0.0, device=device)
        num_available = 0
        if aux_data is not None and batch_idx is not None and unmasked_features is not None:
            # Count available modalities for logging
            num_available = sum(1 for v in aux_data.values() if v is not None)

            # Only compute contrastive loss if at least one modality is available
            if num_available > 0:
                # For global mode, use the original approach
                if self.contrastive_mode == 'global':
                    aux_embeddings = self.encode_auxiliary(aux_data)
                    loss_contrast = self.contrastive_loss_global(unmasked_features, aux_embeddings, batch_idx)
                else:
                    # For spatial mode, work with unmasked features
                    aux_patch_embeddings = self.encode_auxiliary_patches(aux_data)
                    loss_contrast = self.contrastive_loss_spatial(unmasked_features, aux_patch_embeddings, batch_idx)

        # Calculate total loss
        loss = loss_recon + loss_contrast

        return {
            'loss': loss,
            'loss_recon': loss_recon,
            'loss_contrast': loss_contrast,
            'num_modalities': torch.tensor(num_available, device=device),
            'pred': pred,
            'mask': mask,
            'contrastive_mode': self.contrastive_mode
        }