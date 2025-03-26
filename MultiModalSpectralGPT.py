import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from util.video_vit import PatchEmbed, Block
from timm.layers import to_2tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        threshold = 0.005

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

        # Print detailed debugging information
        print(f"Band Selection Debug:")
        print(f"  Input shape: {hsi_img.shape}")
        print(f"  Total bands: {T}")
        print(f"  Target bands: {self.target_bands}")

        # Check if already has the target number of bands
        if T == self.target_bands:
            print("  No band selection needed - already correct number of bands")
            self.selected_indices = list(range(T))
            return hsi_img

        # Determine band selection strategy
        if T <= self.target_bands:
            # If fewer bands than target, use all
            selected_indices = list(range(T))
        else:
            # Create evenly spaced indices
            step = max(1, (T - 1) // (self.target_bands - 1))
            selected_indices = list(range(0, T, step))[:self.target_bands]

        # Ensure we have exactly the target number of bands
        if len(selected_indices) > self.target_bands:
            selected_indices = selected_indices[:self.target_bands]
        elif len(selected_indices) < self.target_bands:
            # Pad with the last index if needed
            selected_indices.extend([selected_indices[-1]] * (self.target_bands - len(selected_indices)))

        # Print selected indices
        print(f"  Selected indices: {selected_indices}")
        print(f"  Number of selected indices: {len(selected_indices)}")

        # Store selected indices for reference
        self.selected_indices = selected_indices

        # Select the specified bands
        try:
            # Use torch.index_select for more flexible indexing
            index_tensor = torch.tensor(selected_indices, device=hsi_img.device, dtype=torch.long)

            # Perform indexing with detailed error handling
            try:
                selected_bands = torch.index_select(hsi_img, 2, index_tensor)
            except Exception as select_error:
                print(f"Error during torch.index_select: {select_error}")
                print(f"Input tensor shape: {hsi_img.shape}")
                print(f"Index tensor: {index_tensor}")
                raise

            # Print shape after selection
            print(f"  Output shape: {selected_bands.shape}")

            return selected_bands
        except Exception as e:
            print(f"Comprehensive error selecting bands: {e}")
            import traceback
            traceback.print_exc()
            raise

    def forward(self, hsi_img, aux_data):
        """
        Preprocesses HSI image by:
        1. Checking if format is already correct (30 bands, 500x500)
        2. Selecting specific spectral bands if needed
        3. Resizing spatial dimensions if needed
        4. Creating mask from thickness modality
        """
        B, C, T, H, W = hsi_img.shape

        # First check if HSI is already in the correct format
        if T == self.target_bands and H == self.analysis_dim and W == self.analysis_dim:
            print("HSI already in correct format, skipping preprocessing.")
            # Still need to set selected_indices for reference
            self.selected_indices = list(range(T))
            hsi_registered = hsi_img
        else:
            # First, select specified spectral bands if needed
            if T != self.target_bands:
                hsi_img = self.select_spectral_bands(hsi_img)
                # Get updated dimensions after band selection
                B, C, T, H, W = hsi_img.shape

            # Resize HSI if needed
            if H == self.analysis_dim and W == self.analysis_dim:
                hsi_registered = hsi_img
            else:
                hsi_reshaped = hsi_img.view(B * T, C, H, W)
                hsi_resized = F.interpolate(
                    hsi_reshaped,
                    size=(self.analysis_dim, self.analysis_dim),
                    mode='bilinear',
                    align_corners=False
                )
                hsi_registered = hsi_resized.view(B, C, T, self.analysis_dim, self.analysis_dim)

        # Process auxiliary modalities (resize only)
        aux_registered = {}
        thickness_mask = None

        for modality, data in aux_data.items():
            if data is not None:
                if data.shape[2] == self.analysis_dim and data.shape[3] == self.analysis_dim:
                    aux_registered[modality] = data
                else:
                    aux_registered[modality] = F.interpolate(
                        data,
                        size=(self.analysis_dim, self.analysis_dim),
                        mode='bilinear',
                        align_corners=False
                    )
            else:
                aux_registered[modality] = None

        # Create mask from thickness image without applying it
        if 'thickness' in aux_registered and aux_registered['thickness'] is not None:
            thickness_mask = self.detect_mask(aux_registered['thickness'])

        return hsi_registered, aux_registered, thickness_mask


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
            use_thickness_mask=False,  # Whether to use thickness mask for rim exclusion
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
        self.use_thickness_mask = use_thickness_mask  # Whether to use thickness mask

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

        # Create patch embedding during initialization
        self.patch_embed = PatchEmbed(
            img_size=analysis_dim,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            frames=num_frames,
            t_patch_size=t_patch_size
        )

        # Set number of patches
        self.num_patches = self.patch_embed.num_patches

        # Initialize positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Create projection layer from embedding to pixel space
        patch_pixels = patch_size[0] * patch_size[1] * t_patch_size * in_chans
        self.pixel_projection = nn.Linear(embed_dim, patch_pixels)
        # Initialize weights
        nn.init.xavier_uniform_(self.pixel_projection.weight)
        nn.init.zeros_(self.pixel_projection.bias)

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
        Ensure patch embedding is configured properly for the current input.
        Now just moves the embedding to the right device.
        """
        # Get device from input tensor
        device = hsi_img.device

        # Move patch embedding to the same device as input
        if self.patch_embed.proj.weight.device != device:
            self.patch_embed = self.patch_embed.to(device)
            self.pos_embed = self.pos_embed.to(device)

    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings
        torch.nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(self, x, mask_ratio):
        """Perform per-sample random masking by per-sample shuffling."""
        N, L, D = x.shape

        # Add this safety check
        if mask_ratio > 0.99:
            mask_ratio = 0.99

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

    def contrastive_loss_spatial(self, hsi_features, aux_embeddings, batch_idx, thickness_mask=None):
        """
        Calculate contrastive loss at the spatial patch level, selectively applying thickness mask.

        Only applies thickness mask when processing the 'thickness' modality if use_thickness_mask is True.
        For other modalities (IR, AF), the thickness mask is not applied.
        """
        device = hsi_features.device
        B = hsi_features.shape[0]

        # Group HSI patches by spatial location
        hsi_spatial = self.group_hsi_patches_by_spatial_location(hsi_features)

        # Project with spatial-specific head
        z_hsi = self.proj_head_spatial(hsi_spatial)  # [B, HW, T*D] -> [B, HW, D]

        # Convert thickness mask to patch level if provided
        patch_thickness_mask = None
        if thickness_mask is not None:
            patch_thickness_mask = self.create_patch_mask_from_pixel_mask(thickness_mask)
            # Reshape to match spatial structure
            patch_thickness_mask = patch_thickness_mask.reshape(B, self.spatial_patches * self.spectral_patches)
            # Group by spatial location (take average of all spectral patches at each location)
            patch_thickness_mask = patch_thickness_mask.reshape(B, self.spatial_patches, self.spectral_patches)
            patch_thickness_mask = patch_thickness_mask.mean(dim=2)  # [B, spatial_patches]

        # Calculate individual losses for available modalities
        individual_losses = []

        for modality, embeddings in aux_embeddings.items():
            if embeddings is not None:
                # Project auxiliary embeddings
                projected_aux = self.aux_spatial_proj(embeddings)
                z_aux = self.proj_head_global(projected_aux)

                # Selectively apply mask only for thickness modality
                if modality == 'thickness' and patch_thickness_mask is not None:
                    # Collect valid patches across all batches
                    valid_hsi_features = []
                    valid_aux_features = []
                    valid_batch_indices = []

                    for b in range(B):
                        # Get indices of valid patches (where mask > 0.5)
                        valid_indices = torch.where(patch_thickness_mask[b] > 0.5)[0]

                        if len(valid_indices) == 0:
                            continue  # Skip if no valid patches in this batch

                        # Extract valid features
                        valid_hsi_features.append(z_hsi[b, valid_indices])
                        valid_aux_features.append(z_aux[b, valid_indices])

                        # Keep track of which batch each patch came from
                        valid_batch_indices.extend([b] * len(valid_indices))

                    # Skip modality if no valid patches
                    if len(valid_hsi_features) == 0:
                        continue

                    # Stack features from all batches
                    valid_hsi = torch.cat(valid_hsi_features, dim=0)
                    valid_aux = torch.cat(valid_aux_features, dim=0)
                    valid_batch_tensor = torch.tensor(valid_batch_indices, device=device)

                    # Calculate similarity matrix for valid patches only
                    sim_matrix = torch.matmul(valid_hsi, valid_aux.T) / self.temperature

                    # For each patch, the matching patch is at the same index
                    # (since we've concatenated them in the same order)
                    labels = torch.arange(valid_hsi.shape[0], device=device)

                    # Calculate loss on valid patches only
                    loss = nn.CrossEntropyLoss()(sim_matrix, labels)
                else:
                    # For non-thickness modalities, use all patches without masking
                    z_hsi_flat = z_hsi.reshape(B * self.spatial_patches, -1)
                    z_aux_flat = z_aux.reshape(B * self.spatial_patches, -1)

                    sim_matrix = torch.matmul(z_hsi_flat, z_aux_flat.T) / self.temperature
                    labels = torch.arange(B * self.spatial_patches, device=device)
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

    def contrastive_loss_global(self, hsi_features, aux_embeddings, batch_idx, thickness_mask=None):
        """Calculate contrastive loss with global representations, selectively applying thickness mask.

        Only applies thickness mask when processing the 'thickness' modality if use_thickness_mask is True.
        For other modalities (IR, AF), the thickness mask is not applied.
        """
        device = hsi_features.device
        B = hsi_features.shape[0]

        # Skip contrastive loss for batch size of 1
        if B <= 1:
            return torch.tensor(0.0, device=device)

        # Create new batch indices relative to this batch (0 to B-1)
        # This replaces the absolute indices with relative ones
        relative_batch_idx = torch.arange(B, device=device)

        # Get global HSI representation (unmasked for all modalities except thickness)
        global_hsi_unmasked = hsi_features.mean(dim=1)

        # Calculate global HSI representation with mask (only for thickness modality)
        global_hsi_masked = None
        if thickness_mask is not None:
            # Convert pixel-level mask to patch-level mask
            patch_mask = self.create_patch_mask_from_pixel_mask(thickness_mask)

            # Apply mask to features (multiply by mask and then average)
            # Expand mask to match feature dimensions
            expanded_mask = patch_mask.unsqueeze(-1)  # [B, num_patches, 1]

            # Apply mask
            masked_features = hsi_features * expanded_mask

            # Sum valid mask values for proper normalization
            mask_sum = patch_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6

            # Get weighted average (sum of masked features / sum of mask)
            global_hsi_masked = (masked_features.sum(dim=1) / mask_sum.squeeze(-1))

        # Calculate individual losses for available modalities
        individual_losses = []

        for modality, embeddings in aux_embeddings.items():
            if embeddings is not None:
                # Project embeddings
                embeddings = self.modality_proj(embeddings)
                z_aux = self.proj_head_global(embeddings)

                # Selectively use masked HSI features only for thickness modality
                if modality == 'thickness' and global_hsi_masked is not None:
                    z_hsi = self.proj_head_global(global_hsi_masked)
                else:
                    z_hsi = self.proj_head_global(global_hsi_unmasked)

                # Calculate similarity and loss
                sim_matrix = torch.matmul(z_hsi, z_aux.T) / self.temperature

                # Use relative indices for loss calculation
                loss = nn.CrossEntropyLoss()(sim_matrix, relative_batch_idx)
                individual_losses.append(loss)

        # Handle case with no modalities
        if not individual_losses:
            return torch.tensor(0.0, device=device)

        # Average and scale losses
        avg_loss = torch.stack(individual_losses).mean()
        scaling_factor = self.num_expected_modalities / len(individual_losses)
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



    def forward_loss_in_embedding_space(self, pred, target, mask, thickness_mask=None):
        """
        Compute loss in embedding space by directly comparing predicted embeddings
        to target embeddings from the patches.

        Args:
            pred (torch.Tensor): Predicted embeddings from decoder of shape [B, L, D]
            target (torch.Tensor): Original embeddings from encoder of shape [B, L, D]
            mask (torch.Tensor): Binary mask indicating which tokens were masked (1=masked, 0=visible)
            thickness_mask (torch.Tensor, optional): Patch-level mask indicating valid image regions

        Returns:
            torch.Tensor: Reconstruction loss in embedding space
        """
        # We only compute loss on masked tokens
        B, L, D = pred.shape

        # Apply the MAE mask (we only compute loss on masked regions)
        # Expand mask for broadcasting
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)  # [B, L, D]

        # If thickness mask is provided, convert it to patch level and apply it
        if thickness_mask is not None and self.use_thickness_mask:
            # Convert pixel-level mask to patch-level
            patch_mask = self.create_patch_mask_from_pixel_mask(thickness_mask)  # [B, num_patches]

            # Expand patch mask for broadcasting
            patch_mask_expanded = patch_mask.unsqueeze(-1).expand_as(pred)  # [B, L, D]

            # Combine MAE mask with thickness mask (both must be 1 for a token to be used)
            # For MAE mask, 1 = masked token, for thickness mask, 1 = valid region
            combined_mask = mask_expanded * patch_mask_expanded

            # Compute squared error only on tokens that are both masked and in valid regions
            loss = ((pred - target) ** 2) * combined_mask

            # Sum loss and normalize by the number of tokens that contribute
            valid_token_count = combined_mask.sum() + 1e-6  # Add small epsilon to avoid division by zero
            return loss.sum() / valid_token_count
        else:
            # No thickness mask, just use MAE mask
            loss = ((pred - target) ** 2) * mask_expanded

            # Sum loss and normalize by the number of masked tokens
            masked_token_count = mask.sum() * D + 1e-6  # Multiply by embedding dim and add epsilon
            return loss.sum() / masked_token_count

    def create_patch_mask_from_pixel_mask(self, pixel_mask):
        """
        Converts a pixel-level mask to patch-level mask for token-based processing.

        Args:
            pixel_mask (torch.Tensor): Pixel-level mask of shape [B, 1, H, W]

        Returns:
            torch.Tensor: Patch-level mask of shape [B, num_patches]
        """
        B = pixel_mask.shape[0]
        device = pixel_mask.device

        # Calculate the spatial dimensions after patching
        spatial_patches_h = self.analysis_dim // self.patch_size[0]
        spatial_patches_w = self.analysis_dim // self.patch_size[1]

        # Use average pooling to convert pixel-level mask to patch-level
        # This approach determines if a patch is mostly valid pixels
        patch_masks = []

        for b in range(B):
            # Get the current batch's mask
            current_mask = pixel_mask[b, 0].unsqueeze(0)  # Shape: [1, H, W]

            # Use average pooling to get one value per patch
            pooled_mask = F.avg_pool2d(
                current_mask,
                kernel_size=(self.patch_size[0], self.patch_size[1]),
                stride=(self.patch_size[0], self.patch_size[1])
            )  # Shape: [1, spatial_patches_h, spatial_patches_w]

            # Flatten spatial dimensions
            pooled_mask_flat = pooled_mask.view(1, -1)  # Shape: [1, spatial_patches_h * spatial_patches_w]

            # Expand across temporal dimension
            # Each spatial location has spectral_patches tokens
            expanded_mask = pooled_mask_flat.unsqueeze(2).expand(
                1, self.spatial_patches, self.spectral_patches
            ).reshape(1, -1)  # Shape: [1, spatial_patches * spectral_patches]

            patch_masks.append(expanded_mask)

        # Concatenate all batch masks
        patch_mask = torch.cat(patch_masks, dim=0)  # Shape: [B, num_patches]

        # Apply threshold - a patch is valid if at least 30% of pixels are valid
        patch_mask = (patch_mask > 0.3).float()

        return patch_mask

    def unpatchify(self, embeddings, original_shape):
        """
        Convert patch embeddings back to pixel space.

        Args:
            embeddings: Tensor of shape [B, L, D] or [B, spectral_patches, spatial_patches, D]
            original_shape: Tuple representing the original input shape [B, C, T, H, W]

        Returns:
            Tensor of shape [B, C, T, H, W] representing reconstructed pixels
        """
        B, C, T, H, W = original_shape

        # Get patch dimensions
        patch_h, patch_w = self.patch_size
        t_patch = self.t_patch_size

        # Reshape if needed
        if len(embeddings.shape) == 3:  # [B, L, D]
            B, L, D = embeddings.shape
            embeddings = embeddings.reshape(B, self.spectral_patches, self.spatial_patches, D)

        # Ensure the pixel projection is on the same device as the embeddings
        if self.pixel_projection.weight.device != embeddings.device:
            self.pixel_projection = self.pixel_projection.to(embeddings.device)

        # Project from embedding space to patch space
        B, S, P, D = embeddings.shape
        embeddings_flat = embeddings.reshape(B * S * P, D)
        patches_flat = self.pixel_projection(embeddings_flat)
        patches = patches_flat.reshape(B, S, P, patch_h * patch_w * t_patch * C)

        # Reshape to get the proper patch dimensions
        spatial_h = H // patch_h
        spatial_w = W // patch_w
        patches = patches.reshape(B, S, spatial_h, spatial_w, t_patch, patch_h, patch_w, C)

        # Rearrange dimensions to prepare for merging patches
        patches = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        patches = patches.reshape(B, C, S * t_patch, H, W)

        # Ensure spectral dimension matches the original
        if S * t_patch != T:
            if S * t_patch > T:
                patches = patches[:, :, :T, :, :]
            else:
                padding = torch.zeros(B, C, T - S * t_patch, H, W, device=patches.device)
                patches = torch.cat([patches, padding], dim=2)

        return patches

    def token_mask_to_pixel_mask(self, token_mask, original_shape):
        """
        Convert a token-level mask to a pixel-level mask.

        Args:
            token_mask: Binary mask of shape [B, L] where 1 indicates masked tokens
            original_shape: Tuple of the original input shape [B, C, T, H, W]

        Returns:
            Binary mask of shape [B, C, T, H, W] for applying at pixel level
        """
        B, C, T, H, W = original_shape

        # Reshape token mask to [B, spectral_patches, spatial_patches]
        token_mask_reshaped = token_mask.reshape(B, self.spectral_patches, self.spatial_patches)

        # Get patch dimensions
        patch_h, patch_w = self.patch_size
        t_patch = self.t_patch_size

        # Initialize pixel-level mask
        pixel_mask = torch.zeros(B, C, T, H, W, device=token_mask.device)

        # Calculate spatial patches in each dimension
        spatial_h = H // patch_h
        spatial_w = W // patch_w

        # Iterate through each token and set corresponding pixel region
        for b in range(B):
            for s in range(self.spectral_patches):
                t_start = s * t_patch
                t_end = min((s + 1) * t_patch, T)

                for h in range(spatial_h):
                    for w in range(spatial_w):
                        # Calculate patch index
                        p = h * spatial_w + w

                        if token_mask_reshaped[b, s, p] > 0.5:  # If token is masked
                            # Calculate pixel coordinates
                            h_start = h * patch_h
                            h_end = min((h + 1) * patch_h, H)
                            w_start = w * patch_w
                            w_end = min((w + 1) * patch_w, W)

                            # Set corresponding pixels to 1 (masked)
                            pixel_mask[b, :, t_start:t_end, h_start:h_end, w_start:w_end] = 1.0

        return pixel_mask

    def forward_loss_in_pixel_space(self, pred_embeddings, target_embeddings, original_input, mask,
                                    thickness_mask=None):
        """
        Compute loss in pixel space by decoding the embeddings back to pixels and
        comparing with the original input.

        Args:
            pred_embeddings: Predicted embeddings from decoder of shape [B, L, D]
            target_embeddings: Original embeddings from encoder of shape [B, L, D]
            original_input: Original input images of shape [B, C, T, H, W]
            mask: Binary mask indicating which tokens were masked (1=masked, 0=visible)
            thickness_mask: Mask indicating valid image regions (optional)

        Returns:
            torch.Tensor: Reconstruction loss in pixel space
        """
        # Convert embeddings back to pixel space
        reconstructed_pixels = self.unpatchify(pred_embeddings, original_input.shape)

        # Convert token mask to pixel mask
        pixel_mask = self.token_mask_to_pixel_mask(mask, original_input.shape)

        # If thickness mask is provided, combine it with the MAE mask
        if thickness_mask is not None and self.use_thickness_mask:
            # Ensure thickness mask has the same dimensions as pixel_mask
            if thickness_mask.ndim == 4:  # [B, 1, H, W]
                # Expand to match [B, C, T, H, W]
                thickness_mask = thickness_mask.unsqueeze(2).expand_as(pixel_mask)

            # Combine masks - both must be 1 for a pixel to be used
            combined_mask = pixel_mask * thickness_mask

            # Compute MSE only on pixels that are both masked and in valid regions
            pixel_mse = ((reconstructed_pixels - original_input) ** 2) * combined_mask

            # Sum loss and normalize by the number of pixels that contribute
            valid_pixel_count = combined_mask.sum() + 1e-6  # Add small epsilon to avoid division by zero
            return pixel_mse.sum() / valid_pixel_count
        else:
            # No thickness mask, just use MAE mask
            pixel_mse = ((reconstructed_pixels - original_input) ** 2) * pixel_mask

            # Sum loss and normalize by the number of masked pixels
            masked_pixel_count = pixel_mask.sum() + 1e-6
            return pixel_mse.sum() / masked_pixel_count

    def forward(self, hsi_img, aux_data=None, batch_idx=None):
        """Forward pass through the full model with configurable contrastive loss mode."""
        # Apply spatial registration to ensure all modalities have the same dimensions
        hsi_img, aux_data, thickness_mask = self.spatial_registration(hsi_img, aux_data)

        # Store original input for pixel-level loss calculation
        original_input = hsi_img.clone()

        # If thickness mask should not be used, set it to None
        if not self.use_thickness_mask:
            thickness_mask = None

        # Dynamically set up patch embedding based on preprocessed image
        self._setup_patch_embedding(hsi_img)

        # Create unmasked embeddings for contrastive learning
        unmasked_features = None
        if aux_data is not None and batch_idx is not None:
            unmasked_features = self.patch_embed(hsi_img)
            B, T, HW, D = unmasked_features.shape
            unmasked_features = unmasked_features.reshape(B, T * HW, D)
            unmasked_features = unmasked_features + self.pos_embed

        # Move auxiliary data to correct device
        device = hsi_img.device
        if aux_data is not None:
            aux_data = {k: v.to(device) if v is not None else None
                        for k, v in aux_data.items()}

        # Move thickness mask to device if it exists
        if thickness_mask is not None:
            thickness_mask = thickness_mask.to(device)

        # Store original tokens for target in loss calculation
        # First, get original patch tokens (without masking)
        original_tokens = self.patch_embed(hsi_img)
        B, T, HW, D = original_tokens.shape
        original_tokens = original_tokens.reshape(B, T * HW, D)
        original_tokens = original_tokens + self.pos_embed

        # Encode with masking for reconstruction
        latent, mask, ids_restore = self.forward_encoder(hsi_img, aux_data)

        # Decode and reconstruct
        pred = self.forward_decoder(latent, ids_restore)

        # Calculate reconstruction loss in pixel space instead of embedding space
        loss_recon = self.forward_loss_in_pixel_space(
            pred,
            original_tokens,
            original_input,
            mask,
            thickness_mask
        )

        # Calculate contrastive loss if auxiliary data present
        loss_contrast = torch.tensor(0.0, device=device)
        num_available = 0
        if aux_data is not None and batch_idx is not None and unmasked_features is not None:
            # Count available modalities for logging
            num_available = sum(1 for v in aux_data.values() if v is not None)

            # Only compute contrastive loss if at least one modality is available
            if num_available > 0:
                if self.contrastive_mode == 'global':
                    aux_embeddings = self.encode_auxiliary(aux_data)
                    # Pass thickness_mask to global contrastive loss
                    loss_contrast = self.contrastive_loss_global(unmasked_features, aux_embeddings, batch_idx,
                                                                 thickness_mask)
                else:
                    aux_patch_embeddings = self.encode_auxiliary_patches(aux_data)
                    loss_contrast = self.contrastive_loss_spatial(unmasked_features, aux_patch_embeddings, batch_idx,
                                                                  thickness_mask)

        # Calculate total loss
        loss = loss_recon + loss_contrast

        return {
            'loss': loss,
            'loss_recon': loss_recon,
            'loss_contrast': loss_contrast,
            'num_modalities': torch.tensor(num_available, device=device),
            'pred': pred,
            'mask': mask,
            'thickness_mask': thickness_mask,
            'contrastive_mode': self.contrastive_mode,
            'original_tokens': original_tokens,  # Include original tokens for visualization
            'original_input': original_input  # Include original input for pixel-level visualization
        }