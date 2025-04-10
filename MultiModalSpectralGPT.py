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
from torch.utils.checkpoint import checkpoint



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

        from data_utils import derive_mask_from_hsi
        thickness_mask = derive_mask_from_hsi(hsi_registered)
        print("Using HSI-derived mask for rim detection instead of thickness mask")

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
            analysis_dim=500,
            patch_size=(25, 25),
            t_patch_size=5,
            in_chans=1,
            embed_dim=768,
            depth=16,
            num_heads=12,
            decoder_embed_dim=512,
            decoder_depth=4,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            num_frames=30,
            aux_chans=1,
            aux_embed_dim=256,
            temperature=0.07,
            mask_ratio=0.75,
            contrastive_mode='global',
            use_thickness_mask=False,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
    ):
        super().__init__()

        # Convert patch_size to tuple if it isn't already
        patch_size = to_2tuple(patch_size)

        # Dynamically calculate pixel output dimension
        pixel_output_dim = (
                patch_size[0] *  # Spatial patch width
                patch_size[1] *  # Spatial patch height
                t_patch_size *  # Temporal patch size
                in_chans  # Input channels
        )

        # Store configuration parameters
        self.analysis_dim = analysis_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.t_patch_size = t_patch_size
        self.contrastive_mode = contrastive_mode
        self.use_thickness_mask = use_thickness_mask

        # Derived parameters for patch organization
        self.patches_per_dim = analysis_dim // patch_size[0]
        self.spatial_patches = self.patches_per_dim * self.patches_per_dim
        self.spectral_patches = num_frames // t_patch_size

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

        # New: Dynamically created projection layer to pixel output dimension
        self.decoder_pred = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),  # Optional intermediate layer
            nn.GELU(),  # Optional activation
            nn.Linear(decoder_embed_dim, pixel_output_dim)  # Final projection to pixel dimension
        )

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

        # Initialize weights
        self.initialize_weights()

        # Add detailed print statements
        print(f"\nPatch Configuration:")
        print(f"  Spatial Patch Size: {patch_size}")
        print(f"  Temporal Patch Size: {t_patch_size}")
        print(f"  Pixel Output Dimension: {pixel_output_dim}")

        # Calculate and print derived parameters
        spatial_patches_h = self.analysis_dim // self.patch_size[0]
        spatial_patches_w = self.analysis_dim // self.patch_size[1]
        spectral_patches = num_frames // self.t_patch_size

        print(f"  Spatial Patches: {spatial_patches_h} x {spatial_patches_w}")
        print(f"  Spectral Patches: {spectral_patches}")
        print(f"  Total Patches: {spatial_patches_h * spatial_patches_w * spectral_patches}")

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

        # Safety check to prevent extreme masking
        if mask_ratio > 0.99:
            mask_ratio = 0.99

        # Calculate the number of tokens to keep
        len_keep = int(L * (1 - mask_ratio))
        len_keep = max(1, len_keep)  # Ensure at least one token is kept

        # Generate random noise for shuffling
        noise = torch.rand(N, L, device=x.device)

        # Sort noise to get shuffling indices
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
        # Restore to original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Add some debugging print statements
        print(f"Total tokens: {L}")
        print(f"Tokens to keep: {len_keep}")
        print(f"Tokens masked: {L - len_keep}")
        print(f"Masking percentage: {(L - len_keep) / L * 100:.2f}%")

        return x_masked, mask, ids_restore

    def forward_encoder(self, hsi_img, aux_data=None):
        """Forward pass through encoder with masking and auxiliary conditioning."""
        with torch.cuda.amp.autocast(enabled=True):
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
                    if embedding is not None:
                        cond_tokens = self.modality_proj(embedding).unsqueeze(1)
                        for block in self.cross_attn:
                            # Use gradient checkpointing for cross-attention blocks
                            def create_custom_forward(mod):
                                def custom_forward(*inputs):
                                    return mod(torch.cat(inputs, dim=1))[:, :-1, :]

                                return custom_forward

                            if self.training:
                                # Use checkpointing during training
                                x = x + checkpoint(
                                    create_custom_forward(block),
                                    x, cond_tokens
                                )
                            else:
                                # Regular forward pass during evaluation
                                x = x + block(torch.cat([x, cond_tokens], dim=1))[:, :-1, :]

            # Apply main transformer blocks with gradient checkpointing
            for i, block in enumerate(self.blocks):
                if self.training:
                    # Use gradient checkpointing during training
                    x = checkpoint(block, x)
                else:
                    # Regular forward pass during evaluation
                    x = block(x)

            x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Decoder to reconstruct masked tokens in embedding space."""
        with torch.cuda.amp.autocast(enabled=True):
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

            # Add position embeddings and apply decoder with gradient checkpointing
            x = x_ + self.decoder_pos_embed

            for block in self.decoder_blocks:
                if self.training:
                    # Use gradient checkpointing during training
                    x = checkpoint(block, x)
                else:
                    # Regular forward pass during evaluation
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

    def encode_auxiliary(self, aux_data):
        """Encode auxiliary modalities.

        Processes each auxiliary modality through its respective encoder
        and applies normalization.

        Args:
            aux_data: Dictionary of auxiliary images

        Returns:
            dict: Encoded and normalized auxiliary embeddings
        """
        with torch.cuda.amp.autocast(enabled=True):
            aux_embeddings = {}
            for modality, data in aux_data.items():
                if data is not None:
                    aux_embeddings[modality] = self.aux_encoder[modality](data)
                    aux_embeddings[modality] = self.aux_norm(aux_embeddings[modality])
            return aux_embeddings

    def contrastive_loss_spatial(self, hsi_features, aux_embeddings, batch_idx, thickness_mask=None):
        """
        Calculate contrastive loss at the spatial patch level, selectively applying thickness mask.

        Only applies thickness mask when processing the 'thickness' modality if use_thickness_mask is True.
        For other modalities (IR, AF), the thickness mask is not applied.
        """
        with torch.cuda.amp.autocast(enabled=True):
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
        """Calculate contrastive loss with global representations, selectively applying thickness mask."""
        with torch.cuda.amp.autocast(enabled=True):
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

    def unpatchify(self, patches, original_shape):
        """
        Convert patch tokens directly to pixel space.

        Args:
            patches: Tensor of shape [B, L, patch_pixels] or [B, S, P, patch_pixels]
            original_shape: Tuple of original input shape [B, C, T, H, W]

        Returns:
            Tensor of shape [B, C, T, H, W] representing reconstructed pixels
        """
        B, C, T, H, W = original_shape
        patch_h, patch_w = self.patch_size
        t_patch = self.t_patch_size

        # Reshape if needed
        if len(patches.shape) == 3:  # [B, L, patch_pixels]
            patches = patches.reshape(B, self.spectral_patches, self.spatial_patches, -1)

        # Calculate spatial dimensions
        spatial_h = H // patch_h
        spatial_w = W // patch_w

        # Reshape to get individual patch dimensions
        patches = patches.reshape(B, self.spectral_patches, spatial_h, spatial_w, t_patch, patch_h, patch_w, C)

        # Permute dimensions to match original image layout
        patches = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()

        # Final reshape to image space
        reconstructed = patches.reshape(B, C, self.spectral_patches * t_patch, H, W)

        # Handle any dimension mismatch (e.g., if spectral dimensions don't match exactly)
        if reconstructed.shape[2] != T:
            if reconstructed.shape[2] > T:
                reconstructed = reconstructed[:, :, :T, :, :]
            else:
                # Pad if needed
                padding = torch.zeros(B, C, T - reconstructed.shape[2], H, W, device=reconstructed.device)
                reconstructed = torch.cat([reconstructed, padding], dim=2)

        return reconstructed

    def token_mask_to_pixel_mask(self, token_mask, original_shape):
        """
        Convert a token-level mask to a 3D pixel-level mask [B, T, H, W].

        Args:
            token_mask: Token-level mask of shape [B, L]
            original_shape: Original input shape [B, C, T, H, W]

        Returns:
            3D pixel mask of shape [B, T, H, W]
        """
        B, C, T, H, W = original_shape

        # Calculate patch dimensions
        spatial_patches_h = H // self.patch_size[0]
        spatial_patches_w = W // self.patch_size[1]
        spectral_patches = T // self.t_patch_size

        total_patches = spectral_patches * spatial_patches_h * spatial_patches_w

        # Validate mask shape
        if token_mask.shape[1] != total_patches:
            raise ValueError(
                f"Mask size incorrect. Expected {total_patches} patches, "
                f"but got {token_mask.shape[1]} patches. "
                f"Spatial patches: {spatial_patches_h}x{spatial_patches_w}, "
                f"Spectral patches: {spectral_patches}"
            )

        # Create 3D pixel mask
        pixel_mask = torch.zeros(B, T, H, W, device=token_mask.device)

        for b in range(B):
            # Reshape mask to [spectral_patches, spatial_patches_h, spatial_patches_w]
            mask_reshaped = token_mask[b].reshape(spectral_patches, spatial_patches_h, spatial_patches_w)

            for t_idx in range(spectral_patches):
                for h_idx in range(spatial_patches_h):
                    for w_idx in range(spatial_patches_w):
                        # Calculate pixel coordinates for this patch
                        h_start = h_idx * self.patch_size[0]
                        w_start = w_idx * self.patch_size[1]

                        # Is this patch masked?
                        if mask_reshaped[t_idx, h_idx, w_idx] > 0.5:
                            # Mark entire spectral patch as masked
                            pixel_mask[b,
                            t_idx * self.t_patch_size:(t_idx + 1) * self.t_patch_size,
                            h_start:h_start + self.patch_size[0],
                            w_start:w_start + self.patch_size[1]
                            ] = 1.0

        return pixel_mask

    def forward_loss_in_pixel_space(self, pred_pixels, original_input, mask, thickness_mask=None):
        """
        Compute loss in pixel space by directly comparing with the original input.
        """
        with torch.cuda.amp.autocast(enabled=True):
            # Convert token mask to 3D pixel mask
            pixel_mask = self.token_mask_to_pixel_mask(mask, original_input.shape)  # [B, T, H, W]
            print(f"Pixel mask shape: {pixel_mask.shape}")

            # If thickness mask is provided and enabled, combine masks
            if thickness_mask is not None and self.use_thickness_mask:
                # Print details about the thickness mask
                print(f"Original thickness mask shape: {thickness_mask.shape}")
                print(f"Thickness mask dimensions: {thickness_mask.dim()}")

                # Examine the thickness mask shape in detail
                if thickness_mask.dim() == 5:
                    print(
                        f"Shape details: [B={thickness_mask.shape[0]}, D1={thickness_mask.shape[1]}, D2={thickness_mask.shape[2]}, H={thickness_mask.shape[3]}, W={thickness_mask.shape[4]}]")

                    # Try to reshape the thickness mask to [B, 1, H, W]
                    try:
                        # Remove the extra dimension (D2) directly
                        reshaped_thickness = thickness_mask[:, :, 0, :, :]
                        print(f"After removing D2: {reshaped_thickness.shape}")

                        # Now expand across wavelength dimension
                        expanded_thickness = reshaped_thickness.expand(-1, pixel_mask.shape[1], -1, -1)
                        print(f"After expansion: {expanded_thickness.shape}")

                        # Combine masks
                        combined_mask = pixel_mask * expanded_thickness

                        # Compute loss
                        pixel_mse = ((pred_pixels - original_input) ** 2) * combined_mask
                        valid_pixel_count = combined_mask.sum() + 1e-6
                        return pixel_mse.sum() / valid_pixel_count

                    except Exception as e:
                        print(f"Error reshaping thickness mask: {e}")
                        # Fall back to just using pixel mask
                        pixel_mse = ((pred_pixels - original_input) ** 2) * pixel_mask
                        masked_pixel_count = pixel_mask.sum() + 1e-6
                        return pixel_mse.sum() / masked_pixel_count

                elif thickness_mask.dim() == 4:
                    try:
                        # Expand across wavelength dimension
                        expanded_thickness = thickness_mask.expand(-1, pixel_mask.shape[1], -1, -1)
                        print(f"After expansion: {expanded_thickness.shape}")

                        # Combine masks
                        combined_mask = pixel_mask * expanded_thickness

                        # Compute loss
                        pixel_mse = ((pred_pixels - original_input) ** 2) * combined_mask
                        valid_pixel_count = combined_mask.sum() + 1e-6
                        return pixel_mse.sum() / valid_pixel_count

                    except Exception as e:
                        print(f"Error expanding thickness mask: {e}")
                        # Fall back to just using pixel mask
                        pixel_mse = ((pred_pixels - original_input) ** 2) * pixel_mask
                        masked_pixel_count = pixel_mask.sum() + 1e-6
                        return pixel_mse.sum() / masked_pixel_count
            else:
                # No thickness mask, just use MAE mask
                pixel_mse = ((pred_pixels - original_input) ** 2) * pixel_mask

                # Sum loss and normalize by the number of masked pixels
                masked_pixel_count = pixel_mask.sum() + 1e-6
                return pixel_mse.sum() / masked_pixel_count

    def forward(self, hsi_img, aux_data=None, batch_idx=None):
        """Forward pass through the full model with direct pixel prediction."""
        # Apply spatial registration
        hsi_img, aux_data, thickness_mask = self.spatial_registration(hsi_img, aux_data)

        # Store original input for pixel-level loss calculation
        original_input = hsi_img.clone()

        # If thickness mask should not be used, set it to None
        if not self.use_thickness_mask:
            thickness_mask = None

        # Setup patch embedding
        self._setup_patch_embedding(hsi_img)

        # Create unmasked embeddings for contrastive learning
        unmasked_features = None
        if aux_data is not None and batch_idx is not None:
            with torch.cuda.amp.autocast(enabled=True):
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

        # Encode with masking for reconstruction
        with torch.cuda.amp.autocast(enabled=True):
            latent, mask, ids_restore = self.forward_encoder(hsi_img, aux_data)

            # Print shape of latent embeddings
            print(f"Latent embeddings shape: {latent.shape}")

            # Decode and reconstruct
            pred_tokens = self.forward_decoder(latent, ids_restore)

            # Print shape of predicted tokens
            print(f"Predicted tokens shape: {pred_tokens.shape}")

            # Reshape for unpatchify (from flat token sequence to organized patches)
            pred_tokens_reshaped = pred_tokens.reshape(
                B, self.spectral_patches, self.spatial_patches, -1
            )

            # Unpatchify to pixel space directly
            reconstructed_pixels = self.unpatchify(pred_tokens_reshaped, original_input.shape)

            # Calculate reconstruction loss directly in pixel space
            loss_recon = self.forward_loss_in_pixel_space(
                reconstructed_pixels,
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
                with torch.cuda.amp.autocast(enabled=True):
                    if self.contrastive_mode == 'global':
                        aux_embeddings = self.encode_auxiliary(aux_data)
                        loss_contrast = self.contrastive_loss_global(unmasked_features, aux_embeddings, batch_idx,
                                                                     thickness_mask)
                    else:
                        aux_patch_embeddings = self.encode_auxiliary_patches(aux_data)
                        loss_contrast = self.contrastive_loss_spatial(unmasked_features, aux_patch_embeddings,
                                                                      batch_idx,
                                                                      thickness_mask)

        # Calculate total loss
        loss = loss_recon + loss_contrast

        return {
            'loss': loss,
            'loss_recon': loss_recon,
            'loss_contrast': loss_contrast,
            'num_modalities': torch.tensor(num_available, device=device),
            'pred': pred_tokens,
            'mask': mask,
            'thickness_mask': thickness_mask,
            'contrastive_mode': self.contrastive_mode,
            'original_input': original_input,
            'reconstructed_pixels': reconstructed_pixels  # Add reconstructed pixels to output
        }