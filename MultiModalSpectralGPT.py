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
            # Add these new parameters with defaults
            intra_div_weight=0.1,
            inter_div_weight=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
    ):
        super().__init__()

        # Store the new diversity loss weights as class attributes
        self.intra_div_weight = intra_div_weight
        self.inter_div_weight = inter_div_weight

        # Print information about diversity loss configuration
        print(f"Diversity loss weights: intra={self.intra_div_weight}, inter={self.inter_div_weight}")

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
            nn.LayerNorm(decoder_embed_dim),
            nn.Linear(decoder_embed_dim, pixel_output_dim)
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
        """Initialize model weights properly to prevent gradient issues."""
        print("Initializing model with robust weight initialization...")

        # Initialize embedding projections with normal distribution
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # Initialize decoder position embeddings
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

        # Initialize main transformer blocks
        for block in self.blocks:
            if hasattr(block.attn, 'qkv'):
                # For fused QKV attention
                nn.init.xavier_uniform_(block.attn.qkv.weight)
                if block.attn.qkv.bias is not None:
                    nn.init.zeros_(block.attn.qkv.bias)
            else:
                # For separate Q, K, V
                for name, param in block.attn.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

            # Output projection
            if hasattr(block.attn, 'proj'):
                nn.init.xavier_uniform_(block.attn.proj.weight)
                if block.attn.proj.bias is not None:
                    nn.init.zeros_(block.attn.proj.bias)

            # MLP layers
            if hasattr(block, 'mlp'):
                for name, param in block.mlp.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        # Initialize cross-attention blocks similarly
        for block in self.cross_attn:
            if hasattr(block.attn, 'qkv'):
                nn.init.xavier_uniform_(block.attn.qkv.weight)
                if block.attn.qkv.bias is not None:
                    nn.init.zeros_(block.attn.qkv.bias)
            else:
                for name, param in block.attn.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

            if hasattr(block.attn, 'proj'):
                nn.init.xavier_uniform_(block.attn.proj.weight)
                if block.attn.proj.bias is not None:
                    nn.init.zeros_(block.attn.proj.bias)

            if hasattr(block, 'mlp'):
                for name, param in block.mlp.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        # Initialize decoder blocks
        for block in self.decoder_blocks:
            if hasattr(block.attn, 'qkv'):
                nn.init.xavier_uniform_(block.attn.qkv.weight)
                if block.attn.qkv.bias is not None:
                    nn.init.zeros_(block.attn.qkv.bias)
            else:
                for name, param in block.attn.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

            if hasattr(block.attn, 'proj'):
                nn.init.xavier_uniform_(block.attn.proj.weight)
                if block.attn.proj.bias is not None:
                    nn.init.zeros_(block.attn.proj.bias)

            if hasattr(block, 'mlp'):
                for name, param in block.mlp.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        # Initialize projection heads for contrastive learning
        for module in self.proj_head_global.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.proj_head_spatial.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize auxiliary encoder components with special care
        for modality, encoder in self.aux_encoder.items():
            for name, module in encoder.named_modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # Initialize modality projection
        nn.init.xavier_uniform_(self.modality_proj.weight)
        if self.modality_proj.bias is not None:
            nn.init.zeros_(self.modality_proj.bias)

        # Initialize final decoder prediction layer with smaller weights
        if isinstance(self.decoder_pred, nn.Sequential):
            for module in self.decoder_pred.modules():
                if isinstance(module, nn.Linear):
                    # Use smaller initialization for the final layer
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # Initialize normalization layers
        if hasattr(self, 'norm'):
            nn.init.constant_(self.norm.weight, 1.0)
            nn.init.constant_(self.norm.bias, 0.0)

        if hasattr(self, 'decoder_norm'):
            nn.init.constant_(self.decoder_norm.weight, 1.0)
            nn.init.constant_(self.decoder_norm.bias, 0.0)

        # Initialize auxiliary norm
        nn.init.constant_(self.aux_norm.weight, 1.0)
        nn.init.constant_(self.aux_norm.bias, 0.0)

        print("Robust initialization complete")

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
        with torch.amp.autocast('cuda', enabled=True):
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
                            # Check if we should disable checkpointing for diagnostics
                            use_checkpointing = self.training and not hasattr(self, '_disable_gradient_checkpointing')

                            # Define custom forward function for checkpointing
                            def create_custom_forward(mod):
                                def custom_forward(*inputs):
                                    return mod(torch.cat(inputs, dim=1))[:, :-1, :]

                                return custom_forward

                            if use_checkpointing:
                                # Use checkpointing during training
                                x = x + checkpoint(
                                    create_custom_forward(block),
                                    x, cond_tokens
                                )
                            else:
                                # Regular forward pass during evaluation or diagnostics
                                x = x + block(torch.cat([x, cond_tokens], dim=1))[:, :-1, :]

            # Apply main transformer blocks with gradient checkpointing
            for i, block in enumerate(self.blocks):
                # Check if we should disable checkpointing for diagnostics
                use_checkpointing = self.training and not hasattr(self, '_disable_gradient_checkpointing')

                if use_checkpointing:
                    # Use gradient checkpointing during training
                    x = checkpoint(block, x)
                else:
                    # Regular forward pass during evaluation or diagnostics
                    x = block(x)

            x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Decoder to reconstruct masked tokens in embedding space."""
        with torch.amp.autocast('cuda', enabled=True):
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
                # Check if we should disable checkpointing for diagnostics
                use_checkpointing = self.training and not hasattr(self, '_disable_gradient_checkpointing')

                if use_checkpointing:
                    # Use gradient checkpointing during training
                    x = checkpoint(block, x)
                else:
                    # Regular forward pass during evaluation or diagnostics
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
        with torch.amp.autocast('cuda', enabled=True):
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
        with torch.amp.autocast('cuda', enabled=True):
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
        with torch.amp.autocast('cuda', enabled=True):
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

    def calculate_inter_patch_diversity_loss(self, orig_patches, pred_patches, patch_mask, reference_similarity=None):
        """
        Calculate diversity loss between patches (inter-patch diversity) more efficiently.
        Uses pre-computed reference similarity if provided.

        Args:
            orig_patches: Original patches tensor [B, C, pixels_per_patch, num_patches]
            pred_patches: Predicted patches tensor [B, C, pixels_per_patch, num_patches]
            patch_mask: Mask indicating masked patches [B, 1, num_patches]
            reference_similarity: Pre-computed reference similarities [B] (optional)

        Returns:
            Tensor: Inter-patch diversity loss
        """
        B, C, pixels_per_patch, num_patches = orig_patches.shape
        device = orig_patches.device

        # Reshape for more efficient computation
        orig_patches = orig_patches.transpose(1, 2).reshape(B, pixels_per_patch * C, num_patches)
        pred_patches = pred_patches.transpose(1, 2).reshape(B, pixels_per_patch * C, num_patches)

        # Get masked and unmasked indicators
        patch_mask = patch_mask.squeeze(1)  # [B, num_patches]
        unmasked_patches = 1.0 - patch_mask  # [B, num_patches]

        # Pre-normalize all patches at once
        orig_patches_norm = F.normalize(orig_patches, p=2, dim=1)
        pred_patches_norm = F.normalize(pred_patches, p=2, dim=1)

        inter_patch_div_loss = 0.0

        for b in range(B):
            # Get masked and unmasked indices
            masked_indices = torch.where(patch_mask[b] > 0.5)[0]
            unmasked_indices = torch.where(unmasked_patches[b] > 0.5)[0]

            # Skip if not enough patches
            if len(masked_indices) < 2:
                continue

            # For masked (reconstructed) patches
            max_samples = min(50, len(masked_indices))
            sampled_masked = masked_indices[torch.randperm(len(masked_indices))[:max_samples]]

            # Extract normalized patches
            masked_patches = pred_patches_norm[b, :, sampled_masked]  # [pixels*C, samples]

            # Calculate similarity matrix efficiently
            masked_sim_matrix = torch.matmul(masked_patches.T, masked_patches)  # [samples, samples]

            # Create mask to exclude self-similarities
            mask_for_diag = torch.ones_like(masked_sim_matrix) - torch.eye(max_samples, device=device)

            # Calculate mean similarity excluding diagonal
            masked_mean_sim = (masked_sim_matrix * mask_for_diag).sum() / (mask_for_diag.sum() + 1e-8)

            # Get reference similarity (either pre-computed or calculate from unmasked patches)
            if reference_similarity is not None:
                orig_mean_sim = reference_similarity[b]
            elif len(unmasked_indices) >= 2:
                # Only calculate if not provided and enough unmasked patches exist
                max_unmasked = min(50, len(unmasked_indices))
                sampled_unmasked = unmasked_indices[torch.randperm(len(unmasked_indices))[:max_unmasked]]

                # Extract normalized patches
                unmasked_patches_norm = orig_patches_norm[b, :, sampled_unmasked]  # [pixels*C, samples]

                # Calculate similarity matrix
                unmasked_sim_matrix = torch.matmul(unmasked_patches_norm.T, unmasked_patches_norm)

                # Exclude self-similarity
                unmasked_mask = torch.ones_like(unmasked_sim_matrix) - torch.eye(max_unmasked, device=device)

                # Calculate mean similarity
                orig_mean_sim = (unmasked_sim_matrix * unmasked_mask).sum() / (unmasked_mask.sum() + 1e-8)
            else:
                # Fallback value if no reference available
                orig_mean_sim = torch.tensor(0.5, device=device)

            # Calculate diversity loss with margin - REPLACE CLAMP WITH SMOOTH HINGE
            margin = 0.1
            diversity_threshold = orig_mean_sim + margin
            # OLD: batch_div_loss = torch.clamp(masked_mean_sim - diversity_threshold, min=0.0)
            batch_div_loss = self.smooth_hinge(masked_mean_sim - diversity_threshold)
            inter_patch_div_loss += batch_div_loss

        # Normalize by batch size
        inter_patch_div_loss = inter_patch_div_loss / B

        return inter_patch_div_loss

    def calculate_batch_reference_variance(self, original_input, unmasked_pixels, num_samples=100):
        """
        Calculate reference variance statistics across the batch once.

        Args:
            original_input: Original HSI tensor [B, C, T, H, W]
            unmasked_pixels: Binary mask of unmasked regions [B, T, H, W]
            num_samples: Maximum number of samples to use

        Returns:
            Tensor with reference variance values [B, C, 1]
        """
        B, C, T, H, W = original_input.shape
        patch_h, patch_w = self.patch_size

        # Sample a subset of spectral bands
        band_step = max(1, T // 5)  # Sample ~5 bands
        reference_variances = []

        # Process a few random bands to estimate variance
        sampled_bands = torch.randperm(T)[:min(5, T)]

        for t in sampled_bands:
            # Get current band
            orig_band = original_input[:, :, t]  # [B, C, H, W]
            mask_band = unmasked_pixels[:, t] if unmasked_pixels.dim() >= 3 else unmasked_pixels  # [B, H, W]

            # Reshape for unfold
            orig_band_flat = orig_band.reshape(B * C, 1, H, W)

            # Extract patches
            orig_patches = F.unfold(
                orig_band_flat,
                kernel_size=(patch_h, patch_w),
                stride=(patch_h, patch_w)
            ).reshape(B, C, patch_h * patch_w, -1)

            # Calculate variance within each patch
            patch_vars = torch.var(orig_patches, dim=2)  # [B, C, num_patches]

            # Create patch-level mask
            mask_band_reshaped = mask_band.reshape(B, 1, H, W)
            mask_patches = F.unfold(
                mask_band_reshaped,
                kernel_size=(patch_h, patch_w),
                stride=(patch_h, patch_w)
            ).reshape(B, 1, patch_h * patch_w, -1)

            # A patch is considered unmasked if majority of pixels are unmasked
            unmasked_patches = (torch.mean(mask_patches, dim=2) > 0.5).float()  # [B, 1, num_patches]

            # Collect reference variance per batch
            batch_reference_vars = []

            for b in range(B):
                # Get indices of unmasked patches for this batch
                unmasked_indices = torch.where(unmasked_patches[b, 0] > 0.5)[0]

                # Skip if not enough unmasked patches
                if len(unmasked_indices) < 5:
                    # Fallback to using all patches if not enough unmasked ones
                    batch_ref_var = patch_vars[b].mean(dim=1, keepdim=True)
                else:
                    # Sample subset of unmasked patches
                    num_samples_actual = min(num_samples, len(unmasked_indices))
                    sampled_indices = unmasked_indices[torch.randperm(len(unmasked_indices))[:num_samples_actual]]

                    # Calculate mean variance of sampled patches
                    batch_vars = patch_vars[b, :, sampled_indices]
                    batch_ref_var = batch_vars.mean(dim=1, keepdim=True)

                batch_reference_vars.append(batch_ref_var)

            # Stack the reference variances for this band
            band_reference_var = torch.stack(batch_reference_vars, dim=0)  # [B, C, 1]
            reference_variances.append(band_reference_var)

        # Average across sampled bands
        avg_reference_variance = torch.stack(reference_variances).mean(dim=0)

        return avg_reference_variance

    def smooth_hinge(self, x, margin=0.0, beta=1.0):
        """
        A smooth approximation to the hinge function max(0, x).

        Args:
            x: Input tensor
            margin: Offset (like in hinge loss)
            beta: Smoothing parameter - larger values make it closer to hard hinge

        Returns:
            Tensor with smoothed hinge values
        """
        # Shift x by margin
        shifted_x = x - margin

        # Smooth approximation of max(0, shifted_x)
        # This is a softplus-based approximation: log(1 + exp(beta * x)) / beta
        return torch.log(1 + torch.exp(beta * shifted_x)) / beta

    def calculate_batch_reference_similarity(self, original_input, unmasked_pixels, num_samples=100):
        """
        Calculate reference inter-patch similarity statistics across the batch once.

        Args:
            original_input: Original HSI tensor [B, C, T, H, W]
            unmasked_pixels: Binary mask of unmasked regions [B, T, H, W]
            num_samples: Maximum number of samples to use

        Returns:
            Tensor with reference similarity values [B]
        """
        B, C, T, H, W = original_input.shape
        patch_h, patch_w = self.patch_size

        # Sample a subset of spectral bands
        band_step = max(1, T // 5)  # Sample ~5 bands
        reference_similarities = []

        # Process a few random bands
        sampled_bands = torch.randperm(T)[:min(3, T)]

        for t in sampled_bands:
            # Get current band
            orig_band = original_input[:, :, t]  # [B, C, H, W]
            mask_band = unmasked_pixels[:, t] if unmasked_pixels.dim() >= 3 else unmasked_pixels  # [B, H, W]

            # Reshape for unfold
            orig_band_flat = orig_band.reshape(B * C, 1, H, W)

            # Extract patches
            orig_patches = F.unfold(
                orig_band_flat,
                kernel_size=(patch_h, patch_w),
                stride=(patch_h, patch_w)
            ).reshape(B, C, patch_h * patch_w, -1)

            # Create patch-level mask
            mask_band_reshaped = mask_band.reshape(B, 1, H, W)
            mask_patches = F.unfold(
                mask_band_reshaped,
                kernel_size=(patch_h, patch_w),
                stride=(patch_h, patch_w)
            ).reshape(B, 1, patch_h * patch_w, -1)

            # A patch is considered unmasked if majority of pixels are unmasked
            unmasked_patches = (torch.mean(mask_patches, dim=2) > 0.5).float()  # [B, 1, num_patches]

            # Calculate inter-patch similarities per batch
            batch_similarities = []

            for b in range(B):
                # Get indices of unmasked patches for this batch
                unmasked_indices = torch.where(unmasked_patches[b, 0] > 0.5)[0]

                # Need at least 2 unmasked patches for similarity
                if len(unmasked_indices) < 2:
                    # Fallback to a default value if not enough patches
                    batch_similarities.append(torch.tensor(0.5, device=original_input.device))
                    continue

                # Sample a reasonable number of patches
                num_samples_actual = min(num_samples, len(unmasked_indices))
                sampled_indices = unmasked_indices[torch.randperm(len(unmasked_indices))[:num_samples_actual]]

                # Get the patches and normalize
                selected_patches = orig_patches[b, :, :, sampled_indices]  # [C, patch_size², samples]
                selected_patches = selected_patches.reshape(C * patch_h * patch_w, num_samples_actual)
                selected_patches_norm = F.normalize(selected_patches, p=2, dim=0)

                # Calculate similarity matrix
                sim_matrix = torch.matmul(selected_patches_norm.T, selected_patches_norm)

                # Exclude self-similarity (diagonal)
                mask = torch.ones_like(sim_matrix) - torch.eye(num_samples_actual, device=sim_matrix.device)

                # Calculate mean similarity excluding diagonal
                mean_sim = (sim_matrix * mask).sum() / (mask.sum() + 1e-8)
                batch_similarities.append(mean_sim)

            # Stack similarities for this band
            band_similarities = torch.stack(batch_similarities)  # [B]
            reference_similarities.append(band_similarities)

        # Average across sampled bands
        avg_reference_similarities = torch.stack(reference_similarities).mean(dim=0)  # [B]

        return avg_reference_similarities

    def forward_loss_in_pixel_space(self, pred_pixels, original_input, mask, thickness_mask=None):
        """
        Compute loss in pixel space with two diversity penalties with improved memory management and NaN handling:
        1. Intra-patch variance loss - penalizes patches that are too uniform internally
        2. Inter-patch diversity loss - penalizes when masked patches are more similar to each other than unmasked ones

        With improved error handling and robust reference value tracking
        """
        with torch.cuda.amp.autocast(enabled=True):
            # Get the device from input tensors
            device = original_input.device
            epsilon = 1e-6  # Small positive constant for numerical stability

            # Convert token mask to 3D pixel mask
            pixel_mask = self.token_mask_to_pixel_mask(mask, original_input.shape)  # [B, T, H, W]

            # Debug: Check masking percentage
            masked_percentage = pixel_mask.float().mean().item() * 100
            print(f"Percentage of pixels masked: {masked_percentage:.2f}%")

            # Standard MSE calculation
            pixel_mse = ((pred_pixels - original_input) ** 2)

            # Apply thickness mask if provided
            if thickness_mask is not None and self.use_thickness_mask:
                # Expand thickness mask to match dimensions
                if thickness_mask.dim() == 4:  # [B, 1, H, W]
                    expanded_thickness = thickness_mask.expand(-1, pixel_mask.shape[1], -1, -1)
                else:
                    expanded_thickness = thickness_mask

                # Calculate percentage of valid pixels in thickness mask
                valid_thickness_percent = expanded_thickness.float().mean().item() * 100
                print(f"Percentage of valid pixels in thickness mask: {valid_thickness_percent:.2f}%")

                # Combine masks
                combined_mask = pixel_mask * expanded_thickness
                valid_pixel_count = combined_mask.sum() + epsilon
                mse_loss = (pixel_mse * combined_mask).sum() / valid_pixel_count

                # Use this mask for diversity calculations
                final_mask = combined_mask

                # Debug: Check combined mask stats
                combined_mask_percent = combined_mask.float().mean().item() * 100
                print(f"Percentage of pixels used in loss calculation: {combined_mask_percent:.2f}%")
            else:
                # Apply only pixel mask
                masked_pixel_count = pixel_mask.sum() + epsilon
                mse_loss = (pixel_mse * pixel_mask).sum() / masked_pixel_count

                # Use pixel mask for diversity calculations
                final_mask = pixel_mask

            # Get basic dimensions
            B, C, T, H, W = original_input.shape
            patch_h, patch_w = self.patch_size

            # Initialize diversity losses
            intra_patch_div_loss = 0.0  # Variance within patches
            inter_patch_div_loss = 0.0  # Diversity between patches
            bands_processed = 0

            # Create containers for reference values
            all_reference_variances = []  # Original patch variances
            all_variance_thresholds = []  # Thresholds for variance
            all_diversity_thresholds = []  # Thresholds for inter-patch similarity

            # New containers for reconstructed values
            all_reconstructed_variances = []  # Reconstructed patch variances
            all_reconstructed_similarities = []  # Reconstructed patch similarities

            # Process a subset of bands to save computation (e.g., every 5th band)
            band_step = max(1, T // 6)  # Process ~6 bands evenly spaced
            print(f"Processing diversity for {T // band_step} bands with step {band_step}")

            for t in range(0, T, band_step):
                print(f"Processing band {t}/{T}...")
                # Get current spectral band
                orig_band = original_input[:, :, t].detach()  # [B, C, H, W]
                pred_band = pred_pixels[:, :, t].detach()
                mask_band = final_mask[:, t].detach() if final_mask.dim() >= 3 else final_mask.detach()  # [B, H, W]

                # Invert mask to get unmasked regions (0 = masked, 1 = visible)
                unmasked = 1.0 - mask_band

                # Reshape for unfold operation
                orig_band_flat = orig_band.reshape(B * C, 1, H, W)
                pred_band_flat = pred_band.reshape(B * C, 1, H, W)

                # Extract patches
                orig_patches = F.unfold(
                    orig_band_flat,
                    kernel_size=(patch_h, patch_w),
                    stride=(patch_h, patch_w)
                ).reshape(B, C, patch_h * patch_w, -1)

                pred_patches = F.unfold(
                    pred_band_flat,
                    kernel_size=(patch_h, patch_w),
                    stride=(patch_h, patch_w)
                ).reshape(B, C, patch_h * patch_w, -1)

                # Calculate variance within each patch (intra-patch)
                orig_var = torch.var(orig_patches, dim=2)  # [B, C, num_patches]
                pred_var = torch.var(pred_patches, dim=2)  # [B, C, num_patches]

                # Handle any nan values in variance calculations
                orig_var = torch.where(~torch.isfinite(orig_var), torch.tensor(epsilon, device=device), orig_var)
                pred_var = torch.where(~torch.isfinite(pred_var), torch.tensor(epsilon, device=device), pred_var)

                # Create patch-level mask
                mask_band_reshaped = mask_band.reshape(B, 1, H, W)
                mask_patches = F.unfold(
                    mask_band_reshaped,
                    kernel_size=(patch_h, patch_w),
                    stride=(patch_h, patch_w)
                ).reshape(B, 1, patch_h * patch_w, -1)

                # A patch is considered masked if majority of pixels are masked
                patch_mask = (torch.mean(mask_patches, dim=2) > 0.5).float()  # [B, 1, num_patches]

                # Debug mask statistics
                num_masked_patches = patch_mask.sum().item()
                patch_masked_percentage = (patch_mask.sum() / patch_mask.numel() * 100).item()
                print(f"  Band {t}: {num_masked_patches} masked patches ({patch_masked_percentage:.2f}%)")

                # Clean up intermediate tensors
                del mask_patches, mask_band_reshaped
                torch.cuda.empty_cache()

                # INTRA-PATCH DIVERSITY
                # Calculate reference variance from unmasked patches
                unmasked_patches = 1.0 - patch_mask

                # Reference variance calculation with memory management
                reference_variance = self._calculate_reference_variance(
                    orig_var, unmasked_patches, B
                )  # [B, C, 1]

                # Debug reference variance
                print(f"  Band {t}: Reference variance mean: {reference_variance.mean().item():.6f}")

                # Check for invalid reference variance and fix if needed
                if not torch.isfinite(reference_variance).all():
                    reference_variance = torch.where(
                        ~torch.isfinite(reference_variance),
                        torch.tensor(epsilon, device=device, dtype=reference_variance.dtype),
                        reference_variance
                    )
                    print(f"  Band {t}: Fixed non-finite values in reference variance")

                # Save reference variance
                reference_variance_detached = reference_variance.detach().clone()
                all_reference_variances.append(reference_variance_detached)
                print(
                    f"  Band {t}: Added reference variance with mean: {reference_variance_detached.mean().item():.6f}")

                # Calculate reconstructed variances (average variance of masked patches)
                reconstructed_variance = self._calculate_reference_variance(
                    pred_var, patch_mask, B
                )  # [B, C, 1]

                # Debug reconstructed variance
                print(f"  Band {t}: Reconstructed variance mean: {reconstructed_variance.mean().item():.6f}")

                # Check for invalid reconstructed variance and fix if needed
                if not torch.isfinite(reconstructed_variance).all():
                    reconstructed_variance = torch.where(
                        ~torch.isfinite(reconstructed_variance),
                        torch.tensor(epsilon, device=device, dtype=reconstructed_variance.dtype),
                        reconstructed_variance
                    )
                    print(f"  Band {t}: Fixed non-finite values in reconstructed variance")

                # Save reconstructed variance
                reconstructed_variance_detached = reconstructed_variance.detach().clone()
                all_reconstructed_variances.append(reconstructed_variance_detached)
                print(
                    f"  Band {t}: Added reconstructed variance with mean: {reconstructed_variance_detached.mean().item():.6f}")

                # Set adaptive threshold as percentage of reference variance
                threshold_ratio = 0.4  # Patches are permitted to have as low as 40% as variance as reference
                min_variance_threshold = reference_variance * threshold_ratio

                # Debug variance threshold
                print(f"  Band {t}: Variance threshold mean: {min_variance_threshold.mean().item():.6f}")

                # Check for invalid variance threshold and fix if needed
                if not torch.isfinite(min_variance_threshold).all():
                    min_variance_threshold = torch.where(
                        ~torch.isfinite(min_variance_threshold),
                        torch.tensor(epsilon, device=device, dtype=min_variance_threshold.dtype),
                        min_variance_threshold
                    )
                    print(f"  Band {t}: Fixed non-finite values in variance threshold")

                # Save variance threshold
                variance_threshold_detached = min_variance_threshold.detach().clone()
                all_variance_thresholds.append(variance_threshold_detached)
                print(
                    f"  Band {t}: Added variance threshold with mean: {variance_threshold_detached.mean().item():.6f}")

                # Calculate variance deficit where prediction is too uniform
                # Only apply to masked patches
                too_uniform = (pred_var < min_variance_threshold) & (patch_mask > 0.5)
                variance_deficit = self.smooth_hinge(min_variance_threshold - pred_var)

                # Handle any potential nan values in variance deficit
                variance_deficit = torch.where(~torch.isfinite(variance_deficit), torch.tensor(0.0, device=device),
                                               variance_deficit)

                # Final intra-patch diversity loss - only applied to suspiciously uniform masked patches
                band_intra_div_loss = (variance_deficit * too_uniform.float()).sum()

                # Normalize by number of masked patches
                num_masked_patches = patch_mask.sum() + epsilon
                band_intra_div_loss = band_intra_div_loss / num_masked_patches

                # Debug intra-patch diversity loss
                print(f"  Band {t}: Intra-patch diversity loss: {band_intra_div_loss.item():.6f}")

                # Clean up intermediate tensors
                del variance_deficit, too_uniform, min_variance_threshold
                torch.cuda.empty_cache()

                # INTER-PATCH DIVERSITY
                # Calculate diversity between patches for this band using memory-optimized function
                band_inter_div_loss = self._calculate_inter_patch_diversity_loss_optimized(
                    orig_patches, pred_patches, patch_mask
                )

                # Debug inter-patch diversity loss
                print(f"  Band {t}: Inter-patch diversity loss: {band_inter_div_loss.item():.6f}")

                # Calculate original unmasked patch similarities
                orig_patches_flat = orig_patches.transpose(1, 2).reshape(B, patch_h * patch_w * C, -1)
                orig_patches_norm = F.normalize(orig_patches_flat, p=2, dim=1)

                # Calculate reconstructed patch similarities
                pred_patches_flat = pred_patches.transpose(1, 2).reshape(B, patch_h * patch_w * C, -1)
                pred_patches_norm = F.normalize(pred_patches_flat, p=2, dim=1)

                # Get average similarity between unmasked patches (original)
                orig_sum_sim = 0.0
                orig_count_sim = 0

                # Get average similarity between masked patches (reconstructed)
                pred_sum_sim = 0.0
                pred_count_sim = 0

                # Process one batch at a time for memory efficiency
                for b in range(B):
                    # Process original unmasked patches
                    unmasked_indices = torch.where(unmasked_patches[b, 0] > 0.5)[0]
                    if len(unmasked_indices) >= 2:
                        max_samples = min(50, len(unmasked_indices))
                        sampled = unmasked_indices[torch.randperm(len(unmasked_indices))[:max_samples]]
                        selected = orig_patches_norm[b, :, sampled]
                        sim_matrix = torch.matmul(selected.T, selected)
                        diag_mask = torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0],
                                                                            device=sim_matrix.device)
                        if diag_mask.sum() > 0:
                            batch_sim = (sim_matrix * diag_mask).sum() / diag_mask.sum()

                            # Check for non-finite values
                            if torch.isfinite(batch_sim):
                                orig_sum_sim += batch_sim.item()
                                orig_count_sim += 1

                    # Process reconstructed masked patches
                    masked_indices = torch.where(patch_mask[b, 0] > 0.5)[0]
                    if len(masked_indices) >= 2:
                        max_samples = min(50, len(masked_indices))
                        sampled = masked_indices[torch.randperm(len(masked_indices))[:max_samples]]
                        selected = pred_patches_norm[b, :, sampled]
                        sim_matrix = torch.matmul(selected.T, selected)
                        diag_mask = torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0],
                                                                            device=sim_matrix.device)
                        if diag_mask.sum() > 0:
                            batch_sim = (sim_matrix * diag_mask).sum() / diag_mask.sum()

                            # Check for non-finite values
                            if torch.isfinite(batch_sim):
                                pred_sum_sim += batch_sim.item()
                                pred_count_sim += 1

                # Calculate average similarities
                if orig_count_sim > 0:
                    orig_mean_sim = orig_sum_sim / orig_count_sim
                    margin = 0.1
                    diversity_threshold = orig_mean_sim + margin

                    # Debug diversity threshold
                    print(f"  Band {t}: Diversity threshold: {diversity_threshold:.6f} (from {orig_count_sim} samples)")

                    # Save diversity threshold for reporting
                    diversity_threshold_tensor = torch.tensor(diversity_threshold, device=device)
                    all_diversity_thresholds.append(diversity_threshold_tensor)
                    print(f"  Band {t}: Added diversity threshold: {diversity_threshold:.6f}")
                else:
                    # Use a default value if no valid similarities calculated
                    diversity_threshold = 0.5 + 0.1  # Default mean + margin
                    diversity_threshold_tensor = torch.tensor(diversity_threshold, device=device)
                    all_diversity_thresholds.append(diversity_threshold_tensor)
                    print(f"  Band {t}: Using default diversity threshold: {diversity_threshold:.6f}")

                if pred_count_sim > 0:
                    pred_mean_sim = pred_sum_sim / pred_count_sim

                    # Debug reconstructed similarity
                    print(f"  Band {t}: Reconstructed similarity: {pred_mean_sim:.6f} (from {pred_count_sim} samples)")

                    # Save reconstructed similarity for reporting
                    reconstructed_similarity_tensor = torch.tensor(pred_mean_sim, device=device)
                    all_reconstructed_similarities.append(reconstructed_similarity_tensor)
                    print(f"  Band {t}: Added reconstructed similarity: {pred_mean_sim:.6f}")
                else:
                    # Use a default value if no valid similarities calculated
                    pred_mean_sim = 0.5  # Default value
                    reconstructed_similarity_tensor = torch.tensor(pred_mean_sim, device=device)
                    all_reconstructed_similarities.append(reconstructed_similarity_tensor)
                    print(f"  Band {t}: Using default reconstructed similarity: {pred_mean_sim:.6f}")

                # Clean up large tensor variables
                del orig_patches, pred_patches, patch_mask
                del orig_band, pred_band, mask_band, orig_band_flat, pred_band_flat
                del orig_var, pred_var, reference_variance, unmasked_patches
                del orig_patches_flat, orig_patches_norm, pred_patches_flat, pred_patches_norm
                torch.cuda.empty_cache()

                # Accumulate diversity losses across bands
                intra_patch_div_loss += band_intra_div_loss
                inter_patch_div_loss += band_inter_div_loss
                bands_processed += 1

            # Normalize by number of bands processed
            if bands_processed > 0:
                intra_patch_div_loss = intra_patch_div_loss / bands_processed
                inter_patch_div_loss = inter_patch_div_loss / bands_processed

            # Get loss weights from config
            intra_div_weight = getattr(self, 'intra_div_weight', 0.1)  # Default if not configured
            inter_div_weight = getattr(self, 'inter_div_weight', 0.1)  # Default if not configured

            # Combined loss
            total_loss = mse_loss + intra_div_weight * intra_patch_div_loss + inter_div_weight * inter_patch_div_loss

            # For debugging
            if not self.training and torch.rand(1).item() < 0.05:  # Log occasionally during validation
                print(f"MSE Loss: {mse_loss.item():.6f}, "
                      f"Intra-Patch Div Loss: {intra_patch_div_loss.item():.6f}, "
                      f"Inter-Patch Div Loss: {inter_patch_div_loss.item():.6f}")

            # Clean up large tensor variables
            del pixel_mask, pixel_mse, final_mask
            torch.cuda.empty_cache()

            # Safely calculate aggregate statistics with robust error handling
            # Calculate mean reference variance
            if all_reference_variances:
                try:
                    mean_reference_variance = torch.stack([v.mean() for v in all_reference_variances]).mean()
                    if not torch.isfinite(mean_reference_variance):
                        mean_reference_variance = torch.tensor(epsilon, device=device)
                except Exception as e:
                    print(f"Error calculating mean reference variance: {e}")
                    mean_reference_variance = torch.tensor(epsilon, device=device)
            else:
                mean_reference_variance = torch.tensor(epsilon, device=device)

            # Calculate mean variance threshold
            if all_variance_thresholds:
                try:
                    mean_variance_threshold = torch.stack([v.mean() for v in all_variance_thresholds]).mean()
                    if not torch.isfinite(mean_variance_threshold):
                        mean_variance_threshold = torch.tensor(epsilon, device=device)
                except Exception as e:
                    print(f"Error calculating mean variance threshold: {e}")
                    mean_variance_threshold = torch.tensor(epsilon, device=device)
            else:
                mean_variance_threshold = torch.tensor(epsilon, device=device)

            # Calculate mean diversity threshold
            if all_diversity_thresholds:
                try:
                    mean_diversity_threshold = torch.stack(all_diversity_thresholds).mean()
                    if not torch.isfinite(mean_diversity_threshold):
                        mean_diversity_threshold = torch.tensor(0.6, device=device)  # Default value
                except Exception as e:
                    print(f"Error calculating mean diversity threshold: {e}")
                    mean_diversity_threshold = torch.tensor(0.6, device=device)  # Default value
            else:
                mean_diversity_threshold = torch.tensor(0.6, device=device)  # Default value

            # Calculate mean reconstructed variance
            if all_reconstructed_variances:
                try:
                    mean_reconstructed_variance = torch.stack([v.mean() for v in all_reconstructed_variances]).mean()
                    if not torch.isfinite(mean_reconstructed_variance):
                        mean_reconstructed_variance = torch.tensor(epsilon, device=device)
                except Exception as e:
                    print(f"Error calculating mean reconstructed variance: {e}")
                    mean_reconstructed_variance = torch.tensor(epsilon, device=device)
            else:
                mean_reconstructed_variance = torch.tensor(epsilon, device=device)

            # Calculate mean reconstructed similarity
            if all_reconstructed_similarities:
                try:
                    mean_reconstructed_similarity = torch.stack(all_reconstructed_similarities).mean()
                    if not torch.isfinite(mean_reconstructed_similarity):
                        mean_reconstructed_similarity = torch.tensor(0.5, device=device)  # Default value
                except Exception as e:
                    print(f"Error calculating mean reconstructed similarity: {e}")
                    mean_reconstructed_similarity = torch.tensor(0.5, device=device)  # Default value
            else:
                mean_reconstructed_similarity = torch.tensor(0.5, device=device)  # Default value

            # Print the final aggregated values
            print(f"Final aggregated values:")
            print(f"  Mean reference variance: {mean_reference_variance.item():.6f}")
            print(f"  Mean variance threshold: {mean_variance_threshold.item():.6f}")
            print(f"  Mean diversity threshold: {mean_diversity_threshold.item():.6f}")
            print(f"  Mean reconstructed variance: {mean_reconstructed_variance.item():.6f}")
            print(f"  Mean reconstructed similarity: {mean_reconstructed_similarity.item():.6f}")

            # Return dictionary with all loss components and diversity values for tracking
            return {
                'total_loss': total_loss,
                'mse_loss': mse_loss,
                'intra_patch_div_loss': intra_patch_div_loss,
                'inter_patch_div_loss': inter_patch_div_loss,
                'reference_variance': mean_reference_variance,
                'variance_threshold': mean_variance_threshold,
                'diversity_threshold': mean_diversity_threshold,
                'reconstructed_variance': mean_reconstructed_variance,
                'reconstructed_similarity': mean_reconstructed_similarity
            }

    def _calculate_reference_variance(self, orig_var, unmasked_patches, batch_size):
        """
        Memory-optimized helper function to calculate reference variance from unmasked patches.
        With improved robustness against NaN values.

        Args:
            orig_var: Variance of original patches [B, C, num_patches]
            unmasked_patches: Mask of unmasked patches [B, 1, num_patches]
            batch_size: Number of batches

        Returns:
            torch.Tensor: Reference variance [B, C, 1]
        """
        max_samples = 100  # Number of samples to use
        reference_variances = []
        epsilon = 1e-6  # Small positive constant for numerical stability

        # Process one batch at a time
        for b in range(batch_size):
            # Get indices of unmasked patches for this batch
            unmasked_indices = torch.where(unmasked_patches[b, 0] > 0.5)[0]

            # Skip if we don't have enough unmasked patches
            if len(unmasked_indices) < 5:  # Need some minimum number
                # Fallback to using all available unmasked patches
                if len(unmasked_indices) > 0:
                    batch_vars = orig_var[b, :, unmasked_indices]

                    # Check for non-finite values and replace them
                    non_finite_mask = ~torch.isfinite(batch_vars)
                    if non_finite_mask.any():
                        # Replace non-finite values with small positive constant
                        batch_vars = torch.where(non_finite_mask,
                                                 torch.tensor(epsilon, device=batch_vars.device,
                                                              dtype=batch_vars.dtype),
                                                 batch_vars)

                    batch_ref_var = batch_vars.mean(dim=1, keepdim=True)

                    # Final check for non-finite values
                    if not torch.isfinite(batch_ref_var).all():
                        # Fallback to a small positive constant if still have issues
                        batch_ref_var = torch.tensor([epsilon], device=orig_var.device).view(1, 1)
                else:
                    # If no unmasked patches, use a default value
                    batch_ref_var = torch.tensor([epsilon], device=orig_var.device).view(1, 1)
            else:
                # Sample subset of unmasked patches
                num_samples = min(max_samples, len(unmasked_indices))
                sampled_indices = unmasked_indices[
                    torch.randperm(len(unmasked_indices), device=unmasked_indices.device)[:num_samples]]

                # Calculate mean variance of sampled patches
                batch_vars = orig_var[b, :, sampled_indices]

                # Check for non-finite values and replace them
                non_finite_mask = ~torch.isfinite(batch_vars)
                if non_finite_mask.any():
                    # Replace non-finite values with small positive constant
                    batch_vars = torch.where(non_finite_mask,
                                             torch.tensor(epsilon, device=batch_vars.device, dtype=batch_vars.dtype),
                                             batch_vars)

                batch_ref_var = batch_vars.mean(dim=1, keepdim=True)

                # Final check for non-finite values
                if not torch.isfinite(batch_ref_var).all():
                    # Fallback to a small positive constant if still have issues
                    batch_ref_var = torch.tensor([epsilon], device=orig_var.device).view(1, 1)

                # Clean up
                del sampled_indices, batch_vars

            reference_variances.append(batch_ref_var)
            del unmasked_indices

        # Stack the reference variances for all batches
        reference_variance = torch.stack(reference_variances, dim=0)  # [B, C, 1]
        del reference_variances

        # Final safety check
        if not torch.isfinite(reference_variance).all():
            # Replace any remaining non-finite values with a small positive constant
            reference_variance = torch.where(
                ~torch.isfinite(reference_variance),
                torch.tensor(epsilon, device=reference_variance.device, dtype=reference_variance.dtype),
                reference_variance
            )

        return reference_variance

    def _calculate_inter_patch_diversity_loss_optimized(self, orig_patches, pred_patches, patch_mask):
        """
        Memory-optimized implementation of inter-patch diversity loss calculation.

        Args:
            orig_patches: Original patches tensor [B, C, patch_pixels, num_patches]
            pred_patches: Predicted patches tensor [B, C, patch_pixels, num_patches]
            patch_mask: Mask indicating masked patches [B, 1, num_patches]

        Returns:
            torch.Tensor: Inter-patch diversity loss
        """
        B, C, pixels_per_patch, num_patches = orig_patches.shape
        device = orig_patches.device

        # We'll use cosine similarity to measure patch similarity
        # First, normalize patches along pixel dimension - use in-place operations where possible
        orig_patches_norm = F.normalize(orig_patches, p=2, dim=2)
        pred_patches_norm = F.normalize(pred_patches, p=2, dim=2)

        # Get unmasked patches mask (0 = masked, 1 = visible)
        unmasked_patches = 1.0 - patch_mask  # [B, 1, num_patches]

        # Sample pairs of patches for diversity calculation
        max_samples = min(50, num_patches // 2)  # Reduced from 100 to 50 for memory efficiency

        # Initialize accumulator for batch diversity losses
        total_batch_div_loss = 0.0
        valid_batches = 0

        # Process one batch at a time to save memory
        for b in range(B):
            try:
                # Get indices of masked and unmasked patches for this batch
                masked_indices = torch.where(patch_mask[b, 0] > 0.5)[0]
                unmasked_indices = torch.where(unmasked_patches[b, 0] > 0.5)[0]

                # Skip if we don't have enough patches
                if len(unmasked_indices) < 2 or len(masked_indices) < 2:
                    continue

                # Process original unmasked patches - compute pairwise similarities
                orig_mean_sim = self._compute_patch_similarity(
                    orig_patches_norm[b], unmasked_indices, max_samples, C, device
                )

                # Process predicted masked patches - compute pairwise similarities
                pred_mean_sim = self._compute_patch_similarity(
                    pred_patches_norm[b], masked_indices, max_samples, C, device
                )

                # Calculate inter-patch diversity loss - use a margin
                margin = 0.1
                diversity_threshold = orig_mean_sim + margin

                # Penalize if reconstructed patches are too similar
                # OLD: batch_div_loss = torch.clamp(pred_mean_sim - diversity_threshold, min=0.0)
                batch_div_loss = self.smooth_hinge(pred_mean_sim - diversity_threshold)
                total_batch_div_loss += batch_div_loss
                valid_batches += 1

                # Clean up
                del masked_indices, unmasked_indices
                del orig_mean_sim, pred_mean_sim, diversity_threshold, batch_div_loss

            except Exception as e:
                print(f"Error in batch {b} diversity calculation: {e}")
                continue

        # Compute final loss
        if valid_batches > 0:
            inter_patch_div_loss = total_batch_div_loss / valid_batches
        else:
            inter_patch_div_loss = torch.tensor(0.0, device=device)

        # Clean up
        del total_batch_div_loss, unmasked_patches
        torch.cuda.empty_cache()

        return inter_patch_div_loss

    def _compute_patch_similarity(self, patches_norm, indices, max_samples, num_channels, device):
        """
        Helper function to compute average similarity between patches.
        Uses sampling and processes in small batches to save memory.

        Args:
            patches_norm: Normalized patches [C, patch_pixels, num_patches]
            indices: Indices of patches to consider
            max_samples: Maximum number of patch pairs to sample
            num_channels: Number of channels
            device: Current device

        Returns:
            torch.Tensor: Mean similarity between patches
        """
        # Sample pairs from provided indices
        num_pairs = min(max_samples, len(indices) // 2 * 2)

        if num_pairs < 2:
            return torch.tensor(0.5, device=device)

        # Create random permutation of indices and take pairs
        sampled = indices[torch.randperm(len(indices), device=device)[:num_pairs]]
        pairs = sampled.reshape(-1, 2)  # Reshape to pairs

        # For memory efficiency, process in smaller batches of pairs
        batch_size = 10  # Process 10 pairs at a time
        total_sim = 0.0
        num_processed = 0

        for start_idx in range(0, len(pairs), batch_size):
            end_idx = min(start_idx + batch_size, len(pairs))
            batch_pairs = pairs[start_idx:end_idx]

            batch_sims = []
            for i, j in batch_pairs:
                # Calculate cosine similarity between this pair
                sim = F.cosine_similarity(
                    patches_norm[:, :, i].view(num_channels, -1),
                    patches_norm[:, :, j].view(num_channels, -1),
                    dim=1
                ).mean()  # Average over channels
                batch_sims.append(sim)

            # Accumulate similarity for this batch
            if batch_sims:
                batch_mean = torch.stack(batch_sims).mean()
                total_sim += batch_mean.item() * len(batch_sims)
                num_processed += len(batch_sims)

                # Clean up
                del batch_sims, batch_mean

        # Clean up
        del sampled, pairs

        # Return mean similarity
        if num_processed > 0:
            return torch.tensor(total_sim / num_processed, device=device)
        else:
            return torch.tensor(0.5, device=device)

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
            with torch.amp.autocast('cuda', enabled=True):
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
        with torch.amp.autocast('cuda', enabled=True):
            latent, mask, ids_restore = self.forward_encoder(hsi_img, aux_data)

            # Decode and reconstruct
            pred_tokens = self.forward_decoder(latent, ids_restore)

            # Reshape for unpatchify (from flat token sequence to organized patches)
            B = hsi_img.shape[0]
            pred_tokens_reshaped = pred_tokens.reshape(
                B, self.spectral_patches, self.spatial_patches, -1
            )

            # Unpatchify to pixel space directly
            reconstructed_pixels = self.unpatchify(pred_tokens_reshaped, original_input.shape)

            # Convert token mask to pixel mask (1 where masked, 0 where visible)
            pixel_mask = self.token_mask_to_pixel_mask(mask, original_input.shape)  # [B, T, H, W]
            pixel_mask = pixel_mask.unsqueeze(1)  # -> [B, 1, T, H, W] for broadcasting


            # Create inverse mask (1 where visible, 0 where masked)
            inverse_pixel_mask = 1.0 - pixel_mask

            # Combine original (unmasked) and reconstructed (masked) pixels
            combined_reconstruction = (reconstructed_pixels * pixel_mask) + (original_input * inverse_pixel_mask)

            # Replace the reconstructed_pixels with the combined version
            reconstructed_pixels = combined_reconstruction

            # Calculate reconstruction losses (now returns a dictionary with all loss components)
            losses = self.forward_loss_in_pixel_space(
                reconstructed_pixels,
                original_input,
                mask,
                thickness_mask
            )

            # Extract individual losses
            loss_recon = losses['total_loss']
            mse_loss = losses['mse_loss']
            intra_patch_div_loss = losses['intra_patch_div_loss']
            inter_patch_div_loss = losses['inter_patch_div_loss']

        # Calculate contrastive loss if auxiliary data present
        loss_contrast = torch.tensor(0.0, device=device)
        num_available = 0
        if aux_data is not None and batch_idx is not None and unmasked_features is not None:
            # Count available modalities for logging
            num_available = sum(1 for v in aux_data.values() if v is not None)

            # Only compute contrastive loss if at least one modality is available
            if num_available > 0:
                with torch.amp.autocast('cuda', enabled=True):
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

        # Get reference and reconstructed values
        reference_variance = losses.get('reference_variance', torch.tensor(0.0, device=device))
        variance_threshold = losses.get('variance_threshold', torch.tensor(0.0, device=device))
        diversity_threshold = losses.get('diversity_threshold', torch.tensor(0.0, device=device))
        reconstructed_variance = losses.get('reconstructed_variance', torch.tensor(0.0, device=device))
        reconstructed_similarity = losses.get('reconstructed_similarity', torch.tensor(0.0, device=device))

        return {
            'loss': loss,
            'loss_recon': loss_recon,
            'loss_contrast': loss_contrast,
            'mse_loss': mse_loss,
            'intra_patch_div_loss': intra_patch_div_loss,
            'inter_patch_div_loss': inter_patch_div_loss,
            'num_modalities': torch.tensor(num_available, device=device),
            'pred': pred_tokens,
            'mask': mask,
            'thickness_mask': thickness_mask,
            'contrastive_mode': self.contrastive_mode,
            'original_input': original_input,
            'reconstructed_pixels': reconstructed_pixels,  # This should now be the combined version
            'reference_variance': reference_variance,
            'variance_threshold': variance_threshold,
            'diversity_threshold': diversity_threshold,
            'reconstructed_variance': reconstructed_variance,
            'reconstructed_similarity': reconstructed_similarity
        }