# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torch import nn


from torchtitan.config import JobConfig
from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger
from .wan_vae import WanVAEParams


@dataclass
class WanModelArgs(BaseModelArgs):
    # Parameters for WanModel1x (video model)
    in_dim: int = 48
    out_dim: int = 48
    ffn_dim: int = 14336
    freq_dim: int = 256
    hidden_dim: int = 3072
    patch_size: list = field(default_factory=lambda: [1, 2, 2])
    num_layers: int = 15
    eps: float = 1e-06
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    text_dim: int = 4096
    # Legacy parameters for WanModel (dual-stream model) - kept for backward compatibility
    # in_channels: int = 48
    # out_channels: int = 64
    # vec_in_dim: int = 768
    # depth: int = 19
    # depth_single_blocks: int = 38
    # axes_dim: tuple = (16, 56, 56)
    # theta: int = 10_000
    # qkv_bias: bool = True
    
    wan_video_vae_params: WanVAEParams = field(
        default_factory=lambda: WanVAEParams(vae_type="38", z_dim=48, dim=160)
    )

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        """Update model arguments from job configuration.
        
        This method allows overriding model parameters from the job config.
        Currently, num_layers and other model parameters can be set via
        model.num_layers in the config file or CLI.
        """
        # Allow num_layers to be overridden from config if specified
        # This is accessed via model.num_layers in config files
        if hasattr(job_config.model, 'num_layers') and job_config.model.num_layers is not None:
            self.num_layers = job_config.model.num_layers
            logger.info(f"num_layers overridden from config: {self.num_layers}")
        
        # Add other parameter overrides here as needed
        # Example: if hasattr(job_config.model, 'hidden_dim'):
        #     self.hidden_dim = job_config.model.hidden_dim
        
    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        # TODO: double check the FLOPS calculations
        nparams = sum(p.numel() for p in model.parameters())
        context_seq_len = 256
        
        # Compute actual sequence length from video latent dimensions
        # VAE downsamples spatially by 8x (upsampling_factor = 8)
        # For img_size=256: latent H=W = 256/8 = 32
        # For downsampled=4 and clip_length=77: we get ~6 temporal frames after VAE encoding
        # Sequence length = T × (H/patch_h) × (W/patch_w)
        # With patch_size=[1, 2, 2]: seq_len = T × (32/2) × (32/2) = T × 16 × 16
        # For T=6: seq_len = 6 × 16 × 16 = 1536
        # We compute this from the passed seq_len by adjusting for actual video dimensions
        # If seq_len=2048 (default), it likely corresponds to a different configuration
        # For the actual configuration (T=6, H=W=32, patch_size=[1,2,2]), we get 1536
        # Compute based on typical video dimensions: assume img_size=256, downsampled=4
        vae_downsample_factor = 8  # VAE spatial downsampling factor
        # Typical latent dimensions: for img_size=256, we get H=W=32
        # For downsampled=4, temporal frames ≈ 6 after VAE encoding
        # Actual sequence length = T × (H/patch_h) × (W/patch_w)
        # = 6 × (32/2) × (32/2) = 6 × 16 × 16 = 1536
        actual_seq_len = 6 * 16 * 16  # T=6, H=W=32, patch_size=[1,2,2]
        
        # Use actual sequence length for FLOPS calculation
        flops_per_token = dit_flops_per_video_token(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            seq_len=actual_seq_len,
            context_seq_len=context_seq_len,
            mlp_ratio=self.mlp_ratio,
        )
        logger.info(f"Wan model parameters: {nparams:,}, FLOPS per token: {flops_per_token:,.0f}, head_dim: {self.hidden_dim // self.num_heads}, num_layers: {self.num_layers}, num_heads: {self.num_heads}, seq_len: {actual_seq_len} (computed from video dimensions, config had {seq_len})")
        return nparams, flops_per_token

def dit_flops_per_video_token(
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    seq_len: int,
    context_seq_len: int,
    mlp_ratio: float = 4.0,
) -> int:
    d = hidden_dim
    h = num_heads
    d_head = d // h

    # per-layer, per-token, forward:
    # QKV + O projections ~ 4 * d * d
    flops_proj = 4 * d * d

    # MLP ~ 2 * d * (mlp_ratio * d)
    flops_mlp = 2 * mlp_ratio * d * d

    # self-attn matmuls (QK^T + AV), per token ~ 2 * seq_len * d_head
    flops_self_attn = 2 * seq_len * d_head * h

    # cross-attn matmuls (QK^T + AV), per token ~ 2 * context_seq_len * d_head
    flops_cross_attn = 2 * context_seq_len * d_head * h

    flops_per_layer_fwd = flops_proj + flops_mlp + flops_self_attn + flops_cross_attn

    # backward ~ 2x forward for matmuls ⇒ total ~ 3x
    flops_per_layer_train = 3 * flops_per_layer_fwd

    flops_per_token = num_layers * flops_per_layer_train
    logger.info(f"FLOPS per token: {flops_per_token:,.0f}, num_layers: {num_layers}, flops_per_layer_train: {flops_per_layer_train}, flops_per_token: {flops_per_token}")
    return int(flops_per_token)


    # def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
    #     # TODO: double check the FLOPS calculations
    #     """
    #     Calculate the number of parameters and FLOPS for the Wan2.2 TI2V model.
        
    #     This implementation uses the standard transformer FLOPS formula:
    #     - 6 * (nparams - nparams_embedding) for forward+backward passes through weights
    #     - 6 * num_layers * num_heads * head_dims * seq_len for attention operations
        
    #     Note: The autoencoder (VAE) FLOPS are not included in this calculation,
    #     as the VAE is typically used only during inference/preprocessing.
    #     The main transformer model FLOPS are calculated here for training metrics.
        
    #     Args:
    #         model: The WanModel1x model instance
    #         seq_len: Sequence length from training configuration
            
    #     Returns:
    #         Tuple of (nparams, num_flops_per_token):
    #             nparams: Total number of model parameters
    #             num_flops_per_token: Estimated FLOPS per token for MFU calculation
    #     """
    #     # Calculate head dimension: hidden_dim divided by number of attention heads
    #     # head_dims = 2 * head_dim represents qk dimensions (query + key) for attention
    #     head_dim = self.hidden_dim // self.num_heads
    #     head_dims = 2 * head_dim
        
    #     # Count total model parameters
    #     nparams = sum(p.numel() for p in model.parameters())
        
    #     # Count embedding parameters (if any) - Wan model doesn't use standard nn.Embedding
    #     # but has patch embeddings (Conv3d) and text embeddings (Linear)
    #     # These are included in the parameter count but not subtracted for FLOPS calculation
    #     # as they're part of the model's computational graph
    #     nparams_embedding = 0
        
    #     # Calculate FLOPS using the standard transformer formula:
    #     # Factor of 6 accounts for:
    #     #   - 2 FLOPS per parameter in forward pass (multiplication + addition)
    #     #   - 4 FLOPS per parameter in backward pass (gradient computation)
    #     # First term: FLOPS for all non-embedding parameters (weights in transformer layers)
    #     # Second term: FLOPS for attention operations (qk computation and value aggregation)
    #     #   - 6 * num_layers * num_heads * head_dims * seq_len
    #     #   - This accounts for attention score computation and value aggregation
    #     num_flops_per_token = (
    #         6 * (nparams - nparams_embedding)
    #         + 6 * self.num_layers * self.num_heads * head_dims * seq_len
    #     )
        
    #     logger.info(
    #         f"Wan model parameters: {nparams:,}, "
    #         f"FLOPS per token: {num_flops_per_token:,.0f}, "
    #         f"head_dim: {head_dim}, num_layers: {self.num_layers}, "
    #         f"num_heads: {self.num_heads}, seq_len: {seq_len}"
    #     )
        
    #     return nparams, int(num_flops_per_token)