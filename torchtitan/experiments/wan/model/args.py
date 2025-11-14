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
        # TODO(jianiw): Add the number of flops for the autoencoder
        nparams = sum(p.numel() for p in model.parameters())
        logger.info(f"nparams: {nparams}")
        logger.warning(
            "get_nparams_and_flops() is not yet implemented for the Wan2.2 TI2V model. "
            "Returning placeholder value of 1 for num_flops_per_token. "
            "MFU and TFLOPs metrics will be incorrect and should be ignored."
        )
        # Return 1 as placeholder to satisfy assertion in metrics_processor.log()
        # This allows training to proceed, but MFU/TFLOPs calculations will be wrong
        # The actual loss computation in wan_video_1x.py doesn't use this value
        return nparams, 1
