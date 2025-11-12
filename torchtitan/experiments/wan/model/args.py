# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torch import nn


from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger
from .wan_vae import WanVAEParams


@dataclass
class WanModelArgs(BaseModelArgs):
    in_channels: int = 64
    out_channels: int = 64
    vec_in_dim: int = 768
    context_in_dim: int = 512
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: tuple = (16, 56, 56)
    theta: int = 10_000
    qkv_bias: bool = True
    wan_video_vae_params: WanVAEParams = field(default_factory=lambda: WanVAEParams(vae_type="38", z_dim=48))

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        # TODO(jianiw): Add the number of flops for the autoencoder
        nparams = sum(p.numel() for p in model.parameters())
        logger.warning("get_nparams_and_flops() is not yet implemented for the Wan2.2 TI2V model.")
        return nparams, -1
