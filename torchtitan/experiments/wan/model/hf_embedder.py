# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from torch import nn, Tensor
from transformers import T5EncoderModel


class WanEmbedder(nn.Module):
    """
    Wrapper for T5 encoder model for Wan model text encoding.
    
    This class provides a unified interface for loading and using T5 encoder models
    with FSDP compatibility.
    """
    def __init__(self, version: str, random_init=False, **hf_kwargs):
        super().__init__()

        if random_init:
            # Initialize T5 model with random weights for test purpose only
            self.hf_module = T5EncoderModel._from_config(
                T5EncoderModel.config_class.from_pretrained(
                    os.path.join(version, "config.json"), **hf_kwargs
                )
            )
        else:
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                version, **hf_kwargs
            )

        self.hf_module = self.hf_module.eval().requires_grad_(False)
        # This is to make sure the encoder works with FSDP
        self.make_parameters_contiguous()

    def make_parameters_contiguous(self):
        """Make all non-contiguous parameters contiguous to avoid FSDP issues."""
        for name, param in self.hf_module.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    def forward(self, batch_tokens: Tensor) -> Tensor:
        """
        Forward pass through the T5 encoder.
        
        Args:
            batch_tokens: Token IDs tensor of shape [bsz, seq_len]
            
        Returns:
            Hidden states tensor of shape [bsz, seq_len, hidden_dim]
        """
        outputs = self.hf_module(
            input_ids=batch_tokens.to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        # T5EncoderModel returns BaseModelOutput with last_hidden_state attribute
        return outputs.last_hidden_state
