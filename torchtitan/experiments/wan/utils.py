# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torch import Tensor

from .model.hf_embedder import WanEmbedder
from .model.wan_vae import WanVideoVAE
# from icecream import ic
from torchtitan.tools.logging import logger


def preprocess_data(
    # arguments from the recipe
    device: torch.device,
    dtype: torch.dtype,
    *,
    # arguments from the config
    wan_video_vae: WanVideoVAE,
    clip_encoder: WanEmbedder,
    t5_encoder: WanEmbedder,
    batch: dict[str, Tensor],
    precomputed_t5_embedding: Optional[Tensor] = None,
    precomputed_clip_embedding: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """
    Take a batch of inputs and encoder as input and return a batch of preprocessed data.

    Args:
        device (torch.device): device to do preprocessing on
        dtype (torch.dtype): data type to do preprocessing in
        autoencoer(AutoEncoder): autoencoder to use for preprocessing
        clip_encoder (HFEmbedder): CLIPTextModel to use for preprocessing
        t5_encoder (HFEmbedder): T5EncoderModel to use for preprocessing
        batch (dict[str, Tensor]): batch of data to preprocess. Tensor shape: [bsz, ...]
        precomputed_t5_embedding (Optional[Tensor]): Precomputed T5 embedding for empty string [seq_len, hidden_dim]
        precomputed_clip_embedding (Optional[Tensor]): Precomputed CLIP embedding for empty string [seq_len, hidden_dim]

    Returns:
        dict[str, Tensor]: batch of preprocessed data
    """

    clip_tokens = batch["clip_tokens"].squeeze(1).to(device=device, dtype=torch.int)
    t5_tokens = batch["t5_tokens"].squeeze(1).to(device=device, dtype=torch.int)

    # Check if we can use precomputed embeddings (when all tokens are empty strings)
    # This allows T5 and CLIP encoders to be offloaded since we only encode ""
    bsz = clip_tokens.shape[0]
    
    # Use precomputed embeddings if available and token sequence lengths match
    # For the 1x-wmds dataset, we always use empty strings, so this optimization applies
    if precomputed_t5_embedding is not None and precomputed_clip_embedding is not None:
        # Check if token sequence lengths match precomputed embeddings
        # If they match, we can use precomputed embeddings (saves encoder forward passes)
        t5_seq_len_match = t5_tokens.shape[1] == precomputed_t5_embedding.shape[0]
        clip_seq_len_match = clip_tokens.shape[1] == precomputed_clip_embedding.shape[0]
        
        if t5_seq_len_match:
            # Expand precomputed T5 embedding to batch size: [seq_len, hidden_dim] -> [bsz, seq_len, hidden_dim]
            t5_text_encodings = precomputed_t5_embedding.unsqueeze(0).expand(bsz, -1, -1).to(device=device, dtype=dtype)
        else:
            # Sequence length doesn't match, compute normally
            t5_text_encodings = t5_encoder(t5_tokens)
        
        if clip_seq_len_match:
            # Expand precomputed CLIP embedding to batch size: [seq_len, hidden_dim] -> [bsz, seq_len, hidden_dim]
            clip_text_encodings = precomputed_clip_embedding.unsqueeze(0).expand(bsz, -1, -1).to(device=device, dtype=dtype)
        else:
            # Sequence length doesn't match, compute normally
            clip_text_encodings = clip_encoder(clip_tokens)
    else:
        # No precomputed embeddings available, compute normally
        clip_text_encodings = clip_encoder(clip_tokens)
        t5_text_encodings = t5_encoder(t5_tokens)

    # Move videos to device and convert to proper dtype
    # Videos come from dataloader as (batch_size, num_frames, height, width, channels)
    videos = batch["video_frames"].to(device=device, dtype=dtype)
    
    # Permute from (B, T, H, W, C) to (B, T, C, H, W)
    videos = videos.permute(0, 1, 4, 2, 3)
    logger.info(videos.shape)
    logger.info(videos.device)
    logger.info(videos.dtype)
    # Normalize video frames from [0, 255] range to [-1, 1] range
    # This is required because the VAE expects input in [-1, 1] range
    max_value = 1.0
    min_value = -1.0
    videos = videos * ((max_value - min_value) / 255.0) + min_value
    logger.info(videos.device)
    logger.info(videos.dtype)
    # Transpose from (B, T, C, H, W) to (B, C, T, H, W) for VAE encoding
    # The VAE encode method expects a batched tensor of shape (B, C, T, H, W)
    videos = videos.transpose(1, 2)  # (B, T, C, H, W) -> (B, C, T, H, W)
    logger.info(f"After transpose: {videos.shape}")
    
    # Encode videos to latents using the WAN Video VAE
    # The encode method processes the entire batch at once for better performance
    video_latents = wan_video_vae.encode(
        videos,  # Batched tensor (B, C, T, H, W)
        device=device,
        tiled=False,
    )
    batch["latents"] = video_latents.to(device=device, dtype=dtype)


    batch["clip_encodings"] = clip_text_encodings.to(dtype)
    batch["t5_encodings"] = t5_text_encodings.to(dtype)

    return batch


def generate_noise_latent(
    bsz: int,
    height: int,
    width: int,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int | None = None,
) -> Tensor:
    """Generate noise latents for the Flux flow model. The random seed will be set at the beginning of training.

    Args:
        bsz (int): batch_size.
        height (int): The height of the image.
        width (int): The width of the image.
        device (str | torch.device): The device to use.
        dtype (torch.dtype): The dtype to use.

    Returns:
        Tensor: The noise latents.
            Shape: [num_samples, LATENT_CHANNELS, height // IMG_LATENT_SIZE_RATIO, width // IMG_LATENT_SIZE_RATIO]

    """
    LATENT_CHANNELS, IMAGE_LATENT_SIZE_RATIO = 16, 8
    return torch.randn(
        bsz,
        LATENT_CHANNELS,
        height // IMAGE_LATENT_SIZE_RATIO,
        width // IMAGE_LATENT_SIZE_RATIO,
        dtype=dtype,
    ).to(device)


def create_position_encoding_for_latents(
    bsz: int, latent_height: int, latent_width: int, position_dim: int = 3
) -> Tensor:
    """
    Create the packed latents' position encodings for the Flux flow model.

    Args:
        bsz (int): The batch size.
        latent_height (int): The height of the latent.
        latent_width (int): The width of the latent.

    Returns:
        Tensor: The position encodings.
            Shape: [bsz, (latent_height // PATCH_HEIGHT) * (latent_width // PATCH_WIDTH), POSITION_DIM)
    """
    PATCH_HEIGHT, PATCH_WIDTH = 2, 2

    height = latent_height // PATCH_HEIGHT
    width = latent_width // PATCH_WIDTH

    position_encoding = torch.zeros(height, width, position_dim)

    row_indices = torch.arange(height)
    position_encoding[:, :, 1] = row_indices.unsqueeze(1)

    col_indices = torch.arange(width)
    position_encoding[:, :, 2] = col_indices.unsqueeze(0)

    # Flatten and repeat for the full batch
    # [height, width, 3] -> [bsz, height * width, 3]
    position_encoding = position_encoding.view(1, height * width, position_dim)
    position_encoding = position_encoding.repeat(bsz, 1, 1)

    return position_encoding


def pack_latents(x: Tensor) -> Tensor:
    """
    Rearrange video latents from (B, C, T, H, W) format into a sequence of patches.
    Packs spatial patches (2x2) while keeping temporal dimension separate.
    Equivalent to `einops.rearrange("b c t (h ph) (w pw) -> b (t h w) (c ph pw)")`.

    Args:
        x (Tensor): The unpacked video latents.
            Shape: [bsz, channels, temporal, latent_height, latent_width]

    Returns:
        Tensor: The packed latents.
            Shape: (bsz, (temporal * latent_height // 2 * latent_width // 2), channels * 4)
    """
    PATCH_HEIGHT, PATCH_WIDTH = 2, 2
    b, c, t, h, w = x.shape
    h_patches = h // PATCH_HEIGHT
    w_patches = w // PATCH_WIDTH

    # Pack spatial patches: (B, C, T, H, W) -> (B, C, T, H/2, W/2, 2, 2)
    x = x.unfold(3, PATCH_HEIGHT, PATCH_HEIGHT).unfold(4, PATCH_WIDTH, PATCH_WIDTH)
    # x is now (B, C, T, H/2, W/2, 2, 2)

    # Rearrange: (B, C, T, H/2, W/2, 2, 2) -> (B, T, H/2, W/2, C, 2, 2) -> (B, T*H/2*W/2, C*4)
    x = x.permute(0, 2, 3, 4, 1, 5, 6).contiguous()
    x = x.reshape(b, t * h_patches * w_patches, c * PATCH_HEIGHT * PATCH_WIDTH)
    
    return x


def unpack_latents(x: Tensor, latent_height: int, latent_width: int) -> Tensor:
    """
    Rearrange video latents from a sequence of patches back into (B, C, T, H, W) format.
    Unpacks spatial patches (2x2) while preserving temporal dimension.
    Equivalent to `einops.rearrange("b (t h w) (c ph pw) -> b c t (h ph) (w pw)")`.

    Args:
        x (Tensor): The packed latents.
            Shape: (bsz, (temporal * latent_height // 2 * latent_width // 2), channels * 4)
        latent_height (int): The height of the unpacked latents.
        latent_width (int): The width of the unpacked latents.

    Returns:
        Tensor: The unpacked video latents.
            Shape: [bsz, channels, temporal, latent_height, latent_width]
    """
    PATCH_HEIGHT, PATCH_WIDTH = 2, 2

    b, seq_len, c_ph_pw = x.shape
    h_patches = latent_height // PATCH_HEIGHT
    w_patches = latent_width // PATCH_WIDTH
    c = c_ph_pw // (PATCH_HEIGHT * PATCH_WIDTH)
    t = seq_len // (h_patches * w_patches)

    # [b, t*h*w, c*ph*pw] -> [b, t, h, w, c, ph, pw]
    x = x.reshape(b, t, h_patches, w_patches, c, PATCH_HEIGHT, PATCH_WIDTH)

    # [b, t, h, w, c, ph, pw] -> [b, c, t, h, ph, w, pw]
    x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()

    # [b, c, t, h, ph, w, pw] -> [b, c, t, h*ph, w*pw]
    x = x.reshape(b, c, t, latent_height, latent_width)
    return x
