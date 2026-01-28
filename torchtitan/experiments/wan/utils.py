# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torch import Tensor

from torchtitan.tools.logging import logger

from .model.hf_embedder import WanEmbedder
from .model.wan_vae import WanVideoVAE


def encode_t5_wan22_style(
    t5_tokens: Tensor,
    t5_encoder: WanEmbedder,
    pad_token_id: int,
    seq_len: int = 512,
) -> Tensor:
    """
    Encode tokens with T5 matching Wan2.2's behavior:
    1. Run T5 encoder
    2. Trim output to actual sequence length (non-padding tokens)
    3. Pad with ZEROS to seq_len

    This is important because Wan2.2's WanModel expects zero-padded context,
    not T5's padding token embeddings.

    Args:
        t5_tokens: Token IDs tensor [B, seq_len]
        t5_encoder: WanEmbedder (HuggingFace T5 encoder)
        pad_token_id: T5 padding token ID
        seq_len: Target sequence length to pad to (default 512)

    Returns:
        T5 embeddings with shape [B, seq_len, hidden_dim], zero-padded
    """
    device = t5_tokens.device
    bsz = t5_tokens.shape[0]

    # Run T5 encoder
    embeddings = t5_encoder(t5_tokens)  # [B, input_seq_len, hidden_dim]
    hidden_dim = embeddings.shape[-1]

    # Get attention mask (non-padding positions)
    attention_mask = t5_tokens != pad_token_id  # [B, input_seq_len]

    # Process each item in batch
    result = []
    for i in range(bsz):
        actual_len = attention_mask[i].sum().item()  # Number of actual tokens

        # Trim to actual length
        trimmed = embeddings[i, :actual_len, :]  # [actual_len, hidden_dim]

        # Pad with zeros to seq_len
        if actual_len < seq_len:
            padding = torch.zeros(
                seq_len - actual_len, hidden_dim, device=device, dtype=embeddings.dtype
            )
            padded = torch.cat([trimmed, padding], dim=0)  # [seq_len, hidden_dim]
        else:
            padded = trimmed[:seq_len]  # Truncate if too long

        result.append(padded)

    return torch.stack(result, dim=0)  # [B, seq_len, hidden_dim]


def preprocess_data(
    # arguments from the recipe
    device: torch.device,
    dtype: torch.dtype,
    *,
    # arguments from the config
    wan_video_vae: WanVideoVAE,
    t5_encoder: Optional[WanEmbedder] = None,
    batch: dict[str, Tensor],
    precomputed_t5_embedding: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """
    Take a batch of inputs and encoder as input and return a batch of preprocessed data.

    Args:
        device (torch.device): device to do preprocessing on
        dtype (torch.dtype): data type to do preprocessing in
        wan_video_vae (WanVideoVAE): Video VAE to use for preprocessing
        t5_encoder (HFEmbedder): T5EncoderModel to use for preprocessing
        batch (dict[str, Tensor]): batch of data to preprocess. Tensor shape: [bsz, ...]
        precomputed_t5_embedding (Optional[Tensor]): Precomputed T5 embedding for empty string [seq_len, hidden_dim]

    Returns:
        dict[str, Tensor]: batch of preprocessed data
    """

    t5_tokens = batch["t5_tokens"].squeeze(1).to(device=device, dtype=torch.int)

    # Check if we can use precomputed embeddings (when all tokens are empty strings)
    # This allows T5 encoder to be offloaded since we only encode ""
    bsz = t5_tokens.shape[0]

    # Use precomputed embeddings if available and token sequence lengths match
    # For the 1x-wmds dataset, we always use empty strings, so this optimization applies
    # T5 uses precomputed empty string ("") embeddings to avoid encoder forward passes
    if precomputed_t5_embedding is not None:
        # Check if token sequence lengths match precomputed embeddings
        # If they match, we can use precomputed embeddings (saves encoder forward passes)
        t5_seq_len_match = t5_tokens.shape[1] == precomputed_t5_embedding.shape[0]

        if t5_seq_len_match:
            # Use precomputed T5 embedding for empty string ("")
            # Expand precomputed T5 embedding to batch size: [seq_len, hidden_dim] -> [bsz, seq_len, hidden_dim]
            logger.debug(
                f"Using precomputed T5 embedding (empty string) for batch size {bsz}"
            )
            t5_text_encodings = (
                precomputed_t5_embedding.unsqueeze(0)
                .expand(bsz, -1, -1)
                .to(device=device, dtype=dtype)
            )
        else:
            # Sequence length doesn't match, need encoder to compute
            if t5_encoder is None:
                raise RuntimeError(
                    "T5 encoder is required but was deleted. Sequence length mismatch: "
                    f"tokens have {t5_tokens.shape[1]} tokens but precomputed embedding has "
                    f"{precomputed_t5_embedding.shape[0]} tokens."
                )
            logger.debug(
                "T5 sequence length mismatch, computing embeddings with encoder"
            )
            # Use Wan2.2-style encoding: trim to actual length, pad with zeros
            pad_token_id = t5_encoder.pad_token_id
            t5_text_encodings = encode_t5_wan22_style(
                t5_tokens=t5_tokens,
                t5_encoder=t5_encoder,
                pad_token_id=pad_token_id,
                seq_len=512,
            )
    else:
        # No precomputed embeddings available, need encoder to compute
        if t5_encoder is None:
            raise RuntimeError(
                "T5 encoder is required but was deleted. Precomputed embeddings are not available."
            )
        # Use Wan2.2-style encoding: trim to actual length, pad with zeros
        # This is important because Wan2.2's WanModel expects zero-padded context
        pad_token_id = t5_encoder.pad_token_id
        t5_text_encodings = encode_t5_wan22_style(
            t5_tokens=t5_tokens,
            t5_encoder=t5_encoder,
            pad_token_id=pad_token_id,
            seq_len=512,
        )

    # Check if latents are already pre-loaded (from LatentDatasetWrapper or ValidationLatentDatasetWrapper)
    if "latents" in batch and batch["latents"] is not None:
        # Use pre-loaded latents - skip VAE encoding entirely
        logger.debug("Using pre-loaded latents, skipping VAE encoding")
        video_latents = batch["latents"].to(device=device, dtype=dtype)
    else:
        # First move to GPU (keeping original dtype to minimize CPU-GPU transfer)
        # Then convert dtype on GPU for better performance
        videos = batch["video_frames"].to(device=device, dtype=dtype)
        # Permute from (B, T, H, W, C) to (B, T, C, H, W)
        videos = videos.permute(0, 1, 4, 2, 3)
        # Normalize video frames from [0, 255] range to [-1, 1] range
        max_value = 1.0
        min_value = -1.0
        videos = videos * ((max_value - min_value) / 255.0) + min_value
        videos = videos.transpose(1, 2)  # (B, T, C, H, W) -> (B, C, T, H, W)

        video_latents = wan_video_vae.encode(
            videos,  # Batched tensor (B, C, T, H, W)
            device=device,
            tiled=False,
        )

    batch["latents"] = video_latents.to(device=device, dtype=dtype)
    batch["t5_encodings"] = t5_text_encodings.to(dtype)
    return batch


def generate_noise_latent(
    bsz: int,
    num_frames: int,
    height: int,
    width: int,
    device: str | torch.device,
    dtype: torch.dtype,
    z_dim: int = 48,
    latent_ratio: int = 16,
    seed: Optional[int] = None,
) -> Tensor:
    """Initialize the noise latent tensor for the Wan model.

    Args:
        bsz (int): batch_size.
        num_frames (int): number of frames in the video.
        height (int): The height of the input video frames.
        width (int): The width of the input video frames.
        device (str | torch.device): The device to create the tensor on.
        dtype (torch.dtype): The dtype for the noise tensor.
        z_dim (int, optional): Latent channel dimension. Defaults to 48.
        latent_ratio (int, optional): Downsampling ratio for spatial dimensions.
            Defaults to 16 (e.g., 512 x 512 -> 32 x 32 latent).
        seed (int, optional): Random seed for reproducibility. If None, uses
            default random state.

    Returns:
        Tensor: The noise latents with shape [B, z_dim, T_latent, H_latent, W_latent]
            where:
            - B = bsz (batch size)
            - T_latent = (num_frames - 1) // 4 + 1 (temporal downsampling)
            - H_latent = height // latent_ratio
            - W_latent = width // latent_ratio
    """
    # Compute the latent dimensions
    latent_time = (num_frames - 1) // 4 + 1
    latent_height = height // latent_ratio
    latent_width = width // latent_ratio

    shape = torch.Size([bsz, z_dim, latent_time, latent_height, latent_width])

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    else:
        noise = torch.randn(shape, device=device, dtype=dtype)

    return noise


#


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
