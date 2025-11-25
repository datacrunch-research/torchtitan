# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from typing import Any, Callable, Optional

import torch
import torchvision
from einops import rearrange
from PIL import ExifTags, Image

from torch import Tensor

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig

from torchtitan.experiments.wan.model.wan_vae import WanVideoVAE
from torchtitan.experiments.wan.model.hf_embedder import WanEmbedder
from torchtitan.experiments.wan.model.model import WanModel
from torchtitan.experiments.wan.utils import (
    create_position_encoding_for_latents,
    generate_noise_latent,
    pack_latents,
    preprocess_data,
    unpack_latents,
)
from torchtitan.tools.logging import logger


# ----------------------------------------
#       Util functions for Sampling
# ----------------------------------------


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


# ----------------------------------------
#       Sampling functions
# ----------------------------------------


def generate_video(
    device: torch.device,
    dtype: torch.dtype,
    job_config: JobConfig,
    model: WanModel,
    input_dict: dict[str, Any],
    wan_video_vae: WanVideoVAE,
    t5_tokenizer: Optional[BaseTokenizer] = None,
    t5_encoder: Optional[WanEmbedder] = None,
    precomputed_t5_embedding: Optional[Tensor] = None,
) -> torch.Tensor:
    """
    Sampling and save a single video from noise.
    For randomized noise generation, the random seed should already be set at the beginning of training.
    Since we will always use the local random seed on this rank, we don't need to pass in the seed again.
    """
    # TODO: TOCHECK if this is needed
    # allow for packing and conversion to latent space. Use the same resolution as training time.
    img_height = 16 * (job_config.training.img_size // 16)
    img_width = 16 * (job_config.training.img_size // 16)

    # TODO: add the cfg support
    enable_classifier_free_guidance = (
        job_config.validation.enable_classifier_free_guidance
    )
    
    logger.info(f"Input dict keys: {input_dict.keys()}")

    batch = preprocess_data(
        device=device,
        dtype=dtype,
        wan_video_vae=wan_video_vae,
        t5_encoder=t5_encoder,
        precomputed_t5_embedding=precomputed_t5_embedding,
        batch=input_dict,
    )

    # TODO: add classifier free guidance support
    # if enable_classifier_free_guidance:
    #     # num_images = len(prompt)

    #     # empty_t5_tokens = t5_tokenizer.encode("")
    #     # empty_t5_tokens = empty_t5_tokens.repeat(num_images, 1)

    #     empty_batch = preprocess_data(
    #         device=device,
    #         dtype=dtype,
    #         autoencoder=None,
    #         t5_encoder=t5_encoder,
    #         batch={
    #             "t5_tokens": empty_t5_tokens,
    #         },
    #     )

    video = denoise(
        device=device,
        dtype=dtype,
        model=model,
        input_dict=input_dict,
        num_cond_frames=input_dict["num_cond_frames"],
        img_width=img_width,
        img_height=img_height,
        denoising_steps=job_config.validation.denoising_steps,
        t5_encodings=batch["t5_encodings"],
        enable_classifier_free_guidance=enable_classifier_free_guidance,
        classifier_free_guidance_scale=job_config.validation.classifier_free_guidance_scale,
    )

    video = wan_video_vae.decode(video, device=device)
    return video


def denoise(
    device: torch.device,
    dtype: torch.dtype,
    model: WanModel,
    input_dict: dict[str, Any],
    num_cond_frames: int,
    img_width: int,
    img_height: int,
    denoising_steps: int,
    t5_encodings: torch.Tensor,
    enable_classifier_free_guidance: bool = False,
    empty_t5_encodings: torch.Tensor | None = None,
    classifier_free_guidance_scale: float | None = None,
) -> torch.Tensor:
    """
    Sampling images from noise using a given prompt, by running inference with trained Wan model.
    Save the generated images to the given output path.
    """
    bsz = t5_encodings.shape[0]
    latents = generate_noise_latent(bsz, img_height, img_width, device, dtype)
    # logger.info(f"Generated noise latents: {latents.shape}")

    _, latent_channels, latent_temp_dim, latent_height, latent_width = latents.shape
    # logger.info(f"input_dict Latent shape: {input_dict["latents"].shape}")
    # logger.info(f"latent shape: {latents.shape}")

    # Addining the conditioning part on the first 
    cond_idxs = 1 + (num_cond_frames-1) // 4 
    conditioning  = input_dict["latents"][:, :, 0:cond_idxs]

    # create denoising schedule
    timesteps = get_schedule(denoising_steps, latent_height * latent_width, shift=True)

    if enable_classifier_free_guidance:
        # Double batch size for CFG: [unconditional, conditional]
        latents = torch.cat([latents, latents], dim=0)
        t5_encodings = torch.cat([empty_t5_encodings, t5_encodings], dim=0)
        bsz *= 2

    # create positional encodings
    POSITION_DIM = 3
    latent_pos_enc = create_position_encoding_for_latents(
        bsz, latent_height, latent_width, POSITION_DIM
    ).to(latents)
    text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM).to(latents)

    # convert img-like latents into sequences of patches
    # latents = pack_latents(latents)

    # this is ignored for schnell
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((bsz,), t_curr, dtype=dtype, device=device)
        latents[:, :, 0:cond_idxs] = conditioning
        # logger.info(f"Latents shape after conditioning: {latents.shape}")

        pred = model(
            x=latents,
            timesteps=t_vec,
            context=t5_encodings,
            robot_states=input_dict["robot_states"],
        )

        # logger.info(f"Pred shape: {pred.shape}")
        # logger.info(f"Pred dtype: {pred.dtype}")
        # logger.info(f"Pred device: {pred.device}")

        if enable_classifier_free_guidance:
            pred_u, pred_c = pred.chunk(2)
            pred = pred_u + classifier_free_guidance_scale * (pred_c - pred_u)

            # repeat along batch dimension to update both unconditional and conditional latents
            pred = pred.repeat(2, 1, 1)

        latents[:, :, cond_idxs:] = latents[:, :, cond_idxs:] + (t_prev - t_curr) * pred[:, :, cond_idxs:]

    # take the conditional latents for the final result
    if enable_classifier_free_guidance:
        latents = latents.chunk(2)[1]

    # TODO: TOCHECK where this is done in the model/original code
    # convert sequences of patches into img-like latents
    # logger.info(f"Latents shape before unpacking: {latents.shape}")
    # latents = unpack_latents(latents, latent_height, latent_width)

    return latents


# TODO: Remove this function 
def save_image(
    name: str,
    output_dir: str,
    x: torch.Tensor,
    add_sampling_metadata: bool,
    prompt: str,
):
    # logger.info(f"Saving image to {output_dir}/{name}")
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, name)

    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;wan"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    exif_data[ExifTags.Base.Model] = name
    if add_sampling_metadata:
        exif_data[ExifTags.Base.ImageDescription] = prompt
    img.save(output_name, exif=exif_data, quality=95, subsampling=0)


def save_video(
    name: str,
    output_dir: str,
    video: torch.Tensor,
    add_sampling_metadata: bool,
):
    """
    Save a video tensor to an MP4 file.
    
    Args:
        name: Output filename (should end with .mp4)
        output_dir: Directory to save the video
        video: Video tensor with shape [batch, channels, frames, height, width] or [channels, frames, height, width]
               Values should be in range [-1, 1] (float32)
        add_sampling_metadata: Whether to add metadata (currently unused, kept for API compatibility)
    """
    logger.info(f"Saving video to {output_dir}/{name}")
    logger.info(f"Video shape: {video.shape}, dtype: {video.dtype}, device: {video.device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, name)
    
    # Remove batch dimension if present: [1, C, T, H, W] -> [C, T, H, W]
    if video.dim() == 5:
        video = video[0]
    
    # Clamp values to [-1, 1] range (required for proper conversion)
    video = video.clamp(-1, 1)
    
    # Convert from [-1, 1] to [0, 255] uint8
    # Formula: (video + 1.0) * 127.5 maps [-1, 1] to [0, 255]
    video = (video + 1.0) * 127.5
    video = video.clamp(0, 255).byte()
    
    # Rearrange from [C, T, H, W] to [T, H, W, C] for video writing
    video = rearrange(video, "c t h w -> t h w c")
    
    # Move to CPU if on GPU and convert to numpy array
    video_np = video.cpu().numpy()
    
    # Save video using torchvision
    # torchvision.io.write_video expects [T, H, W, C] format and uint8 values
    # fps: frames per second (default to 8 fps, adjust as needed)
    fps = 8.0
    torchvision.io.write_video(output_name, video_np, fps=fps, video_codec="libx264")
    logger.info(f"✓ Video saved successfully: {output_name}")