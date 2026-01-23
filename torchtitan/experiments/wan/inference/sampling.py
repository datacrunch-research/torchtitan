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
from torchtitan.experiments.wan.inference.flow_match_scheduler import FlowMatchScheduler
from torchtitan.experiments.wan.inference.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler,
)
from torchtitan.experiments.wan.model.hf_embedder import WanEmbedder
from torchtitan.experiments.wan.model.model import WanModel

from torchtitan.experiments.wan.model.wan_vae import WanVideoVAE
from torchtitan.experiments.wan.utils import (
    # create_position_encoding_for_latents,
    generate_noise_latent,
    # pack_latents,
    preprocess_data,
    # unpack_latents,
)

# from torchtitan.tools.logging import logger


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
    precomputed_t5_embedding_null: Optional[Tensor] = None,
) -> torch.Tensor:
    """
    Generate video from conditioning frames using the Wan diffusion model.

    This function implements the full video generation pipeline:
    1. Preprocess input data (encode video frames, get text embeddings)
    2. Initialize noise latents
    3. Apply conditioning frames to the latent
    4. Run the denoising loop with FlowMatchScheduler
    5. Decode latents to video frames

    Args:
        device: Target device for computation
        dtype: Target dtype (typically bfloat16)
        job_config: Job configuration containing validation settings
        model: The WanModel diffusion model
        input_dict: Dictionary containing:
            - video_frames: Input video frames for conditioning
            - robot_states: Robot state information
            - t5_tokens: Text tokens for encoding
            - num_cond_frames: Number of conditioning frames
        wan_video_vae: VAE for encoding/decoding
        t5_tokenizer: Optional T5 tokenizer
        t5_encoder: Optional T5 encoder
        precomputed_t5_embedding: Optional precomputed T5 embedding
        precomputed_t5_embedding_null: Optional precomputed negative prompt embedding for CFG

    Returns:
        Generated video tensor with shape (B, C, T, H, W)
    """
    # Get image dimensions aligned to 16
    img_height = job_config.training.img_size
    img_width = job_config.training.img_size

    # Get validation config
    enable_classifier_free_guidance = (
        job_config.validation.enable_classifier_free_guidance
    )
    cfg_scale = job_config.validation.classifier_free_guidance_scale
    num_inference_steps = job_config.validation.denoising_steps
    num_cond_frames = input_dict["num_cond_frames"]

    # cfg_str = cfg_scale if enable_classifier_free_guidance else 'disabled'
    # logger.info(f"Generating video with {num_inference_steps} steps, CFG={cfg_str}")

    # Preprocess data: encode video frames and get text embeddings
    batch = preprocess_data(
        device=device,
        dtype=dtype,
        wan_video_vae=wan_video_vae,
        t5_encoder=t5_encoder,
        precomputed_t5_embedding=precomputed_t5_embedding,
        batch=input_dict,
    )

    # Get the conditioning latents from the encoded video
    video_latents = batch["latents"]  # Shape: (B, z_dim, T_latent, H_latent, W_latent)
    t5_encodings = batch["t5_encodings"]  # Shape: (B, seq_len, hidden_dim)
    robot_states = input_dict.get("robot_states", None)
    if robot_states is not None:
        robot_states = robot_states.to(device=device, dtype=dtype)

    # bsz = video_latents.shape[0]
    z_dim = video_latents.shape[1]
    num_frames = input_dict["video_frames"].shape[1]  # Original number of frames

    # logger.info(f"Video latents shape: {video_latents.shape}")
    # logger.info(f"T5 encodings shape: {t5_encodings.shape}")

    # TODO: add to the configs the hyps for the sampler such as sigma, sample_solver, etc.
    # Run denoising
    latents = denoise(
        device=device,
        dtype=dtype,
        model=model,
        video_latents=video_latents,
        t5_encodings=t5_encodings,
        robot_states=robot_states,
        num_cond_frames=num_cond_frames,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale if enable_classifier_free_guidance else 1.0,
        z_dim=z_dim,
        height=img_height,
        width=img_width,
        num_frames=num_frames,
        upsampling_factor=wan_video_vae.upsampling_factor,
        t5_encodings_null=precomputed_t5_embedding_null,
    )

    # Decode latents to video
    # logger.info(f"Decoding latents with shape: {latents.shape}")
    video = wan_video_vae.decode(latents, device=device)
    # logger.info(f"Decoded video shape: {video.shape}")

    return video


def denoise(
    device: torch.device,
    dtype: torch.dtype,
    model: WanModel,
    video_latents: torch.Tensor,
    t5_encodings: torch.Tensor,
    robot_states: Optional[torch.Tensor],
    num_cond_frames: int,
    num_inference_steps: int,
    cfg_scale: float,
    z_dim: int,
    height: int,
    width: int,
    num_frames: int,
    upsampling_factor: int,
    sigma_shift: float = 5.0,
    sample_solver: str = "unipc",
    t5_encodings_null: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Denoise latents using the FlowMatchScheduler.

    This implements the denoising loop following the original Wan TI2V approach:
    1. Initialize noise latents
    2. Create conditioning mask (0 for cond frames, 1 for gen frames)
    3. Apply conditioning frames to the latent using mask
    4. For each timestep:
       - Pass num_cond_frames to model (model creates per-token timesteps internally)
       - Run model forward pass
       - Apply CFG if enabled
       - Update latents with scheduler step
       - Reapply conditioning frames using mask

    Args:
        device: Target device
        dtype: Target dtype
        model: WanModel diffusion model
        video_latents: Encoded video latents for conditioning (B, z_dim, T, H, W)
        t5_encodings: Text embeddings (B, seq_len, hidden_dim)
        robot_states: Optional robot state tensor
        num_cond_frames: Number of conditioning frames
        num_inference_steps: Number of denoising steps
        cfg_scale: Classifier-free guidance scale (1.0 = no CFG)
        z_dim: Latent channel dimension
        img_height: Image height in pixels
        img_width: Image width in pixels
        num_frames: Number of video frames
        upsampling_factor: VAE upsampling factor
        sigma_shift: Shift parameter for FlowMatchScheduler
        t5_encodings_null: Negative prompt embeddings for CFG unconditional pass.
                          If None and cfg_scale != 1.0, uses same as t5_encodings.

    Returns:
        Denoised latents with shape (B, z_dim, T_latent, H_latent, W_latent)
    """
    bsz = video_latents.shape[0]

    # Initialize scheduler
    if sample_solver == "unipc":
        # Use UniPC scheduler (same as original Wan2.2) for better quality
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(num_inference_steps, device=device, shift=sigma_shift)
        timesteps = scheduler.timesteps
    else:
        # Use simple FlowMatchScheduler (faster but lower quality)
        scheduler = FlowMatchScheduler(
            shift=sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        timesteps = scheduler.timesteps

    # Initialize noise latents with same shape as video_latents
    noise = generate_noise_latent(
        bsz=bsz,
        num_frames=num_frames,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        z_dim=z_dim,
    )

    # Calculate number of conditioning latent frames
    # TODO: this may change when we will move to support also WanVAE-2.1
    num_cond_latents = (num_cond_frames - 1) // 4 + 1

    # ========================================================================
    # OLD APPROACH (simple slice assignment, no mask):
    # ========================================================================
    # # Initialize with random noise, then set conditioning
    # latents = torch.randn_like(video_latents)
    # cond_idxs = 1 + num_cond_frames // 4
    # latents[:, :, 0:cond_idxs] = video_latents[:, :, 0:cond_idxs]
    # ========================================================================

    # ========================================================================
    # NEW APPROACH (following original Wan TI2V with mask):
    # ========================================================================
    # Create conditioning mask: 0 for conditioning frames, 1 for frames to generate
    # Shape: (B, C, T, H, W)
    mask = torch.ones_like(noise)
    mask[:, :, :num_cond_latents, :, :] = 0.0

    # Initialize latents: conditioning frames from encoded video, rest is noise
    # latents = (1 - mask) * video_latents + mask * noise
    latents = (1.0 - mask) * video_latents + mask * noise

    # Compute per-token timesteps externally (like Wan2.2 video2video.py)
    # mask[:, :, :num_cond_latents] = 0, rest = 1
    # After spatial downsampling by 2 (patch_size), we get per-patch timesteps
    # Take first channel of mask, downsample spatially by 2
    latent_t, latent_h, latent_w = (
        video_latents.shape[2],
        video_latents.shape[3],
        video_latents.shape[4],
    )
    seq_len = latent_t * (latent_h // 2) * (latent_w // 2)

    # Create mask for per-token timesteps: shape [T, H//2, W//2]
    ts_mask = torch.ones(
        latent_t, latent_h // 2, latent_w // 2, device=device, dtype=dtype
    )
    ts_mask[:num_cond_latents, :, :] = 0.0

    # Denoising loop
    for progress_id, timestep in enumerate(timesteps):
        # Compute per-token timesteps using mask (like Wan2.2)
        # Conditioning tokens get 0, generation tokens get current timestep
        per_token_ts = (ts_mask * timestep).flatten()  # [seq_len]
        # Pad if needed
        if per_token_ts.size(0) < seq_len:
            per_token_ts = torch.cat(
                [
                    per_token_ts,
                    per_token_ts.new_ones(seq_len - per_token_ts.size(0)) * timestep,
                ]
            )
        per_token_ts = per_token_ts.unsqueeze(0).expand(bsz, -1)  # [B, seq_len]

        # Forward pass - conditional
        # Pass per-token timesteps directly (model should NOT recompute them)
        noise_pred_cond = model(
            x=latents,
            timesteps=per_token_ts,  # Per-token timesteps [B, seq_len]
            context=t5_encodings,
            robot_states=robot_states,
            num_cond_latents=None,  # Don't use internal per-token timestep computation
        )

        # Apply classifier-free guidance if scale != 1.0
        if cfg_scale != 1.0:
            # Use negative prompt embedding for unconditional pass (like Wan2.2)
            # If no negative prompt provided, fall back to same embeddings
            context_uncond = (
                t5_encodings_null if t5_encodings_null is not None else t5_encodings
            )
            noise_pred_uncond = model(
                x=latents,
                timesteps=per_token_ts,  # Use same per-token timesteps
                context=context_uncond,  # Use negative prompt embedding
                robot_states=None,  # No robot conditioning for unconditional
                num_cond_latents=None,
            )
            # CFG formula: pred = pred_uncond + scale * (pred_cond - pred_uncond)
            noise_pred = noise_pred_uncond + cfg_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred = noise_pred_cond

        # Scheduler step: update latents
        if sample_solver == "unipc":
            # UniPC scheduler expects different API
            step_output = scheduler.step(
                noise_pred,
                timestep,
                latents,
                return_dict=False,
            )
            latents = (
                step_output[0]
                if isinstance(step_output, tuple)
                else step_output.prev_sample
            )
        else:
            # Simple FlowMatchScheduler
            latents = scheduler.step(noise_pred, timesteps[progress_id], latents)

        # ====================================================================
        # OLD APPROACH (simple slice re-assignment):
        # ====================================================================
        # latents[:, :, 0:cond_idxs] = video_latents[:, :, 0:cond_idxs]
        # ====================================================================

        # ====================================================================
        # NEW APPROACH (mask-based re-application):
        # ====================================================================
        # Reapply conditioning frames using mask
        # Keep encoded conditioning frames fixed, only update generation frames
        latents = (1.0 - mask) * video_latents + mask * latents

    return latents


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
    # logger.info(f"Saving video to {output_dir}/{name}")
    # logger.info(f"Video shape: {video.shape}, dtype: {video.dtype}, device: {video.device}")

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
