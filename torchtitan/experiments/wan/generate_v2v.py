# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone Video-to-Video Generation Script for Torchtitan

This script generates videos by conditioning on input video frames using the
pretrained Wan model, similar to generate_v2v.py in the Wan2.2 repository.

Use this to verify that weights and code are correct by comparing outputs.

Video loading matches torchtitan dataset behavior:
- Loads clip_length consecutive frames starting from start_frame
- Applies downsampling: 4->20 frames, 2->41 frames, 1->all frames

Example usage:
    python -m torchtitan.experiments.wan.generate_v2v \
        --input_video /path/to/video.mp4 \
        --weights_path assets/hf/Wan2.2-TI2V-5B \
        --vae_path /path/to/wan_vae.pt \
        --output_dir ./outputs \
        --clip_length 77 \
        --downsampled 4 \
        --num_cond_frames 5
"""

import argparse
import os
from datetime import datetime

import torch
import torchvision.transforms.functional as TF
from torchmetrics.functional.image import peak_signal_noise_ratio

# Use decord for video loading (same as torchtitan dataset)
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "1")
import decord

decord.bridge.set_bridge("torch")

from torchtitan.experiments.wan import wan_configs, WanModel
from torchtitan.experiments.wan.inference.sampling import denoise, save_video
from torchtitan.experiments.wan.model.hf_embedder import WanEmbedder
from torchtitan.experiments.wan.model.wan_vae import load_wan_vae
from torchtitan.tools.logging import init_logger, logger


def get_frame_indices(clip_length: int, downsampled: int):
    """
    Get frame indices matching torchtitan dataset logic.

    Args:
        clip_length: Number of consecutive frames to load (e.g., 77)
        downsampled: Downsampling factor (1, 2, or 4)

    Returns:
        List of frame indices to use
    """
    if downsampled == 4:
        # Every 4th frame + last frame: [0, 4, 8, ..., 72, 76] -> 20 frames
        frame_idxs = list(range(clip_length))[::4] + [clip_length - 1]
    elif downsampled == 2:
        # 41 frames uniformly spaced from 0 to clip_length-1
        num_output_frames = 41
        frame_idxs = [
            round(i * (clip_length - 1) / (num_output_frames - 1))
            for i in range(num_output_frames)
        ]
    elif downsampled == 1:
        # All frames
        frame_idxs = list(range(clip_length))
    else:
        raise ValueError(f"downsampled must be 1, 2, or 4, got {downsampled}")
    return frame_idxs


def load_video(
    video_path, clip_length=77, downsampled=4, start_frame=0, target_size=None
):
    """
    Load a video file using the same approach as torchtitan dataset.

    Takes consecutive frames from start_frame and applies downsampling,
    matching the RawVideoDataset behavior.

    Args:
        video_path: Path to video file
        clip_length: Number of consecutive frames to load (default: 77)
        downsampled: Downsampling factor - 1, 2, or 4 (default: 4)
        start_frame: Starting frame index (default: 0)
        target_size: Tuple of (H, W) to resize video to (optional)

    Returns:
        torch.Tensor: Video tensor of shape [B, T, H, W, C] with uint8 values in [0, 255]
    """
    # Load video with decord (same as torchtitan dataset)
    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)

    # Check we have enough frames
    end_frame = start_frame + clip_length
    if end_frame > total_frames:
        logger.warning(
            f"Video has {total_frames} frames, but need {end_frame} frames "
            f"(start={start_frame}, clip_length={clip_length}). "
            f"Adjusting start_frame to 0."
        )
        start_frame = 0
        if clip_length > total_frames:
            raise ValueError(
                f"Video has only {total_frames} frames, but clip_length={clip_length} required."
            )

    # Load consecutive frames
    frame_range = range(start_frame, start_frame + clip_length)
    video = vr.get_batch(list(frame_range))  # [T, H, W, C]

    # Apply downsampling (same logic as torchtitan dataset)
    frame_idxs = get_frame_indices(clip_length, downsampled)
    video = video[frame_idxs]

    logger.info(
        f"Loaded {len(frame_idxs)} frames from {clip_length} consecutive frames "
        f"(downsampled={downsampled}, start_frame={start_frame})"
    )

    # Resize if target size is specified
    if target_size is not None:
        target_h, target_w = target_size
        # Convert to [T, C, H, W] for resize
        video = video.permute(0, 3, 1, 2).float()
        video = TF.resize(video, [target_h, target_w], antialias=True)
        # Convert back to [T, H, W, C]
        video = video.permute(0, 2, 3, 1).to(torch.uint8)

    # Add batch dimension: [T, H, W, C] -> [B, T, H, W, C]
    video = video.unsqueeze(0)

    return video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video from input video using Torchtitan Wan model"
    )
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to input video file (mp4, avi, etc.)",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="assets/hf/Wan2.2-TI2V-5B",
        help="Path to pretrained Wan model weights",
    )
    parser.add_argument(
        "--vae_path", type=str, required=True, help="Path to Wan VAE checkpoint"
    )
    parser.add_argument(
        "--t5_encoder",
        type=str,
        default="google/umt5-xxl",
        help="T5 encoder model name or path",
    )
    parser.add_argument(
        "--model_flavor",
        type=str,
        default="wan-video",
        choices=list(wan_configs.keys()),
        help="Model configuration flavor",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save generated videos",
    )
    parser.add_argument(
        "--size", type=str, default="512", help="Target video size (height and width)"
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        default=77,
        help="Number of consecutive frames to load from video (default: 77, same as torchtitan dataset)",
    )
    parser.add_argument(
        "--downsampled",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help="Downsampling factor: 4->20 frames, 2->41 frames, 1->all frames (default: 4)",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Starting frame index in the video (default: 0)",
    )
    parser.add_argument(
        "--num_cond_frames",
        type=int,
        default=5,
        help="Number of conditioning frames. Should be 4n+1 (e.g., 5, 9, 13, 17). "
        "Default 5 is appropriate for downsampled=4 which gives 20 frames.",
    )
    parser.add_argument(
        "--denoising_steps", type=int, default=50, help="Number of denoising steps"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale (5.0 is default for Wan)",
    )
    parser.add_argument(
        "--sigma_shift", type=float, default=5.0, help="Sigma shift for scheduler"
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "euler"],
        help="Sampling solver: 'unipc' (higher quality, default) or 'euler' (faster)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no_psnr", action="store_true", default=False, help="Disable PSNR computation"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--prompt", type=str, default="", help="Text prompt for generation (optional)"
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help="Negative prompt for CFG (default: Wan2.2 standard negative prompt)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize logger
    init_logger()
    logger.info("Starting Torchtitan V2V generation script")
    logger.info(f"Args: {args}")

    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Parse target size
    target_size = int(args.size)
    target_size = (target_size, target_size)

    # Load input video (using torchtitan dataset approach)
    logger.info(f"Loading input video from: {args.input_video}")
    logger.info(
        f"clip_length={args.clip_length}, downsampled={args.downsampled}, start_frame={args.start_frame}"
    )
    video_frames = load_video(
        args.input_video,
        clip_length=args.clip_length,
        downsampled=args.downsampled,
        start_frame=args.start_frame,
        target_size=target_size,
    )
    logger.info(f"Input video shape: {video_frames.shape}")  # [B, T, H, W, C]

    # Get model configuration
    model_args = wan_configs[args.model_flavor]
    logger.info(f"Using model config: {args.model_flavor}")

    # Load VAE
    logger.info(f"Loading VAE from: {args.vae_path}")
    wan_video_vae = load_wan_vae(
        chkpt_path=args.vae_path,
        wan_vae_params=model_args.wan_video_vae_params,
        device=device,
        dtype=dtype,
    )
    logger.info("VAE loaded successfully")

    # Load T5 encoder
    logger.info(f"Loading T5 encoder: {args.t5_encoder}")
    t5_encoder = WanEmbedder(
        version=args.t5_encoder,
    ).to(device=device, dtype=dtype)
    logger.info("T5 encoder loaded successfully")

    # Load WanModel
    logger.info("=" * 80)
    logger.info("Loading WanModel (diffusion model)...")
    logger.info("=" * 80)
    model = WanModel(model_args)

    # Load pretrained weights
    logger.info(f"Loading pretrained weights from: {args.weights_path}")
    model.init_weights(pretrained_weights_path=args.weights_path)
    logger.info("Pretrained weights loaded successfully")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    logger.info(f"WanModel loaded and moved to {device} with dtype {dtype}")


    # Encode prompts with T5
    # IMPORTANT: Wan2.2 trims T5 output to actual sequence length, then pads with ZEROS
    # This is different from keeping the full T5 output with padding token embeddings
    logger.info("Encoding prompts with T5 encoder...")
    from torchtitan.experiments.wan.tokenizer import WanTokenizer

    # Build tokenizer
    t5_seq_len = 512  # Default max length (text_len in Wan2.2)
    t5_tokenizer = WanTokenizer(args.t5_encoder, max_length=t5_seq_len)

    def encode_prompt_like_wan22(
        prompt_text, tokenizer, encoder, device, dtype, seq_len=512
    ):
        """
        Encode prompt matching Wan2.2's T5EncoderModel behavior:
        1. Tokenize and get attention mask
        2. Run T5 encoder
        3. Trim output to actual sequence length (non-padding tokens)
        4. Pad with ZEROS to seq_len (like Wan2.2's WanModel does)
        """
        # Tokenize
        tokens = tokenizer.encode(prompt_text).to(device=device)  # [1, seq_len]

        # Get attention mask (non-padding positions)
        pad_token_id = tokenizer._tokenizer.pad_token_id
        attention_mask = tokens != pad_token_id  # [1, seq_len]
        actual_len = attention_mask.sum(dim=1).item()  # Number of actual tokens

        # Run T5 encoder
        embeddings = encoder(tokens)  # [1, seq_len, hidden_dim]

        # Trim to actual length, then pad with zeros (like Wan2.2)
        trimmed = embeddings[0, :actual_len, :]  # [actual_len, hidden_dim]
        hidden_dim = trimmed.shape[-1]

        # Pad with zeros to seq_len
        if actual_len < seq_len:
            padding = torch.zeros(
                seq_len - actual_len, hidden_dim, device=device, dtype=embeddings.dtype
            )
            padded = torch.cat([trimmed, padding], dim=0)  # [seq_len, hidden_dim]
        else:
            padded = trimmed[:seq_len]  # Truncate if too long

        # Add batch dimension back
        result = padded.unsqueeze(0).to(dtype=dtype)  # [1, seq_len, hidden_dim]

        logger.info(f"  Actual tokens: {actual_len}, padded to: {seq_len}")
        return result

    # Encode positive prompt (or empty string if not provided)
    prompt = args.prompt if args.prompt else ""
    logger.info(
        f"Positive prompt: '{prompt[:50]}...' "
        if len(prompt) > 50
        else f"Positive prompt: '{prompt}'"
    )

    # Get T5 embeddings with Wan2.2-style trimming and zero-padding
    with torch.inference_mode():
        t5_encodings = encode_prompt_like_wan22(
            prompt, t5_tokenizer, t5_encoder, device, dtype, t5_seq_len
        )
        logger.info(f"T5 encodings shape (positive): {t5_encodings.shape}")

        # Encode negative prompt for CFG
        neg_prompt = args.neg_prompt
        logger.info(
            f"Negative prompt: '{neg_prompt[:50]}...' "
            if len(neg_prompt) > 50
            else f"Negative prompt: '{neg_prompt}'"
        )
        t5_encodings_null = encode_prompt_like_wan22(
            neg_prompt, t5_tokenizer, t5_encoder, device, dtype, t5_seq_len
        )
        logger.info(f"T5 encodings shape (negative): {t5_encodings_null.shape}")

    # Encode video with VAE directly (skip preprocess_data for simpler control)
    logger.info("Encoding video with VAE...")

    # Prepare video: [B, T, H, W, C] uint8 -> [B, C, T, H, W] float [-1, 1]
    videos = video_frames.to(device=device, dtype=dtype)
    videos = videos.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] -> [B, T, C, H, W]
    videos = videos * (2.0 / 255.0) - 1.0  # Normalize to [-1, 1]
    videos = videos.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]

    video_latents = wan_video_vae.encode(videos, device=device, tiled=False)
    video_latents = video_latents.to(device=device, dtype=dtype)

    logger.info(f"Video latents shape: {video_latents.shape}")
    logger.info(f"T5 encodings shape: {t5_encodings.shape}")

    # Run denoising
    logger.info("=" * 80)
    logger.info("Running denoising...")
    logger.info("=" * 80)
    logger.info(f"Num conditioning frames: {args.num_cond_frames}")
    logger.info(f"Denoising steps: {args.denoising_steps}")
    logger.info(f"CFG scale: {args.cfg_scale}")
    logger.info(f"Sigma shift: {args.sigma_shift}")
    logger.info(f"Sample solver: {args.sample_solver}")

    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with torch.inference_mode():
        latents = denoise(
            device=device,
            dtype=dtype,
            model=model,
            video_latents=video_latents,
            t5_encodings=t5_encodings,
            robot_states=None,
            num_cond_frames=args.num_cond_frames,
            num_inference_steps=args.denoising_steps,
            cfg_scale=args.cfg_scale,
            z_dim=video_latents.shape[1],
            img_height=target_size[0],
            img_width=target_size[1],
            num_frames=video_frames.shape[1],
            upsampling_factor=wan_video_vae.upsampling_factor,
            sigma_shift=args.sigma_shift,
            sample_solver=args.sample_solver,
            t5_encodings_null=t5_encodings_null,  # Negative prompt embedding for CFG
        )

        logger.info(f"Denoised latents shape: {latents.shape}")

        # ========== LATENT SPACE DIAGNOSTICS ==========
        logger.info("=" * 80)
        logger.info("LATENT SPACE DIAGNOSTICS")
        logger.info("=" * 80)

        num_cond_latents = (args.num_cond_frames - 1) // 4 + 1
        logger.info(f"Num conditioning latent frames: {num_cond_latents}")

        # Compare original vs denoised latents
        logger.info("\n--- Original video latents (input) ---")
        logger.info(f"  Shape: {video_latents.shape}")
        logger.info(
            f"  Min: {video_latents.min().item():.4f}, Max: {video_latents.max().item():.4f}"
        )
        logger.info(
            f"  Mean: {video_latents.mean().item():.4f}, Std: {video_latents.std().item():.4f}"
        )

        logger.info("\n--- Denoised latents (output) ---")
        logger.info(f"  Shape: {latents.shape}")
        logger.info(
            f"  Min: {latents.min().item():.4f}, Max: {latents.max().item():.4f}"
        )
        logger.info(
            f"  Mean: {latents.mean().item():.4f}, Std: {latents.std().item():.4f}"
        )

        # Check conditioning frames preservation
        cond_orig = video_latents[:, :, :num_cond_latents, :, :]
        cond_denoised = latents[:, :, :num_cond_latents, :, :]
        cond_mse = ((cond_orig - cond_denoised) ** 2).mean()

        logger.info("\n--- Conditioning latents comparison ---")
        logger.info(
            f"  Original cond - Min: {cond_orig.min().item():.4f}, Max: {cond_orig.max().item():.4f}"
        )
        logger.info(
            f"  Denoised cond - Min: {cond_denoised.min().item():.4f}, Max: {cond_denoised.max().item():.4f}"
        )
        logger.info(f"  MSE (should be ~0): {cond_mse.item():.8f}")

        # Check generated frames
        gen_orig = video_latents[:, :, num_cond_latents:, :, :]
        gen_denoised = latents[:, :, num_cond_latents:, :, :]
        gen_mse = ((gen_orig - gen_denoised) ** 2).mean()

        logger.info("\n--- Generated latents comparison ---")
        logger.info(
            f"  Original gen - Min: {gen_orig.min().item():.4f}, Max: {gen_orig.max().item():.4f}"
        )
        logger.info(
            f"  Denoised gen - Min: {gen_denoised.min().item():.4f}, Max: {gen_denoised.max().item():.4f}"
        )
        logger.info(f"  MSE: {gen_mse.item():.4f}")

        # Overall MSE
        total_mse = ((video_latents - latents) ** 2).mean()
        logger.info(
            f"\n--- Overall MSE between original and denoised: {total_mse.item():.4f} ---"
        )

        # Check for NaN/Inf
        if torch.isnan(latents).any():
            logger.error("  WARNING: Denoised latents contain NaN!")
        if torch.isinf(latents).any():
            logger.error("  WARNING: Denoised latents contain Inf!")
        logger.info("=" * 80)

        # Decode latents to video
        logger.info("Decoding latents to video...")
        generated_video = wan_video_vae.decode(latents, device=device)
        logger.info(f"Generated video shape: {generated_video.shape}")

        # Also decode original latents to verify VAE roundtrip
        logger.info("Decoding original latents for comparison...")
        reconstructed_video = wan_video_vae.decode(video_latents, device=device)
        logger.info(f"Reconstructed video shape: {reconstructed_video.shape}")

    # Save generated video
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = f"generated_v2v_{timestamp}.mp4"

    logger.info(f"Saving generated video to: {args.output_dir}/{video_name}")
    save_video(
        name=video_name,
        output_dir=args.output_dir,
        video=generated_video,
        add_sampling_metadata=True,
    )

    # Save reconstructed video (VAE roundtrip) for comparison
    recon_video_name = f"reconstructed_v2v_{timestamp}.mp4"
    logger.info(
        f"Saving reconstructed video (VAE roundtrip) to: {args.output_dir}/{recon_video_name}"
    )
    save_video(
        name=recon_video_name,
        output_dir=args.output_dir,
        video=reconstructed_video,
        add_sampling_metadata=True,
    )

    # Compute PSNR
    if not args.no_psnr:
        logger.info("=" * 80)
        logger.info("Computing PSNR...")
        logger.info("=" * 80)

        # Prepare ground truth video
        # video_frames: (B, T, H, W, C) uint8 [0, 255]
        # generated_video: (B, C, T, H, W) float [-1, 1]
        gt_video = video_frames.float()
        gt_video = (gt_video / 127.5) - 1.0  # Normalize to [-1, 1]
        gt_video = gt_video.to(device=device, dtype=dtype)
        gt_video = gt_video.permute(0, 4, 1, 2, 3)  # (B, T, H, W, C) -> (B, C, T, H, W)

        # Clamp generated video
        gen_video = generated_video.clamp(-1.0, 1.0)

        # Convert to CPU float32 for PSNR
        gt_cpu = gt_video.cpu().float()
        gen_cpu = gen_video.cpu().float()

        B, C, T, H, W = gt_cpu.shape

        # Compute frame-by-frame PSNR
        logger.info("Frame-by-frame PSNR:")
        psnr_values = []
        for t in range(T):
            gt_frame = gt_cpu[:, :, t, :, :]
            gen_frame = gen_cpu[:, :, t, :, :]
            psnr_frame = peak_signal_noise_ratio(
                gen_frame, gt_frame, data_range=2.0, reduction="none", dim=(1, 2, 3)
            )
            if psnr_frame.dim() == 0:
                psnr_frame = psnr_frame.unsqueeze(0)
            psnr_val = psnr_frame.mean().item()
            psnr_values.append(psnr_val)
            logger.info(f"  Frame {t:3d}: {psnr_val:.2f} dB")

        # Compute statistics
        avg_psnr = sum(psnr_values) / len(psnr_values)
        min_psnr = min(psnr_values)
        max_psnr = max(psnr_values)

        # Compute PSNR for generated frames only (excluding conditioning)
        gen_frame_psnr = psnr_values[args.num_cond_frames :]
        if gen_frame_psnr:
            avg_gen_psnr = sum(gen_frame_psnr) / len(gen_frame_psnr)
        else:
            avg_gen_psnr = 0.0

        logger.info("=" * 60)
        logger.info("PSNR Summary:")
        logger.info("=" * 60)
        logger.info(
            f"  All frames - Mean: {avg_psnr:.4f} dB, Min: {min_psnr:.2f} dB, Max: {max_psnr:.2f} dB"
        )
        logger.info(
            f"  Generated frames only (frames {args.num_cond_frames}-{T - 1}): {avg_gen_psnr:.4f} dB"
        )
        logger.info(f"  Last frame: {psnr_values[-1]:.4f} dB")
        logger.info("=" * 60)

    logger.info("Done!")


if __name__ == "__main__":
    main()
